import jax.numpy as jnp
from jax.nn import one_hot

from inertia import SpatialInertiaTensor
from joint import Joint, Free, Revolute
from kinematics import fk
from rbt import RigidBodyTree, Body, make_v
from transforms import (
    SpatialMotionVector,
    SpatialForceVector,
    SO3_hat,
)


def compute_body_force_from_accleration(body: Body,
                                        v: SpatialMotionVector,
                                        a: SpatialMotionVector) -> SpatialForceVector:
    """Compute the force on a body required to achieve a given acceleration."""
    # Get the spatial inertia tensor of the body at the CoM
    spatial_inertia = SpatialInertiaTensor(body.inertia, body.mass)
    # Convert it to the body frame
    I = spatial_inertia.offset(body.com)
    # Compute the force on the body. Featherstone (5.9)
    # TODO: add multiplication functions to SpatialInertiaTensor
    return SpatialForceVector(I @ a.vec + v.skew() @ I @ v.vec)


def compute_joint_forces_from_body_forces(rbt: RigidBodyTree,
                                          net_forces,
                                          f_ext):
    """Compute the force transmitted across each joint, given the net force on
    each body and the external force on each body."""

    joint_forces = [None for _ in rbt.bodies]

    for body in reversed(rbt.bodies):
        # Sum the forces that this body is transmitting to its children
        child_joint_forces = sum((joint_forces[child.idx] for child in body.children), start=SpatialForceVector())
        # Compute the force transmitted from the parent to this body
        # Featherstone (5.10)
        joint_forces[body.idx] = net_forces[body.idx] - f_ext[body.idx] + child_joint_forces

    return joint_forces


def id(rbt, q, v, a, f_ext = None):
    """Inverse dynamics using the recursive Newton-Euler algorithm.
    See Featherstone section 5.3"""
    # If external forces are not supplied, assume they are zero for every body
    if f_ext is None:
        f_ext = [SpatialForceVector() for _ in rbt.bodies]

    # 1. Compute the velocity, and acceleration of each body
    _, spatial_vs, spatial_as = fk(rbt, q, v, a)
    # 2. Compute the forces on each body required to achieve the accelerations
    net_forces = [compute_body_force_from_accleration(body, s_v, s_a) for body, s_v, s_a in zip(rbt.bodies, spatial_vs, spatial_as)]
    # 3. Compute the force transmitted across each joint
    joint_forces = compute_joint_forces_from_body_forces(rbt, net_forces, f_ext)

    # Convert the joint forces to generalized coordinates.  Featherstone (5.11)
    return jnp.concatenate([b.joint.S.T @ f_j.vec for b, f_j in zip(rbt.bodies, joint_forces)])


def fd_differential(rbt, q, v, tau, f_ext=None):
    """Forward dynamics using the differential algorithm.
    See Featherstone section 6.1"""
    # Calculate the joint space bias force by computing the inverse dynamics
    # with zero acceleration. Featherstone (6.2)
    C = id(rbt, q, v, make_v(rbt))

    # Calculate the joint space inertia matrix by using differential inverse
    # dynamics. Featherstone (6.4)
    def id_differential(alpha):
        return id(rbt, q, v, one_hot(alpha, tau.shape[0]), f_ext) - C

    H = jnp.stack([id_differential(alpha) for alpha in range(tau.shape[0])]).T

    # Solve H * qdd = tau - C for qdd Featherstone (6.1)
    return jnp.linalg.solve(H, tau - C)
