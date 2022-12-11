import jax.numpy as jnp

from inertia import SpatialInertiaTensor
from joint import Joint, Free, Revolute
from kinematics import fk
from rbt import RigidBodyTree, Body
from transforms import (
    SpatialMotionVector,
    SpatialForceVector,
)
from misc_math import skew


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
                                          net_forces: list[SpatialForceVector],
                                          external_forces: list[SpatialForceVector]):
    """Compute the force transmitted across each joint, given the net force on
    each body and the external force on each body."""

    joint_forces = [None for _ in rbt.bodies]

    for body in reversed(rbt.bodies):
        # Sum the forces that this body is transmitting to its children
        child_joint_forces = sum((joint_forces[child.idx] for child in body.children), start=SpatialForceVector())
        # Compute the force transmitted from the parent to this body
        # Featherstone (5.10)
        joint_forces[body.idx] = net_forces[body.idx] - external_forces[body.idx] + child_joint_forces

    return joint_forces


def id(rbt, q, v, a, external_forces = None):
    """Inverse dynamics using the recursive Newton-Euler algorithm.
    See Featherstone section 5.3"""
    # If external forces are not supplied, assume they are zero for every body
    if external_forces is None:
        external_forces = [SpatialForceVector() for _ in rbt.bodies]
    # 1. Compute the velocity, and acceleration of each body
    _, spatial_vs, spatial_as = fk(rbt, q, v, a)
    # 2. Compute the forces on each body required to achieve the accelerations
    net_forces = [compute_body_force_from_accleration(body, s_v, s_a) for body, s_v, s_a in zip(rbt.bodies, spatial_vs, spatial_as)]
    # 3. Compute the force transmitted across each joint
    return compute_joint_forces_from_body_forces(rbt, net_forces, external_forces)


# def joint_wrench(joint: Joint, u: jnp.ndarray) -> jnp.ndarray:
#     """Get the wrench exerted by a joint for a given input u"""
#     assert u.shape == (joint.nf,)

#     if isinstance(joint, Revolute):
#         return jnp.array([0, 0, 0, 0, 0, u[0]])
#     elif isinstance(joint, Free):
#         return u
#     else:
#         return jnp.zeros(6)