import jax
import jax.numpy as jnp
from jax.nn import one_hot

from inertia import SpatialInertiaTensor
from joint import Joint, Free, Revolute, joint_transform
from kinematics import fk
from rbt import RigidBodyTree, Body, make_v, seg_q
from transforms import (
    SpatialMotionVector,
    SpatialForceVector,
    SpatialTransform,
    SO3_hat,
)


def id(rbt, q, v, a, f_ext) -> jnp.ndarray:
    """Inverse dynamics using the recursive Newton-Euler algorithm.

    Returns the joint torques required to achieve the given accelerations.
    See Featherstone section 5.3"""

    # 1. Compute the velocity, and acceleration of each body
    body_poses, body_vels, body_accs = fk(rbt, q, v, a)

    # 2. Compute the forces on each body (in the body frame) required to achieve
    #    the accelerations
    body_forces = []
    for body, X_i, V_i, A_i in zip(rbt.bodies, body_poses, body_vels, body_accs):
        # Get the spatial inertia tensor of the body
        I = body.inertia.mat
        # Compute the force on the body to produce the acceleration. Featherstone (5.9)
        # TODO: add multiplication functions to SpatialInertiaTensor
        f_a = SpatialForceVector(I @ A_i.vec + V_i.skew() @ I @ V_i.vec)
        # Remove the external force from the body
        f_x = X_i.rotation().inv() * f_ext[body.idx]
        body_forces.append(f_a - f_x)

    # 3. Compute the force transmitted across each joint. Featherstone (5.20)
    taus = []
    for body in reversed(rbt.bodies):
        f_body = body_forces[body.idx]
        # Convert the forces to generalized coordinates. Featherstone (5.11)
        taus.append(body.joint.S.T @ f_body.vec)
        # If the body has a parent, apply the force to the parent
        if body.parent_idx != -1:
            # Get the transform from the parent to this body
            X_parent_body = joint_transform(body.joint, seg_q(body, q))
            # Apply the force from the body to the parent
            body_forces[body.parent_idx] += X_parent_body * f_body

    return jnp.concatenate(taus)


def fd_differential(rbt, q, v, tau, f_ext):
    """Forward dynamics using the differential algorithm.
    See Featherstone section 6.1"""
    # Calculate the joint space bias force by computing the inverse dynamics
    # with zero acceleration. Featherstone (6.2)
    C = id(rbt, q, v, make_v(rbt), f_ext)

    # Calculate the joint space inertia matrix, H, by using differential inverse
    # dynamics. Featherstone (6.4)
    def id_differential(alpha):
        return id(rbt, q, v, one_hot(alpha, tau.shape[0]), f_ext) - C

    H = jnp.stack([id_differential(alpha) for alpha in range(tau.shape[0])]).T

    # Solve H * qdd = tau - C for qdd Featherstone (6.1)
    return jnp.linalg.solve(H, tau - C)


def fd_composite(rbt, q, v, tau, f_ext):
    """Forward dynamics using the composite rigid body algorithm.
    See Featherstone section 6.2"""

    # TODO(@ib): store the velocity dimension in the RigidBodyTree
    nv = sum([b.joint.nv for b in rbt.bodies])
    H = jnp.zeros((nv, nv))

    # Init the composite inertia with the inertia of each body
    I_c = [b.inertia for b in rbt.bodies]

    for body in rbt.bodies:
        if body.parent_idx != -1:
            X_parent_body = joint_transform(body.joint, seg_q(body, q))
            I_c[body.parent_idx] += I_c[body.idx].transform(X_parent_body)

        F = I_c[body].mat @ body.joint.S