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

    # 2. Compute the forces on each body required to achieve the accelerations
    net_forces = []
    for body, X_i, V_i, A_i in zip(rbt.bodies, body_poses, body_vels, body_accs):
        # Get the spatial inertia tensor of the body in the world frame
        I = body.inertia.transform(X_i).mat
        # Compute the force on the body. Featherstone (5.9)
        # TODO: add multiplication functions to SpatialInertiaTensor
        net_forces.append(SpatialForceVector(I @ A_i.vec + V_i.skew() @ I @ V_i.vec))


    # 3. Compute the force transmitted across each joint
    # Featherstone (5.20)
    joint_forces = [None for _ in rbt.bodies]

    for body in reversed(rbt.bodies):
        X_i = body_poses[body.idx]
        # Sum the forces that this body is transmitting to its children
        child_joint_forces = SpatialForceVector()
        for child in body.children:
            # Get the transform to the child
            X_i_child = joint_transform(child.joint, seg_q(child, q))
            # Express the child force in body_i coordinates
            child_joint_forces += X_i_child.inv() * joint_forces[child.idx]

        # Compute the force transmitted from the parent to this body
        joint_forces[body.idx] = net_forces[body.idx] - X_i.inv() * f_ext[body.idx] + child_joint_forces


    # TODO: investigate why this is not working

    # Convert the joint forces to generalized coordinates.  Featherstone (5.11)
    taus = []
    for body, X, f_j in zip(rbt.bodies, body_poses, joint_forces):
        taus.append(body.joint.S.T @ (X.inv() * f_j).vec)

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

    # print("C:\t", C)
    # print("H:\t", H)
    # print("tau:\t", tau)
    # Solve H * qdd = tau - C for qdd Featherstone (6.1)
    return jnp.linalg.solve(H, tau - C)
