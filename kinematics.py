from functools import partial

import jax
import jax.numpy as jnp

from joint import joint_transform
from rbt import RigidBodyTree, seg_q, seg_v
from transforms import SpatialMotionVector, SpatialTransform, smv_cross_smv


@partial(jax.jit, static_argnames=['rbt'])
def fk(rbt: RigidBodyTree, q: jnp.ndarray, v: jnp.ndarray, a: jnp.ndarray):
    """Forward kinematics for a rigid body tree.

    This function is jitted each time it is invoked with a different RigidBodyTree.
     - Before jitting, FK for a 5-dof arm takes 30.5 ms.
     - After jitting, FK for a 5-dof arm takes 1.7 ms.

    Given homogenous coordinate vectors
        q: [nq, 1] the joint positions
        v: [nv, 1] the joint velocities
        a: [nv, 1] the joint accelerations

    Compute the pose, velocity, and acceleration of each body in the tree.
        Poses are given as SpatialTransforms X_world_body
        Velocities are given as SpatialMotionVectors in the body frame
        Accelerations are given as SpatialMotionVectors in the body frame
    """

    # NOTE: The root body does not have a parent, but rather than use an if
    # statement, we just add a dummy body at the start of the list, and shift
    # the indices of the other bodies by 1.

    body_poses = [SpatialTransform()]
    body_velocities = [SpatialMotionVector()]
    body_accelerations = [SpatialMotionVector()]

    for body in rbt.bodies:
        # Get the segment of q, v, and a corresponding to the body's joint
        q_i = seg_q(body, q)
        v_i = seg_v(body, v)
        a_i = seg_v(body, a)

        # Compute the joint transform, velocity, and acceleration. Note that
        # the velocity and acceleration are in the body frame.
        X_j = joint_transform(body.joint, q_i)
        v_j = SpatialMotionVector(body.joint.S @ v_i)
        a_j = SpatialMotionVector(body.joint.S @ a_i)

        # Get the pose of the parent body
        X_p = body_poses[body.parent_idx + 1]
        # Compute the pose of the body
        X_i = X_p * X_j

        # Get the velocity of the parent in the parent frame
        v_p = body_velocities[body.parent_idx + 1]
        # Compute the velocity of the body in the body frame (Featherstone 5.14)
        v_i = X_j.inv() * v_p + v_j

        # Get the acceleration of the parent
        a_p = body_accelerations[body.parent_idx + 1]
        # Compute the acceleration of the body in the body frame (Featherstone 5.15)
        # Note that the joint motion subspace is constant, so we can omit the
        # term involving its time derivative.
        a_i = X_j.inv() * a_p + a_j + v_i.cross(v_j)

        body_poses.append(X_i)
        body_velocities.append(v_i)
        body_accelerations.append(a_i)

    # NOTE: Drop the dummy body before returning
    return body_poses[1:], body_velocities[1:], body_accelerations[1:]


# @jax.jit
# def _compose_joint_terms(parent_idxs, Xjs, Vjs, Ajs):
#     # NOTE: The root body does not have a parent, but rather than use an if
#     # statement, we just add a dummy body at the start of the list, and shift
#     # the indices of the other bodies by 1.

#     body_poses = [jnp.eye(6)]
#     body_velocities = [jnp.zeros(6)]
#     body_accelerations = [jnp.zeros(6)]

#     for parent_idx, Xj, Vj, Aj in zip(parent_idxs, Xjs, Vjs, Ajs):
#         # Get the pose of the parent
#         X_world_parent = body_poses[parent_idx + 1]
#         # Compute the pose of the body
#         X_body = X_world_parent @ Xj
#         body_poses.append(X_body)

#         # Get the velocity of the parent
#         v_parent = body_velocities[parent_idx + 1]
#         # Compute the velocity of the body (Featherstone 5.7)
#         v_body = v_parent + X_body @ Vj
#         body_velocities.append(v_body)

#         # Get the acceleration of the parent
#         a_parent = body_accelerations[parent_idx + 1]
#         # Compute the acceleration of the body (Featherstone 5.4)
#         a_body = a_parent + Aj + smv_cross_smv(v_body, Vj)
#         body_accelerations.append(a_body)

#     # NOTE: Drop the dummy body before returning
#     return body_poses[1:], body_velocities[1:], body_accelerations[1:]


# def fk_fast(rbt: RigidBodyTree, q: jnp.ndarray, v: jnp.ndarray, a: jnp.ndarray):
#     """Compute body pose, vel, acc given joint pose, vel, acc"""

#     # Compute, the pose, velocity, acceleration of each joint
#     Xjs = [joint_transform(b.joint, seg_q(b, q)).mat for b in rbt.bodies]
#     Vjs = [b.joint.S @ seg_v(b, v) for b in rbt.bodies]
#     Ajs = [b.joint.S @ seg_v(b, a) for b in rbt.bodies]

#     # Compose the joint terms
#     return _compose_joint_terms(rbt.parent_idxs, Xjs, Vjs, Ajs)
