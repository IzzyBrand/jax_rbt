import jax
import jax.numpy as jnp

from joint import joint_transform
from rbt import RigidBodyTree, seg_q, seg_v
from transforms import SpatialMotionVector, SpatialTransform, smv_cross_smv




def fk(rbt: RigidBodyTree, q: jnp.ndarray, v: jnp.ndarray, a: jnp.ndarray):
    """Compute body pose, vel, acc given joint pose, vel, acc"""

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

        # Compute, the pose, velocity, acceleration of the joint.
        # Assumes constant joint motion subspace and no bias velocity

        # Get the pose of the parent
        X_world_parent = body_poses[body.parent_idx + 1]
        # Compute the joint transform
        X_parent_child = joint_transform(body.joint, q_i)
        # Compute the pose of the body
        X_body = X_world_parent * X_parent_child
        body_poses.append(X_body)

        # Get the velocity of the parent
        v_parent = body_velocities[body.parent_idx + 1]
        # Compute the joint velocity in world frame (Featherstone 3.32)
        v_joint = X_body * SpatialMotionVector(body.joint.S @ v_i)
        # Compute the velocity of the body (Featherstone 5.7)
        v_body = v_parent + v_joint
        body_velocities.append(v_body)

        # Get the acceleration of the parent
        a_parent = body_accelerations[body.parent_idx + 1]
        # Compute the joint acceleration in world frame (Featherstone 3.40)
        a_joint = X_body * SpatialMotionVector(body.joint.S @ a_i)
        # Compute the acceleration of the body (Featherstone 5.4)
        a_body = a_parent + a_joint + v_body.cross(v_joint)
        body_accelerations.append(a_body)

    # NOTE: Drop the dummy body before returning
    return body_poses[1:], body_velocities[1:], body_accelerations[1:]


@jax.jit
def _compose_joint_terms(parent_idxs, Xjs, Vjs, Ajs):
    # NOTE: The root body does not have a parent, but rather than use an if
    # statement, we just add a dummy body at the start of the list, and shift
    # the indices of the other bodies by 1.

    body_poses = [jnp.eye(6)]
    body_velocities = [jnp.zeros(6)]
    body_accelerations = [jnp.zeros(6)]

    for parent_idx, Xj, Vj, Aj in zip(parent_idxs, Xjs, Vjs, Ajs):
        # Get the pose of the parent
        X_world_parent = body_poses[parent_idx + 1]
        # Compute the pose of the body
        X_body = X_world_parent @ Xj
        body_poses.append(X_body)

        # Get the velocity of the parent
        v_parent = body_velocities[parent_idx + 1]
        # Compute the velocity of the body (Featherstone 5.7)
        v_body = v_parent + X_body @ Vj
        body_velocities.append(v_body)

        # Get the acceleration of the parent
        a_parent = body_accelerations[parent_idx + 1]
        # Compute the acceleration of the body (Featherstone 5.4)
        a_body = a_parent + Aj + smv_cross_smv(v_body, Vj)
        body_accelerations.append(a_body)

    # NOTE: Drop the dummy body before returning
    return body_poses[1:], body_velocities[1:], body_accelerations[1:]


def fk_fast(rbt: RigidBodyTree, q: jnp.ndarray, v: jnp.ndarray, a: jnp.ndarray):
    """Compute body pose, vel, acc given joint pose, vel, acc"""

    # Compute, the pose, velocity, acceleration of each joint
    Xjs = [joint_transform(b.joint, seg_q(b, q)).mat for b in rbt.bodies]
    Vjs = [b.joint.S @ seg_v(b, v) for b in rbt.bodies]
    Ajs = [b.joint.S @ seg_v(b, a) for b in rbt.bodies]

    # Compose the joint terms
    return _compose_joint_terms(rbt.parent_idxs, Xjs, Vjs, Ajs)
