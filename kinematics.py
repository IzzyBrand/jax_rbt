import jax.numpy as jnp

from joint import joint_transform
from rbt import RigidBodyTree, seg_q, seg_v
from transforms import SpatialMotionVector, SpatialTransform


def fk(rbt: RigidBodyTree, q: jnp.ndarray, v: jnp.ndarray, a: jnp.ndarray):
    """Compute body pose, vel, acc given joint pose, vel, acc"""
    body_poses = []
    body_velocities = []
    body_accelerations = []

    for body in rbt.bodies:
        # Get the segment of q, v, and a corresponding to the body's joint
        q_joint = seg_q(body, q)
        v_joint = seg_v(body, v)
        a_joint = seg_v(body, a)

        # Compute, the pose, velocity, acceleration of the joint.
        # Assumes constant joint motion subspace and no bias velocity
        X_joint = joint_transform(body.joint, q_joint)
        v_joint = SpatialMotionVector(body.joint.S @ v_joint)  # (Featherstone 3.32
        a_joint = SpatialMotionVector(body.joint.S @ a_joint)  # (Featherstone 3.40)

        # Compute the pose of the body
        X_parent = body_poses[body.parent.idx] if body.parent else SpatialTransform()
        X_body = X_parent * X_joint
        body_poses.append(X_body)

        # Compute the velocity of the body (Featherstone 5.7)
        v_parent = body_velocities[body.parent.idx] if body.parent else SpatialMotionVector()
        v_body = v_parent + X_body.inv() * v_joint
        body_velocities.append(v_body)

        # Compute the acceleration of the body (Featherstone 5.8)
        a_parent = body_accelerations[body.parent.idx] if body.parent else SpatialMotionVector()
        a_body = a_parent + X_body.inv() * a_joint + v_body.cross(v_joint)
        body_accelerations.append(a_body)

    return body_poses, body_velocities, body_accelerations
