from typing import Union

import jax.numpy as jnp

from joint import Joint, Fixed, Revolute, Free, joint_transform
from rbt import RigidBodyTree, Body
from transforms import (
    make_homogenous_transform,
    x_rotation,
    SpatialMotionVector,
    SpatialForceVector,
    SpatialTransform,
)


def zero_q(model: Union[Joint, Body, RigidBodyTree]) -> jnp.ndarray:
    """Get a zero configuration for a model."""
    if isinstance(model, Free):
        return jnp.array([1, 0, 0, 0, 0, 0, 0])
    elif isinstance(model, Joint):
        return jnp.zeros(joint.nq)
    elif isinstance(model, Body):
        return zero_q(model.joint)
    elif isinstance(model, RigidBodyTree):
        return jnp.concatenate([zero_q(b.joint) for b in model.bodies])
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def seg_q(body: Body, q: jnp.ndarray) -> jnp.ndarray:
    """Get the segment of q corresponding to the body's joint"""
    return q[body.q_idx:body.q_idx + body.joint.nq]

def seg_v(body: Body, v: jnp.ndarray) -> jnp.ndarray:
    """Get the segment of v corresponding to the body's joint"""
    return v[body.idx:body.idx + body.joint.nf]

# Actuators correspond to the degrees of freedom of the robot
seg_u = seg_a = seg_v


def fk(rbt: RigidBodyTree, q: jnp.ndarray, v: jnp.ndarray, a: jnp.ndarray):
    """Compute body poses, velocities, and accelerations given q, qd, qdd."""
    body_poses = []
    body_velocities = []
    body_accelerations = []

    for body in rbt.bodies:
        # Get the segment of q, v, and a corresponding to the body's joint
        q_joint = seg_q(body, q)
        v_joint = seg_v(body, v)
        a_joint = seg_a(body, a)

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


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    t_z = jnp.array([0, 0, 0.1]);
    theta = jnp.pi / 6
    # T_body_i_joint is offset by t_z rotated around x by theta
    T_in = SpatialTransform(x_rotation(theta), t_z)
    # Create the revolute joint
    joint = Revolute(T_in)

    # Create the bodies
    bodies = [Body(0, Fixed(), None, "base")]
    for i in range(1, 7):
        bodies.append(Body(i, joint, i - 1, f"body_{i}"))

    # Create the tree
    rbt = RigidBodyTree(bodies)

    # Do forward kinematics
    q0 = zero_q(rbt)
    poses, _, _ = fk(rbt, q0, jnp.zeros_like(q0), jnp.zeros_like(q0))

    # Plot the positions of each body
    for pose in poses:
        plt.scatter(pose.t[1], pose.t[2], s=100)
    plt.show()
