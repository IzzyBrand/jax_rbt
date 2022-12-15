import jax.numpy as jnp

from rbt import RigidBodyTree, seg_q, seg_v
from joint import Joint, Fixed, Revolute, Free
from transforms import SO3_exp, mat_from_quat, quat_from_mat



def euler_step(rbt: RigidBodyTree, q, v, a, dt):
    """Euler integration step"""
    # Different joints have different generalized positions and velocities,
    # so we need to update each joint position individually
    new_qs = []
    for body in rbt.bodies:
        q_j = seg_q(body, q)
        v_j = seg_v(body, v)
        # We need to handle Free joints differently because of the quaternion
        if isinstance(body.joint, Free):
            # Integrate the angular velocity using the exponential map
            R = mat_from_quat(q_j[:4]) @ SO3_exp(v_j[:3] * dt)
            # Integrate the linear velocity
            t = q_j[4:] + v_j[3:] * dt
            # Convert the rotation matrix back to a quaternion and concatenate
            new_qs.append(jnp.concatenate([quat_from_mat(R), t]))
        else:
            new_qs.append(q_j + v_j * dt)

    new_q = jnp.concatenate(new_qs)

    # Integrating the velocity is easier
    new_v = v + a * dt

    return new_q, new_v