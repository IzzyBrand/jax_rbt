from functools import partial

import jax
import jax.numpy as jnp

from manifold import rbt_manifold_dyn
from rbt import RigidBodyTree, seg_q, seg_v
from joint import Joint, Fixed, Revolute, Free
from transforms import SO3_exp, mat_from_quat, quat_from_mat
from dynamics import fd_composite



# RK-4 method
def rk4(f, x0, dt):
    """Takes a single step of the RK4 method, where f is a function that does
    not depend on time. Returns the new state."""
    k1 = dt * f(x0)
    k2 = dt * f(x0 + 0.5*k1)
    k3 = dt * f(x0 + 0.5*k2)
    k4 = dt * f(x0 + k3)
    k = (k1 + 2*k2 + 2*k3 + k4)/6
    return x0 + k


def rbt_rk4(rbt: RigidBodyTree, q, v, tau, f_ext, dt):
    """Takes a single step of the RK4 method for a rigid body tree. Returns the
    new state."""
    def f(x):
        # X is a GroupManifold, and we need to break out the position and
        # velocity components
        q, v = (m.x for m in x.submanifolds)

        # Compute the dynamics acceleration
        a = fd_composite(rbt, q, v, tau, f_ext)

        # Return the vel and acc as a tangent vector of the manifold
        return jnp.concatenate([v, a])

    # Convert the state to a GroupManifold
    x = rbt_manifold_dyn(rbt, q, v)

    # Take a single step of the RK4 method
    x_new = rk4(f, x, dt)

    # Convert the new state back to a tuple of jnp arrays
    return (m.x for m in x_new.submanifolds)


@partial(jax.jit, static_argnames=['rbt'])
def euler_step(rbt: RigidBodyTree, q, v, a, dt):
    """Forward euler integration step."""
    # Different joints dtave different generalized positions and velocities,
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
