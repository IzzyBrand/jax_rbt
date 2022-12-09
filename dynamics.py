import jax.numpy as jnp

from joint import Joint, Free, Revolute
from rbt import RigidBodyTree, Body
from util import skew


def spatial_inertia(body: Body) -> jnp.ndarray:
    """Get the spatial inertia matrix of a body at the body frame.
    Featherstone (2.62), (2.63)"""
    I = body.inertia
    m = body.mass

    if (body.com == 0).all():
        # If the center of mass is at the body origin, then the spatial inertia
        # has a simpler form.
        return jnp.block([[I, jnp.zeros((3, 3))],
                          [jnp.zeros((3, 3)), m * jnp.eye(3)]])
    else:
        # If the center of mass is not at the body origin, then we have to
        # consider the offset.
        c_cross = skew(body.com)
        return jnp.block([
            [I + m * c_cross @ c_cross.T, m * c_cross],
            [m * c_cross.T, m * jnp.eye(3)]])


def joint_wrench(joint: Joint, u: jnp.ndarray) -> jnp.ndarray:
    """Get the wrench exerted by a joint for a given input u"""
    assert u.shape == (joint.nf,)

    if isinstance(joint, Revolute):
        return jnp.array([0, 0, 0, 0, 0, u[0]])
    elif isinstance(joint, Free):
        return u
    else:
        return jnp.zeros(6)