import jax.numpy as jnp

from inertia import SpatialInertiaTensor
from joint import Joint, Free, Revolute
from rbt import RigidBodyTree, Body
from transforms import (
    SpatialMotionVector,
    SpatialForceVector,
)
from misc_math import skew


def compute_force_on_body(body: Body,
                          v: SpatialMotionVector,
                          a: SpatialMotionVector) -> SpatialForceVector:
    """Compute the force on a body given its velocity and acceleration."""
    # Get the spatial inertia tensor of the body at the CoM
    spatial_inertia = SpatialInertiaTensor(body.inertia, body.mass)
    # Convert it to the body frame
    I = spatial_inertia.offset(body.com)
    # Compute the force on the body (Featherstone 5.9)
    # TODO: add multiplication functions to SpatialInertiaTensor
    return SpatialForceVector(I @ a.vec + v.skew() @ I @ v.vec)


def compute_joint_forces(tree: RigidBodyTree,
                         net_forces: list[SpatialForceVector],
                         external_forces: list[SpatialForceVector]):
    """Compute the force transmitted across each joint."""

    joint_forces = [None] * tree.n_bodies

    for body in reversed(tree.bodies):
        i = body.idx
        # Sum the forces that this body is transmitting to its children
        child_joint_forces = sum(joint_forces[child.idx] for child in body.children)
        # Compute the force transmitted from the parent to this body
        # Featherstone (5.10)
        joint_forces[i] = net_forces[i] - external_forces[i] + child_joint_forces

    return joint_forces


def joint_wrench(joint: Joint, u: jnp.ndarray) -> jnp.ndarray:
    """Get the wrench exerted by a joint for a given input u"""
    assert u.shape == (joint.nf,)

    if isinstance(joint, Revolute):
        return jnp.array([0, 0, 0, 0, 0, u[0]])
    elif isinstance(joint, Free):
        return u
    else:
        return jnp.zeros(6)