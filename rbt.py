from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp

from joint import Joint, Free


@dataclass
class Body:
    id: int
    joint: Joint = Free()
    parent_id: Optional[int] = None
    name: Optional[str] = None

    # Inertial properties
    inertia: jnp.ndarray = jnp.zeros((3, 3))
    mass: float = 0.0
    com: jnp.ndarray = jnp.zeros(3)

    # These fields are set by the RigidBodyTree
    parent: Optional[Body] = None
    children: Optional[list[Body]] = None

    idx: Optional[int] = None
    q_idx: Optional[int] = None
    v_idx: Optional[int] = None


class RigidBodyTree:

    def __init__(self, bodies):
        """Some setup steps

        1. Store a map from id to body
        2. Set the root, parents and children of each body
        3. Re-order bodies list so children come after parents
        4. Set the idx, q_idx, v_idx fields
        """
        # Store a map from id to body
        self.id_to_body = {b.id: b for b in bodies}

        # Update the root and children of each body
        for body in bodies:
            body.children = []
            if body.parent_id is None:
                self.root = body
            else:
                parent_body = self.id_to_body[body.parent_id]
                body.parent = parent_body
                parent_body.children.append(body)

        # Re-order the bodies so that the indices of the children are always
        # greater than the parent. This is called "Regular Numbering" in
        # Featherstone section 4.1.2
        self.bodies = []

        # Define a helper function to recur throught the children depth-first
        def add_to_bodies(body):
            self.bodies.append(body)
            for child in body.children:
                add_to_bodies(child)

        # Call the helper function on the root
        add_to_bodies(self.root)

        # Update the indices for each body
        idx = 0
        q_idx = 0
        v_idx = 0
        for body in self.bodies:
            body.idx = idx
            body.q_idx = q_idx
            body.v_idx = v_idx
            idx += 1
            q_idx += body.joint.nq
            v_idx += body.joint.nf


def zero_q(model: Union[Joint, Body, RigidBodyTree]) -> jnp.ndarray:
    """Get a zero configuration for a model."""
    if isinstance(model, Free):
        return jnp.array([1, 0, 0, 0, 0, 0, 0])
    elif isinstance(model, Joint):
        return jnp.zeros(model.nq)
    elif isinstance(model, Body):
        return zero_q(model.joint)
    elif isinstance(model, RigidBodyTree):
        return jnp.concatenate([zero_q(b.joint) for b in model.bodies])
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def zero_v(model: Union[Joint, Body, RigidBodyTree]) -> jnp.ndarray:
    """Get a zero velocity for a model."""
    if isinstance(model, Joint):
        return jnp.zeros(model.nf)
    elif isinstance(model, Body):
        return zero_v(model.joint)
    elif isinstance(model, RigidBodyTree):
        return jnp.concatenate([zero_v(b.joint) for b in model.bodies])
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def seg_q(body: Body, q: jnp.ndarray) -> jnp.ndarray:
    """Get the segment of q corresponding to the body's joint"""
    return q[body.q_idx:body.q_idx + body.joint.nq]

def seg_v(body: Body, v: jnp.ndarray) -> jnp.ndarray:
    """Get the segment of v corresponding to the body's joint"""
    return v[body.idx:body.idx + body.joint.nf]

# Actuators correspond to the degrees of freedom of the robot
zero_u = zero_a = zero_v
seg_u = seg_a = seg_v
