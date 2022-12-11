from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import jax
import jax.numpy as jnp

from joint import Joint, Free
from misc_math import normalize


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


def make_q(model: Union[Joint, Body, RigidBodyTree],
           prng_key = None) -> jnp.ndarray:
    """Get a configuration vector for a model. If a prng_key is provided, the
    configuration will be random."""
    # Free joints are special because the orientation component is a unit
    # quaternion. The joint configuration is [qx, qy, qz, qw, tx, ty, tz]
    if isinstance(model, Free) and prng_key is None:
        return jnp.array([1, 0, 0, 0, 0, 0, 0])
    elif isinstance(model, Free):
        quat = jax.random.normal(prng_key, (4,))
        t = jax.random.normal(prng_key, (3,))
        return jnp.concatenate([normalize(quat), t])
    # Other joint's can be handled generically using the joint's nq field
    elif isinstance(model, Joint):
        return jnp.zeros(model.nq) if prng_key is None else jax.random.normal(prng_key, (model.nq,))
    elif isinstance(model, Body):
        return make_q(model.joint, prng_key)
    elif isinstance(model, RigidBodyTree):
        sub_qs = []
        for body in model.bodies:
            sub_qs.append(make_q(body.joint, prng_key))
            if prng_key is not None:
                prng_key, _ = jax.random.split(prng_key)

        return jnp.concatenate(sub_qs)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def make_v(model: Union[Joint, Body, RigidBodyTree],
           prng_key = None) -> jnp.ndarray:
    """Get a velocity vector for a model. If a prng_key is provided, the
    velocity will be random."""
    if isinstance(model, Joint):
        return jnp.zeros(model.nf) if prng_key is None else jax.random.normal(prng_key, (model.nf,))
    elif isinstance(model, Body):
        return make_v(model.joint, prng_key)
    elif isinstance(model, RigidBodyTree):
        # Get the total number of velocity variables
        nf = sum(b.joint.nf for b in model.bodies)
        # Create a velocity vector
        return jnp.zeros(nf) if prng_key is None else jax.random.normal(prng_key, (nf,))
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def seg_q(body: Body, q: jnp.ndarray) -> jnp.ndarray:
    """Get the segment of q corresponding to the body's joint"""
    return q[body.q_idx:body.q_idx + body.joint.nq]

def seg_v(body: Body, v: jnp.ndarray) -> jnp.ndarray:
    """Get the segment of v corresponding to the body's joint"""
    return v[body.v_idx:body.v_idx + body.joint.nf]

# Actuators correspond to the degrees of freedom of the robot
make_u = make_a = make_v
seg_u = seg_a = seg_v
