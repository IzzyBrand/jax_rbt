from dataclasses import dataclass

import jax.numpy as jnp

from transforms import (
    z_rotation,
    mat_from_quat,
    SpatialTransform,
)


@dataclass
class Joint:
    # Transform from the parent body to the joint frame
    T_in: SpatialTransform = SpatialTransform()

    # Joint properties set by subclasses


class Fixed(Joint):
    # Size of the generalized coordinate vector for this joint
    nq: int = 0
    # Degrees of freedom of this joint
    nv: int = 0
    # Number of constraints imposed by this joint
    nc: int = 6
    # Number of degrees of actuation for this joint
    na: int = 0

    # Joint motion subspace matrix (6 x nv) maps generalized joint velocity to
    # relative spatial velocity of the parent and child bodies
    #
    #     v = S(q)q̇         Featherstone (3.33)
    #
    # where v is the velocity of the child body relative to the parent body.
    S = jnp.zeros((6, 0))
    # Constraint force subspace matrix (6 x nc)
    Tc = jnp.eye(6)
    # Active force subspace matrix (6 x na)
    Ta = jnp.zeros((6, 0))


class Revolute(Joint):
    nq: int = 1
    nv: int = 1
    nc: int = 5
    na: int = 1
    # See Featherstone Table 4.1
    S = jnp.array([[0, 0, 1, 0, 0, 0]]).T
    Tc = jnp.eye(6)[:, [0, 1, 3, 4, 5]]
    Ta = jnp.array([[0, 0, 1, 0, 0, 0]]).T


class Free(Joint):
    nq: int = 7
    nv: int = 6
    nc: int = 0
    na: int = 6
    S = jnp.eye(6)
    Tc = jnp.zeros((6, 0))
    Ta = jnp.eye(6)


def joint_transform(joint: Joint, q: jnp.ndarray) -> SpatialTransform:
    """Get the transform for a joint at position q."""
    assert q.shape == (joint.nq,)

    if isinstance(joint, Revolute):
        T = SpatialTransform(z_rotation(q[0]), jnp.zeros(3))
    elif isinstance(joint, Free):
        T = SpatialTransform(mat_from_quat(q[:4]), q[4:])
    elif isinstance(joint, Fixed):
        T = SpatialTransform()
    else:
        raise NotImplementedError(f"Joint type {type(joint)} not implemented")

    return joint.T_in * T


if __name__ == "__main__":
    # Verify joint types
    for j in [Fixed, Revolute, Free]:
        print("Checking", j.__name__)
        assert issubclass(j, Joint)

        # Make sure S has the right shape
        assert j.S.shape == (6, j.nv)
        # Make sure Tc has the right shape
        assert j.Tc.shape == (6, j.nc)
        # Make sure Ta has the right shape
        assert j.Ta.shape == (6, j.na)

        # Make sure the subspaces have the right dimensions
        # See Example 3.1 in Featherstone
        assert j.nc + j.na == 6
        assert j.nv == j.na

        # Check that the constraint and motion subspaces are orthogonal
        # Featherstone (3.36)
        assert jnp.allclose(j.Tc.T @ j.S, 0)

        # Check that the actuation and motion subpsaces are aligned
        # Featherstone (3.35)
        assert jnp.allclose(j.Ta.T @ j.S, jnp.eye(j.na))
