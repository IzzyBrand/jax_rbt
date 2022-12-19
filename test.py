import jax
import jax.numpy as jnp
import pytest

from misc_math import *
from joint import *
from transforms import *

def test_transforms():
    key_gen = prng_key_gen()
    for _ in range(10):
        q = jax.random.normal(next(key_gen), (4,))
        t = jax.random.normal(next(key_gen), (3,))
        q = q / jnp.linalg.norm(q)
        R = mat_from_quat(q)

        print("Check quat_from_mat and mat_from_quat")
        q2 = quat_from_mat(R)
        assert jnp.isclose(jnp.linalg.norm(q2), 1)
        R2 = mat_from_quat(q2)
        assert jnp.allclose(R, R2, atol=1e-5)

        print("Checking make_homogenous_transform")
        T = make_homogenous_transform(R, t)
        T_inv = make_homogenous_transform(R.T, -R.T @ t)
        assert jnp.allclose(T @ T_inv, jnp.eye(4), atol=1e-5)

        print("Checking SpatialTransform")
        X = SpatialTransform(R, t)
        assert jnp.allclose((X * X.inv()).mat, jnp.eye(6), atol=1e-5)


@pytest.mark.parametrize("j", [Fixed, Revolute, Free])
def test_joints(j):
    # Verify joint types
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