import jax
import jax.numpy as jnp
import pytest

from dynamics import *
from joint import *
from kinematics import *
from misc_math import *
from rbt import *
from transforms import *

def test_transforms():
    """Test that the transforms module works as expected by inverting and
    composing various transform types"""
    key_gen = prng_key_gen()
    for _ in range(10):
        q = jax.random.normal(next(key_gen), (4,))
        t = jax.random.normal(next(key_gen), (3,))
        q = q / jnp.linalg.norm(q)
        R = mat_from_quat(q)

        print("Check quat_from_mat and mat_from_quat")
        q2 = quat_from_mat(R)
        R2 = mat_from_quat(q2)
        assert jnp.isclose(jnp.linalg.norm(q2), 1)
        assert jnp.allclose(R, R2, atol=1e-5)

        print("Checking make_homogenous_transform")
        T = make_homogenous_transform(R, t)
        T_inv = make_homogenous_transform(R.T, -R.T @ t)
        assert jnp.allclose(T @ T_inv, jnp.eye(4), atol=1e-5)

        print("Checking SpatialTransform")
        X = SpatialTransform(R, t)
        X_2 = SpatialTransform(X.mat)
        assert jnp.allclose((X * X.inv()).mat, jnp.eye(6), atol=1e-5)
        assert jnp.allclose(X.inv().mat.T, X.tinv(), atol=1e-5)
        assert jnp.allclose(X.t, X_2.t, atol=1e-5)

        # print("Checking new SpatialTransforms")
        # X = make_spatial_R_t(R, t)
        # XX = X @ X
        # TT = T @ T
        # assert jnp.allclose(XX, make_spatial_R_t(TT[:3, :3], TT[:3, 3]), atol=1e-5)
        # X_inv = inv_spatial_R_t(R, t)
        # assert jnp.allclose(X @ X_inv, jnp.eye(6), atol=1e-5)
        # X_inv_2 = make_spatial_R_t(R.T, -R.T @ t)
        # assert jnp.allclose(X_inv, X_inv_2, atol=1e-5)


@pytest.mark.parametrize("j", [Fixed, Revolute, Free])
def test_joints(j):
    """Verify the implemented joint types"""
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


# @pytest.mark.skip("Refactoring")
def test_fk():
    """Define a simple single-pendulum and verify that the forward kinematics
    works as expected. The pendulum rotates around the world x-axis and begins
    pointed directly along the world y-axis."""
    l = 1.0
    T_w_j = SpatialTransform(y_rotation(jnp.pi/2), jnp.zeros(3))
    T_j_m = SpatialTransform(jnp.eye(3), jnp.array([0, l, 0]))

    rod = Body(0, Revolute(T_w_j), -1, "rod")
    end = Body(1, Fixed(T_j_m), 0, "end")

    rbt = RigidBodyTree([rod, end])

    # Check that the forward kinematics works as expected with zero inputs
    q = jnp.array([0.0])
    v = jnp.array([0.0])
    a = jnp.array([0.0])

    body_poses, body_vels, body_accs = fk(rbt, q, v, a)
    assert jnp.allclose(body_poses[0].mat, T_w_j.mat)
    assert jnp.allclose(body_poses[1].mat, T_w_j.mat @ T_j_m.mat)
    assert jnp.allclose(body_vels[0].vec, jnp.zeros((6,)))
    assert jnp.allclose(body_vels[1].vec, jnp.zeros((6,)))
    assert jnp.allclose(body_accs[0].vec, jnp.zeros((6,)))
    assert jnp.allclose(body_accs[1].vec, jnp.zeros((6,)))

    # Now set the velocity to 1 rad/s and check that the forward kinematics
    # works as expected
    v = jnp.array([1.0])
    body_poses, body_vels, body_accs = fk(rbt, q, v, a)
    assert jnp.allclose(body_poses[0].mat, T_w_j.mat)
    assert jnp.allclose(body_poses[1].mat, T_w_j.mat @ T_j_m.mat)

    # Convert the body-frame velocities to world-frame velocities. Note that we
    # don't consider the body translation, just the orientation.
    world_vels = [X_i.rotation() * v_i for X_i, v_i in zip(body_poses, body_vels)]
    assert jnp.allclose(world_vels[0].vec, jnp.array([1, 0, 0, 0, 0, 0]), atol=1e-6)
    assert jnp.allclose(world_vels[1].vec, jnp.array([1, 0, 0, 0, 0, 1]), atol=1e-6)


def test_id():
    """Define a simple single-pendulum and verify that the inverse dynamics
    works as expected. The pendulum rotates around the world x-axis and begins
    pointed directly along the world y-axis. There a point mass at the end of
    the pendulum."""

    l = 1.23
    m = 4.56
    T_w_j = SpatialTransform(y_rotation(jnp.pi/2), jnp.zeros(3))
    T_j_m = SpatialTransform(jnp.eye(3), jnp.array([0, l, 0]))

    rod = Body(0, Revolute(T_w_j), -1, "rod")
    end = Body(1, Fixed(T_j_m), 0, "end", SpatialInertiaTensor.from_m(m))

    rbt = RigidBodyTree([rod, end])

    # Check that the inverse dynamics works as expected with zero inputs
    q = jnp.array([0.0])
    v = jnp.array([0.0])
    a = jnp.array([0.0])
    f_ext = [SpatialForceVector() for _ in rbt.bodies]

    tau = id(rbt, q, v, a, f_ext)
    assert jnp.allclose(tau, jnp.zeros((1,)))

    # Now set the acceleration to 1 rad/s^2 and check that the inverse dynamics
    # finds the expected torque
    a = jnp.array([1.0])
    tau = id(rbt, q, v, a, f_ext)
    print(tau)
    assert jnp.allclose(tau, jnp.array([m * l**2]), atol=1e-6)

    # Now set the external force to apply a z-force of 1 N and check that the
    # inverse dynamics finds the expected torque
    f_ext = [SpatialForceVector(jnp.array([0,0,0,0,0,-1])) for _ in rbt.bodies]
    tau = id(rbt, q, v, a, f_ext)
    assert jnp.allclose(tau, jnp.array([l + m * l**2]), atol=1e-6)