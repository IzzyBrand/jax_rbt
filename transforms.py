from __future__ import annotations
from collections import namedtuple


import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

###############################################################################
# Spatial Algebra
###############################################################################

@jax.jit
def smv_cross_smv(v1: jnp.ndarray, v2: jnp.ndarray):
    """Cross product of two spatial motion vectors."""
    w, v = v1.split(2)
    return jnp.block([[SO3_hat(w), jnp.zeros((3, 3))],
                      [SO3_hat(v), SO3_hat(w)]]) @ v2

# The following classes are not used currently, but were useful to sanity
# check how Featherstone's spatial algebra works.
@register_pytree_node_class
class SpatialMotionVector:
    def __init__(self, vec=None):
        self.vec = vec if vec is not None else jnp.zeros(6)
        assert self.vec.shape == (6,)

    def __add__(self, other):
        if isinstance(other, SpatialMotionVector):
            return SpatialMotionVector(self.vec + other.vec)
        else:
            raise NotImplementedError(f"SpatialMotionVector + {type(other)}")

    def skew(self):
        w, v = self.vec[:3], self.vec[3:]
        return jnp.block([[SO3_hat(w), jnp.zeros((3, 3))],
                          [SO3_hat(v), SO3_hat(w)]])

    def cross(self, other):
        if isinstance(other, SpatialMotionVector):
            return SpatialMotionVector(self.skew() @ other.vec)
        elif isinstance(other, SpatialForceVector):
            return SpatialForceVector(-self.skew().T @ other.vec)
        else:
            return super().cross(other)

    def __str__(self):
        return str(self.vec)

    def tree_flatten(self):
        return (self.vec,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])

@register_pytree_node_class
class SpatialForceVector:
    def __init__(self, vec=None):
        self.vec = vec if vec is not None else jnp.zeros(6)
        assert self.vec.shape == (6,)

    def __add__(self, other):
        if isinstance(other, SpatialForceVector):
            return SpatialForceVector(self.vec + other.vec)
        else:
            raise NotImplementedError(f"SpatialForceVector + {type(other)}")

    def __sub__(self, other):
        if isinstance(other, SpatialForceVector):
            return SpatialForceVector(self.vec - other.vec)
        else:
            raise NotImplementedError(f"SpatialForceVector - {type(other)}")

    def __str__(self):
        return str(self.vec)

    def tree_flatten(self):
        return (self.vec,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])

@register_pytree_node_class
class SpatialTransform:
    """A spatial transform is a 6x6 matrix that transforms spatial vectors.

    See Featherstone section 2.8 for details."""

    def __init__(self, *args):
        # Init from a rotation matrix and a translation vector
        if len(args) == 2:
            self.R, self.t = args
            # See Feathersteone (2.24)
            # ┌                  ┐
            # │ E              0 │
            # │ -E@SO3_hat(r)  E │
            # └                  ┘
            # where R = E and r = -E.T @ t
            r = -self.R.T @ self.t
            self.mat = jnp.block([[self.R, jnp.zeros((3, 3))],
                                  [-self.R @ SO3_hat(r), self.R]])

        # Init from a 6x6 matrix
        elif len(args) == 1:
            self.mat = args[0]
            # Extract R and t by inverting the above formula
            self.R = self.mat[:3, :3]
            self.t = self.R @ SO3_vee(self.R.T @ self.mat[3:, :3])

        # Default is the identity transform
        elif len(args) == 0:
            self.R, self.t = jnp.eye(3), jnp.zeros(3)
            self.mat = jnp.eye(6)

        else:
            raise NotImplementedError

    def rotation(self) -> SpatialTransform:
        """Return the rotation part of the transform."""
        return SpatialTransform(self.R, jnp.zeros(3))

    def inv(self) -> SpatialTransform:
        """Inverse of the transform."""
        return SpatialTransform(self.R.T, -self.R.T @ self.t)

    def tinv(self) -> jnp.ndarray:
        """Transpose of the inverse of the transform."""
        # Swap the top right and bottom left blocks. Featherstone (2.25)
        return jnp.block([[self.mat[:3, :3], self.mat[3:, :3]],
                          [self.mat[:3, 3:], self.mat[3:, 3:]]])

    def __mul__(self, other):
        if isinstance(other, SpatialTransform):
            # return SpatialTransform(self.mat @ other.mat)
            return SpatialTransform(self.R @ other.R,
                                    self.R @ other.t + self.t)
        elif isinstance(other, SpatialMotionVector):
            return SpatialMotionVector(self.mat @ other.vec)
        elif isinstance(other, SpatialForceVector):
            return SpatialForceVector(self.tinv() @ other.vec)
        elif isinstance(other, jnp.ndarray):
            return self.mat @ other
        else:
            raise NotImplementedError(f"SpatialTransform * {type(other)}")

    def __str__(self):
        return str(self.mat)

    def homogenous(self):
        """Return the 4x4 homogenous transform matrix"""
        return jnp.block([[self.R, self.t[:, None]],
                          [jnp.zeros((1, 3)), 1]])

    def tree_flatten(self):
        return (self.mat, self.R, self.t), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[1], children[2])

# The following functions are not used currently, but were useful to sanity
# check how Featherstone's spatial algebra works.

# def make_spatial_E_r(E, r):
#     return jnp.block([[E, jnp.zeros((3,3))],
#                       [-E @ SO3_hat(r), E]])

# def make_spatial_R_t(R, t):
#     r = -R.T @ t
#     return make_spatial_E_r(R, r)

# def make_homogenous_R_t(R, t):
#     return jnp.block([[R, t[:, None]],
#                       [jnp.zeros((1, 3)), 1]])

# def make_homogenous_E_r(E, r):
#     t = -E.T @ r
#     return make_homogenous_R_t(E, t)

# def inv_spatial_E_r(E, r):
#     return jnp.block([[E.T, jnp.zeros((3,3))],
#                       [SO3_hat(r) @ E.T, E.T]])

# def inv_spatial_R_t(R, t):
#     r = -R.T @ t
#     return inv_spatial_E_r(R, r)

# SpatialTransform = namedtuple("SpatialTransform", ["X", "R", "t"])

# def X_from_R_t(R, t):
#     # See Feathersteone (2.24)
#     # ┌                  ┐
#     # │ E              0 │
#     # │ -E@SO3_hat(r)  E │
#     # └                  ┘
#     # where R = E and r = -E.T @ t
#     r = -R.T @ t
#     X = jnp.block([[R, jnp.zeros((3, 3))],
#                    [-R @ SO3_hat(r), R]])

#     return SpatialTransform(X, R, t)

# def X_app_X(X1, X2):
#     pass

# def X_app_m(X, m):
#     return X.X @ m

# def X_app_f(X, f):
#     return X_inv(X).X.T @ f

# def X_app_I(X, I):
#     X_inv = X_inv(X).X
#     return X_inv.T @ I @ X_inv

# def X_inv(X):
#     return X_from_R_t(X.R.T, -X.R.T @ X.t)

# def X_inv_X(X1, X2):
#     pass

# def X_inv_m(X, m):
#     pass

# def X_inv_f(X, f):
#     pass

# def X_inv_I(X, I):
#     pass

###############################################################################
# SO3 lie group (rotation matrices)
###############################################################################

def x_rotation(theta: float) -> jnp.ndarray:
    return jnp.array([[1, 0, 0],
                      [0, jnp.cos(theta), -jnp.sin(theta)],
                      [0, jnp.sin(theta), jnp.cos(theta)]])

def y_rotation(theta: float) -> jnp.ndarray:
    return jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                      [0, 1, 0],
                      [-jnp.sin(theta), 0, jnp.cos(theta)]])

def z_rotation(theta: float) -> jnp.ndarray:
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0],
                      [jnp.sin(theta), jnp.cos(theta), 0],
                      [0, 0, 1]])

@jax.jit
def SO3_from_euler(euler: jnp.ndarray) -> jnp.ndarray:
    return z_rotation(euler[2]) @ y_rotation(euler[1]) @ x_rotation(euler[0])

@jax.jit
def SO3_hat(w: jnp.ndarray) -> jnp.ndarray:
    # A Micro Lie Theory (Example 3)
    wx, wy, wz = w
    S = jnp.array([[0, -wz, wy],
                   [wz, 0, -wx],
                   [-wy, wx, 0]])
    return S

@jax.jit
def SO3_vee(S: jnp.ndarray) -> jnp.ndarray:
    # A Micro Lie Theory (Example 3)
    w = jnp.array([S[2, 1], S[0, 2], S[1, 0]])
    return w

# @jax.jit
def SO3_exp(w: jnp.ndarray) -> jnp.ndarray:
    # A Micro Lie Theory (Example 4)
    # Compute the magnitude of the rotation
    θ = jnp.linalg.norm(w)
    # Avoid division by zero
    θ = jnp.where(θ, θ, 1e-8)
    S = SO3_hat(w)
    R = np.eye(3) \
      + S * np.sin(θ) / θ\
      + S @ S * (1.0 - np.cos(θ)) / θ**2
    return R

###############################################################################
# SE3 lie group (homogenous transforms)
###############################################################################

@jax.jit
def make_homogenous_transform(R: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Take a rotation matrix and a translation vector and return a homogeneous
    transformation matrix
        ┌      ┐
        │ R  t │
        │ 0  1 │
        └      ┘
    """
    return jnp.block([[R, t.squeeze()[:, None]], [jnp.zeros((1, 3)), 1]])


###############################################################################
# S3 lie group (quaternions)
###############################################################################

# From https://github.com/brentyi/jaxlie/blob/76c3042340d3db79854e4a5ca83f04dae0330079/jaxlie/_so3.py#L270-L281
@jax.jit
def mat_from_quat(quat: jnp.ndarray) -> jnp.ndarray:
    norm = quat @ quat
    q = quat * jnp.sqrt(2.0 / norm)
    q = jnp.outer(q, q)
    return jnp.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
        ]
    )

@jax.jit
def quat_from_mat(matrix: jnp.ndarray) -> jnp.ndarray:
    assert matrix.shape == (3, 3)

    # Modified from:
    # > "Converting a Rotation Matrix to a Quaternion" from Mike Day
    # > https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

    def case0(m):
        t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
        q = jnp.array(
            [
                m[2, 1] - m[1, 2],
                t,
                m[1, 0] + m[0, 1],
                m[0, 2] + m[2, 0],
            ]
        )
        return t, q

    def case1(m):
        t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
        q = jnp.array(
            [
                m[0, 2] - m[2, 0],
                m[1, 0] + m[0, 1],
                t,
                m[2, 1] + m[1, 2],
            ]
        )
        return t, q

    def case2(m):
        t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
        q = jnp.array(
            [
                m[1, 0] - m[0, 1],
                m[0, 2] + m[2, 0],
                m[2, 1] + m[1, 2],
                t,
            ]
        )
        return t, q

    def case3(m):
        t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
        q = jnp.array(
            [
                t,
                m[2, 1] - m[1, 2],
                m[0, 2] - m[2, 0],
                m[1, 0] - m[0, 1],
            ]
        )
        return t, q

    # Compute four cases, then pick the most precise one.
    # Probably worth revisiting this!
    case0_t, case0_q = case0(matrix)
    case1_t, case1_q = case1(matrix)
    case2_t, case2_q = case2(matrix)
    case3_t, case3_q = case3(matrix)

    cond0 = matrix[2, 2] < 0
    cond1 = matrix[0, 0] > matrix[1, 1]
    cond2 = matrix[0, 0] < -matrix[1, 1]

    t = jnp.where(
        cond0,
        jnp.where(cond1, case0_t, case1_t),
        jnp.where(cond2, case2_t, case3_t),
    )
    q = jnp.where(
        cond0,
        jnp.where(cond1, case0_q, case1_q),
        jnp.where(cond2, case2_q, case3_q),
    )

    # We can also choose to branch, but this is slower.
    # t, q = jax.lax.cond(
    #     matrix[2, 2] < 0,
    #     true_fun=lambda matrix: jax.lax.cond(
    #         matrix[0, 0] > matrix[1, 1],
    #         true_fun=case0,
    #         false_fun=case1,
    #         operand=matrix,
    #     ),
    #     false_fun=lambda matrix: jax.lax.cond(
    #         matrix[0, 0] < -matrix[1, 1],
    #         true_fun=case2,
    #         false_fun=case3,
    #         operand=matrix,
    #     ),
    #     operand=matrix,
    # )

    return q * 0.5 / jnp.sqrt(t)

@jax.jit
def qmul(q1, q0):
    """Multiply two quaternions."""
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return jnp.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                       x1*w0 + y1*z0 - z1*y0 + w1*x0,
                      -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                       x1*y0 - y1*x0 + z1*w0 + w1*z0])

