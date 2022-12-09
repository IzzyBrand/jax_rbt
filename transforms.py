import jax.numpy as jnp

from util import skew, deskew, normalize

###############################################################################
# Spatial Algebra
###############################################################################

class SpatialMotionVector:
    def __init__(self, vec=None):
        self.vec = vec if vec is not None else jnp.zeros(6)
        assert self.vec.shape == (6,)

    def __add__(self, other):
        if isinstance(other, SpatialMotionVector):
            return SpatialMotionVector(self.vec + other.vec)
        else:
            raise NotImplementedError

    def skew(self):
        w, v = self.vec[:3], self.vec[3:]
        return jnp.block([[skew(w), jnp.zeros((3, 3))],
                          [skew(v), skew(w)]])

    def cross(self, other):
        if isinstance(other, SpatialMotionVector):
            return SpatialMotionVector(self.skew() @ other.vec)
        elif isinstance(other, SpatialForceVector):
            return SpatialForceVector(-self.skew().T @ other.vec)
        else:
            return super().cross(other)


class SpatialForceVector:
    def __init__(self, vec=None):
        self.vec = vec if vec is not None else jnp.zeros(6)
        assert self.vec.shape == (6,)

    def __add__(self, other):
        if isinstance(other, SpatialForceVector):
            return SpatialForceVector(self.vec + other.vec)
        else:
            raise NotImplementedError


class SpatialTransform:
    """A spatial transform is a 6x6 matrix that transforms spatial vectors.

    See Featherstone section 2.8 for details."""

    def __init__(self, *args):
        # Init from a rotation matrix and a translation vector
        if len(args) == 2:
            self.R, self.t = args
            # See Feathersteone (2.24)
            # ┌               ┐
            # │ R           0 │
            # │ -R@skew(t)  R │
            # └               ┘
            self.mat = jnp.block([[self.R, jnp.zeros((3, 3))],
                                  [-self.R @ skew(self.t), self.R]])

        # Init from a 6x6 matrix
        elif len(args) == 1:
            self.R, self.t = None, None
            self.mat = args[0]
            assert self.mat.shape == (6, 6)

        # Default is the identity transform
        elif len(args) == 0:
            self.R, self.t = jnp.eye(3), jnp.zeros(3)
            self.mat = jnp.eye(6)

        else:
            raise NotImplementedError

    def inv(self):
        """Inverse of the transform."""
        if self.R is not None:
            # See Featherstone (2.26)
            return SpatialTransform(jnp.block([
                [self.R.T, jnp.zeros((3, 3))],
                [skew(self.t) @ self.R.T, self.R.T]]))
        else:
            return SpatialTransform(jnp.linalg.inv(self.mat))

    def __mul__(self, other):
        if isinstance(other, SpatialTransform):
            if self.R is None and other.R is None:
                return SpatialTransform(self.mat @ other.mat)
            else:
                return SpatialTransform(self.R @ other.R,
                                        self.R @ other.t + self.t)
        elif isinstance(other, SpatialMotionVector):
            return SpatialMotionVector(self.mat @ other.vec)
        elif isinstance(other, SpatialForceVector):
            return SpatialForceVector(self.inv().mat.T @ other.vec)
        elif isinstance(other, jnp.ndarray):
            return self.mat @ other
        else:
            raise NotImplementedError

    def __str__(self):
        return str(self.mat)

###############################################################################
# Homogeneous transforms
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

def mat_from_euler(euler: jnp.ndarray) -> jnp.ndarray:
    return z_rotation(euler[2]) @ y_rotation(euler[1]) @ x_rotation(euler[0])


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
# Quaternions
###############################################################################

# From https://github.com/brentyi/jaxlie/blob/76c3042340d3db79854e4a5ca83f04dae0330079/jaxlie/_so3.py#L270-L281
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


if __name__ == "__main__":
    mat = jnp.eye(3)
    quat = quat_from_mat(mat)
    mat2 = mat_from_quat(quat)
    assert jnp.linalg.norm(quat) == 1
    assert jnp.allclose(mat, mat2)

    R = mat_from_euler(jnp.array([1,2,3]))
    t = jnp.array([1,2,3])

    T = make_homogenous_transform(R, t)
    T_inv = make_homogenous_transform(R.T, -R.T @ t)
    assert jnp.allclose(T @ T_inv, jnp.eye(4), atol=1e-6)

    X = SpatialTransform(R, t)
    assert jnp.allclose((X * X.inv()).mat, jnp.eye(6), atol=1e-6)