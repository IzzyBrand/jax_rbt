from __future__ import annotations

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from transforms import (
    SpatialMotionVector,
    SpatialForceVector,
    SpatialTransform,
    SO3_hat,
    SO3_from_euler,
)

@register_pytree_node_class
class SpatialInertiaTensor:
    def __init__(self, mat=jnp.zeros((6, 6))):
        """Construct a spatial inertia tensor.
        """
        self.mat = mat
        # The bottom right 3x3 is eye(3) * mass
        self.mass = jnp.mean(jnp.diag(mat[3:, 3:]))

    @staticmethod
    def from_I_m(I: jnp.ndarray, m: float) -> SpatialInertiaTensor:
        """Construct a SpatialInertiaTensor from 3x3 inertia matrix and mass."""
        # See Featherstone (2.62)
        return SpatialInertiaTensor(jnp.block([[I, jnp.zeros((3, 3))],
                                               [jnp.zeros((3, 3)), m * jnp.eye(3)]]))


    @staticmethod
    def from_m(m: float) -> SpatialInertiaTensor:
        """Construct a SpatialInertiaTensor from mass."""
        return SpatialInertiaTensor.from_I_m(jnp.zeros((3, 3)), m)

    def __mul__(self, v):
        # Spatial inertia is a mapping from spatial motion to spatial force.
        # Featherstone section 2.13
        if isinstance(v, SpatialMotionVector):
            return SpatialForceVector(self.mat @ v.vec)
        elif isinstance(v, jnp.ndarray) and v.shape == (6,):
            return self.mat @ v
        else:
            raise NotImplementedError

    def transform(self, X: SpatialTransform) -> SpatialInertiaTensor:
        """Apply a spatial transform to the spatial inertia tensor."""
        # Featherstone (2.66)
        X_inv = X.inv().mat
        return SpatialInertiaTensor(X_inv.T @ self.mat @ X_inv)

    def tree_flatten(self):
        return (self.mat,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __add__(self, other):
        if isinstance(other, SpatialInertiaTensor):
            return SpatialInertiaTensor(self.mat + other.mat)
        else:
            raise NotImplementedError

    # def offset(self, c: jnp.ndarray) -> jnp.ndarray:
    #     """Compute the spatial inertia tensor at a point offset from the the
    #     center of mass.

    #     c is a vector from the offset frame to the center of mass.
    #     """
    #     # Featherstone (2.63)
    #     cx = SO3_hat(c)
    #     return jnp.block([
    #         [self.I + self.m * cx @ cx.T, self.m * cx],
    #         [self.m * cx.T, self.m * jnp.eye(3)]])

###############################################################################
# Inertial matrices for common shapes
###############################################################################

def inertia_of_cylinder(m: float, r: float, h: float) -> jnp.ndarray:
    Ixx = Iyy = (1 / 12) * m * (3 * r ** 2 + h ** 2)
    Izz = (1 / 2) * m * r ** 2
    return jnp.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

def inertia_of_box(m:float, s: jnp.ndarray) -> jnp.ndarray:
    Ixx = (1 / 12) * m * (s[1] ** 2 + s[2] ** 2)
    Iyy = (1 / 12) * m * (s[0] ** 2 + s[2] ** 2)
    Izz = (1 / 12) * m * (s[0] ** 2 + s[1] ** 2)
    return jnp.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

def inertia_of_sphere(m: float, r: float) -> jnp.ndarray:
    Ixx = Iyy = Izz = (2 / 5) * m * r ** 2
    return jnp.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])


if __name__ == "__main__":
    mass = 1.0
    I = inertia_of_box(mass, jnp.array([1, 2, 3]))
    inertia = SpatialInertiaTensor.from_I_m(I, mass)

    # Check translation
    # offset = jnp.array([1,2,3])
    # # TODO: ok but why is this negative?
    # X = SpatialTransform(jnp.eye(3), -offset)
    # assert jnp.allclose(inertia.transform(X).mat,
    #                     inertia.offset(offset))

    # Check rotation
    R = SO3_from_euler(jnp.array([1, 2, 3]))
    X = SpatialTransform(R, jnp.zeros(3))
    assert jnp.allclose(inertia.transform(X).mat,
                        SpatialInertiaTensor.from_I_m(R @ I @ R.T, mass).mat,
                        atol=1e-6)
