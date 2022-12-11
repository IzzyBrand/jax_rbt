import jax.numpy as jnp

from misc_math import skew
from transforms import (
    SpatialMotionVector,
    SpatialForceVector,
    SpatialTransform
)

class SpatialInertiaTensor:
    def __init__(self, I, m):
        """Construct a spatial inertia tensor.

        Args:
            I (jnp.ndarray): 3x3 rotational inertia matrix
            m (float): mass (kg)
        """
        self.I, self.m = I, m

        # See Featherstone (2.62)
        self.mat = jnp.block([[self.I, jnp.zeros((3, 3))],
                              [jnp.zeros((3, 3)), self.m * jnp.eye(3)]])

    def __mul__(self, v):
        # Spatial inertia is a mapping from spatial motion to spatial force.
        # Featherstone section 2.13
        if isinstance(v, SpatialMotionVector):
            return SpatialForceVector(self.mat @ v.vec)
        elif isinstance(v, jnp.ndarray) and v.shape == (6,):
            return self.mat @ v
        else:
            raise NotImplementedError

    def __add__(self, other):
        # TODO: implement addition of spatial inertia tensors
        return self

    def offset(self, c: jnp.ndarray) -> jnp.ndarray:
        """Compute the spatial inertia tensor at a point offset from the the
        center of mass.

        c is a vector from the offset frame to the center of mass.
        """
        # Featherstone (2.63)
        c_cross = skew(c)
        return jnp.block([
            [self.I + self.m * c_cross @ c_cross.T, self.m * c_cross],
            [self.m * c_cross.T, self.m * jnp.eye(3)]])

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