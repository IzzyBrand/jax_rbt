import jax.numpy as jnp

###############################################################################
# General Math
###############################################################################
def skew(v):
    """Convert a vector to a skew-symmetric matrix. Applied to another vector u,
    skew(v) @ u = v.cross(u)
    Featherstone (2.23)"""
    return jnp.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

def deskew(m):
    """Convert a skew-symmetric matrix to a vector."""
    return jnp.array([m[2, 1], m[0, 2], m[1, 0]])

def normalize(v):
    """Normalize a vector."""
    return v / jnp.linalg.norm(v)

###############################################################################
# Inertial properties
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

