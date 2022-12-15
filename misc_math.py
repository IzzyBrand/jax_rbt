import jax
import jax.numpy as jnp

###############################################################################
# PRNG helpers
###############################################################################

def prng_key_gen(seed:int = 0):
    """A generator that returns a new PRNG key each time it is called."""
    key = jax.random.PRNGKey(seed)
    while True:
        key, subkey = jax.random.split(key)
        yield subkey

###############################################################################
# General Math
###############################################################################
@jax.jit
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

