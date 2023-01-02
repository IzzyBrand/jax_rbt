import time

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
# Timing
###############################################################################

def timer(fn):
    start = time.time()
    fn()
    stop = time.time()
    return stop - start

def stats(v):
    v = jnp.array(v)
    return {
        "min": jnp.min(v),
        "max": jnp.max(v),
        "mean": jnp.mean(v),
        "25th": jnp.percentile(v, 25),
        "75th": jnp.percentile(v, 75),
    }