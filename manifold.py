import jax.numpy as jnp

from transforms import *
from rbt import *


class Manifold:

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self._add(other)
        elif isinstance(other, jnp.ndarray) and other.shape == (self.nv, ):
            # Raw jnp.ndarrays are interpreted as tangent-space vectors, so we
            # apply the exponential map to get a new manifold point before
            # adding.
            return self._add(self.exp(other))
        else:
            raise ValueError(f"Cannot add {type(self)} and {type(other)}")


class S3(Manifold):
    """A 3D rotation manifold using quaternions."""

    nx = 4
    nv = 3

    def __init__(self, x):
        self.x = x

    def _add(self, other):
        """Composition of two quaternions"""
        return S3(qmul(self.x, other.x))

    @staticmethod
    def exp(v):
        """Exponential map for quaternions"""
        # TODO: implement exponential map for quaternions directly, rather than
        # converting to a rotation matrix first
        return S3(quat_from_mat(SO3_exp(v)))


class Euclidean(Manifold):
    """A Euclidean manifold."""

    def __init__(self, x):
        self.x = x
        self.nx = x.shape[0]
        self.nv = x.shape[0]

    def _add(self, other):
        return Euclidean(self.x + other.x)

    @staticmethod
    def exp(v):
        return Euclidean(v)


class GroupedManifold(Manifold):
    """A manifold that is the product of other manifolds."""

    def __init__(self, *args):
        self.submanifolds = args
        self.x = jnp.concatenate([m.x for m in args])
        self.nx = self.x.shape[0]
        self.nv = sum(m.nv for m in args)

    def _add(self, other):
        xs = []
        for m1, m2 in zip(self.submanifolds, other.submanifolds):
            xs.append(m1 + m2)

        return GroupedManifold(*xs)

    def exp(self, v):
        # Split the vector into the appropriate pieces and apply the exponential
        # map to each submanifold.
        i = 0
        xs = []
        for m in self.submanifolds:
            xs.append(m.exp(v[i:i+m.nv]))
            i += m.nv

        return GroupedManifold(*xs)


def joint_manifold_kin(joint: Joint, q):
    assert q.shape == (joint.nq,)
    if isinstance(joint, Free):
        return GroupedManifold(S3(q[:4]), Euclidean(q[4:]))
    else:
        return Euclidean(q)

def joint_manifold_dyn(joint: Joint, q, v):
    assert q.shape == (joint.nq,)
    assert v.shape == (joint.nv,)
    return GroupedManifold(joint_manifold_kin(joint, q), Euclidean(v))

def rbt_manifold_kin(rbt: RigidBodyTree, q):
    return GroupedManifold(*[joint_manifold_kin(body.joint, seg_q(body, q)) for body in rbt.bodies])

def rbt_manifold_dyn(rbt: RigidBodyTree, q, v):
    return GroupedManifold(rbt_manifold_kin(rbt, q), Euclidean(v))
