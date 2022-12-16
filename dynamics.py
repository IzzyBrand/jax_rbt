import jax
import jax.numpy as jnp
from jax.nn import one_hot

from inertia import SpatialInertiaTensor
from joint import Joint, Free, Revolute
from kinematics import fk
from rbt import RigidBodyTree, Body, make_v
from transforms import (
    SpatialMotionVector,
    SpatialForceVector,
    SpatialTransform,
    SO3_hat,
)


def id(rbt, q, v, a, f_ext) -> jnp.ndarray:
    """Inverse dynamics using the recursive Newton-Euler algorithm.

    Returns the joint torques required to achieve the given accelerations.
    See Featherstone section 5.3"""

    # 1. Compute the velocity, and acceleration of each body
    body_poses, body_vels, body_accs = fk(rbt, q, v, a)

    # 2. Compute the forces on each body required to achieve the accelerations
    net_forces = []
    for body, X, s_v, s_a in zip(rbt.bodies, body_poses, body_vels, body_accs):
        # Get the spatial inertia tensor of the body in the world frame
        I = body.inertia.transform(X).mat
        # Compute the force on the body. Featherstone (5.9)
        # TODO: add multiplication functions to SpatialInertiaTensor
        net_forces.append(SpatialForceVector(I @ s_a.vec + s_v.skew() @ I @ s_v.vec))


    # 3. Compute the force transmitted across each joint
    joint_forces = [None for _ in rbt.bodies]

    for body in reversed(rbt.bodies):
        # Sum the forces that this body is transmitting to its children
        child_joint_forces = sum((joint_forces[child.idx] for child in body.children), start=SpatialForceVector())
        # Compute the force transmitted from the parent to this body
        # Featherstone (5.10)
        joint_forces[body.idx] = net_forces[body.idx] - f_ext[body.idx] + child_joint_forces

    print("q[0]:\t", q[0])
    print("v[0]:\t", v[0])
    print("a[0]:\t", a[0])
    print("Net Forces[0]:\t", net_forces[0])
    print("Joint Forces[0]:\t", body_poses[0].inv() * joint_forces[0])
    print("joint.S.T\t", rbt.bodies[0].joint.S.T)

    # TODO: investigate why this is not working

    # Convert the joint forces to generalized coordinates.  Featherstone (5.11)
    taus = []
    for body, X, f_j in zip(rbt.bodies, body_poses, joint_forces):
        taus.append(body.joint.S.T @ (X.inv() * f_j).vec)

    return jnp.concatenate(taus)


def fd_differential(rbt, q, v, tau, f_ext):
    """Forward dynamics using the differential algorithm.
    See Featherstone section 6.1"""
    # Calculate the joint space bias force by computing the inverse dynamics
    # with zero acceleration. Featherstone (6.2)
    C = id(rbt, q, v, make_v(rbt), f_ext)

    # Calculate the joint space inertia matrix, H, by using differential inverse
    # dynamics. Featherstone (6.4)
    def id_differential(alpha):
        return id(rbt, q, v, one_hot(alpha, tau.shape[0]), f_ext) - C

    H = jnp.stack([id_differential(alpha) for alpha in range(tau.shape[0])]).T

    print("C:\t", C)
    print("H:\t", H)
    print("tau:\t", tau)
    # Solve H * qdd = tau - C for qdd Featherstone (6.1)
    return jnp.linalg.solve(H, tau - C)
