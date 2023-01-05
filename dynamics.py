from functools import partial

import jax
import jax.numpy as jnp
from jax.nn import one_hot

from inertia import SpatialInertiaTensor
from joint import Joint, Free, Revolute, joint_transform
from kinematics import fk
from rbt import RigidBodyTree, Body, make_v, seg_q
from transforms import (
    SpatialMotionVector,
    SpatialForceVector,
    SpatialTransform,
    SO3_hat,
)


def energy(rbt, q, v) -> float:
    """Compute the total kinetic and potential energy of the system"""
    # Compute the position, velocity, and acceleration of each body
    body_poses, body_vels, _ = fk(rbt, q, v, jnp.zeros_like(v))

    # Compute the kinetic energy of each body
    kinetic = 0
    potential = 0
    for body, X_i, V_i in zip(rbt.bodies, body_poses, body_vels):
        kinetic += 0.5 * V_i.vec.T @ body.inertia.mat @ V_i.vec
        potential += 9.81 * body.inertia.mass * X_i.t[2]

    print(f"Kinetic: {kinetic}, Potential: {potential}")
    return kinetic + potential

@partial(jax.jit, static_argnames=['rbt'])
def id(rbt, q, v, a, f_ext) -> jnp.ndarray:
    """Inverse dynamics using the recursive Newton-Euler algorithm.

    Returns the joint torques required to achieve the given accelerations.
    See Featherstone section 5.3"""

    # 1. Compute the velocity, and acceleration of each body
    body_poses, body_vels, body_accs = fk(rbt, q, v, a)

    # 2. Compute the forces on each body (in the body frame) required to achieve
    #    the accelerations
    body_forces = []
    for body, X_i, V_i, A_i in zip(rbt.bodies, body_poses, body_vels, body_accs):
        # Get the spatial inertia tensor of the body
        I = body.inertia.mat
        # Compute the force on the body to produce the acceleration. Featherstone (5.9)
        # TODO: add multiplication functions to SpatialInertiaTensor
        f_a = SpatialForceVector(I @ A_i.vec + V_i.skew() @ I @ V_i.vec)
        # Remove the external force from the body
        f_x = X_i.rotation().inv() * f_ext[body.idx]
        body_forces.append(f_a - f_x)

    # 3. Compute the force transmitted across each joint. Featherstone (5.20)
    taus = []
    for body in reversed(rbt.bodies):
        f_body = body_forces[body.idx]
        # Convert the forces to generalized coordinates. Featherstone (5.11)
        taus.append(body.joint.S.T @ f_body.vec)
        # If the body has a parent, apply the force to the parent
        if body.p_idx != -1:
            # Get the transform from the parent to this body
            X_parent_body = joint_transform(body.joint, seg_q(body, q))
            # Apply the force from the body to the parent
            body_forces[body.p_idx] += X_parent_body * f_body

    return jnp.concatenate(list(reversed(taus)))


@partial(jax.jit, static_argnames=['rbt'])
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

    # Solve H * qdd = tau - C for qdd Featherstone (6.1)
    return jnp.linalg.solve(H, tau - C)


@partial(jax.jit, static_argnames=['rbt'])
def fd_composite(rbt, q, v, tau, f_ext):
    """Forward dynamics using the composite rigid body algorithm.
    See Featherstone section 6.2"""

    # Compute the joint transforms and body poses for later use
    X_joint = [joint_transform(b.joint, seg_q(b, q)) for b in rbt.bodies]
    X_body, _, _ = fk(rbt, q, v, jnp.zeros_like(v))

    # Calculate the joint space bias force by computing the inverse dynamics
    # with zero acceleration. Featherstone (6.2)
    C = id(rbt, q, v, jnp.zeros_like(v), f_ext)

    # Create an empty joint-space intertia matrix.
    nv = v.shape[0]
    H = jnp.zeros((nv, nv))

    # Init the composite inertia with the inertia of each body to be the inertia
    # of the body itself. We will accumulate the composite inertia of each body
    # in the next step.
    I_c = [b.inertia for b in rbt.bodies]

    # Iterate the bodies in reverse (from leaves to root)
    for body in reversed(rbt.bodies):
        i = body.idx
        # Add the composite inertia of this body to its parent's composite
        # inertia. Given the reverse loop ordering, this has the effect of
        # accumulating the composite inertia up the tree to the root.
        if body.p_idx != -1:
            I_c[body.p_idx] = I_c[body.p_idx] + I_c[i].transform(X_joint[i])

        # This is the simple way of doing it, but it's inneficient, so we'll
        # use the slightly optimized method presented in Featherstone Table 6.2
        # j = i
        # while j != -1:
        #     S_i = rbt.bodies[i].joint.S
        #     S_j = rbt.bodies[j].joint.S
        #     X_ij = X_body[i].inv() * X_body[j]
        #     # Featherstone (6.18)
        #     H_ij = S_i.T @ I_c[i].mat @ X_ij.mat @ S_j

        #     i_start = rbt.bodies[i].v_idx
        #     i_end = i_start + rbt.bodies[i].joint.nv
        #     j_start = rbt.bodies[j].v_idx
        #     j_end = j_start + rbt.bodies[j].joint.nv
        #     H = H.at[i_start:i_end, j_start:j_end].set(H_ij)
        #     H = H.at[j_start:j_end, i_start:i_end].set(H_ij.T)

        #     j = rbt.bodies[j].p_idx

        # Project the composite inertia of the subtree rooted at this body
        # into the joint space ot compute the joint-space intertia.
        F = I_c[i].mat @ body.joint.S
        H_ii = body.joint.S.T @ F

        # Plug the block into the joint-space inertia matrix.
        i_start = rbt.bodies[i].v_idx
        i_end = i_start + body.joint.nv
        H = H.at[i_start:i_end, i_start:i_end].set(H_ii)

        j = i
        while rbt.bodies[j].p_idx != -1:
            # Express F in the parent body coordinates (about to become body j)
            # Featherstone (6.19)
            F = X_joint[j].tinv() @ F
            # Now actually increment to the next body
            j = rbt.bodies[j].p_idx
            # Compute the off-diagonal bloxk of the joint-space intertia matrix
            # Featherstone (6.20)
            H_ij = F.T @ rbt.bodies[j].joint.S

            # Plug the block into the joint-space inertia matrix.
            j_start = rbt.bodies[j].v_idx
            j_end = j_start + rbt.bodies[j].joint.nv
            H = H.at[i_start:i_end, j_start:j_end].set(H_ij)
            H = H.at[j_start:j_end, i_start:i_end].set(H_ij.T)

    # Solve H * qdd = tau - C for qdd Featherstone (6.1)
    return jnp.linalg.solve(H, tau - C)
