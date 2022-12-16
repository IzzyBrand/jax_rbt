import jax.numpy as jnp

from dynamics import id, fd_differential
from inertia import inertia_of_cylinder, inertia_of_box, SpatialInertiaTensor
from integrate import euler_step
from kinematics import fk
from joint import Revolute, Fixed, Free
from misc_math import prng_key_gen
from rbt import RigidBodyTree, Body, make_q, make_v
from transforms import SpatialTransform, SpatialForceVector, x_rotation
import visualize as vis

def make_simple_arm(num_joints: int,
                    joint_angle: float = jnp.pi / 6,
                    link_length: float = 0.1,
                    body_mass: float = 1.0):
    """Make a simple arm with num_joints revolute joints"""
    # Create the transform from each joint to the next: a translation along z
    # and a rotation about x
    t_z = jnp.array([0, 0, link_length])
    T_parent_to_child = SpatialTransform(x_rotation(joint_angle), t_z)

    # Create the revolute joint used by all bodies
    joint = Revolute(T_parent_to_child)

    # The center of mass of each body is between the two joints
    T_body_to_com = SpatialTransform(x_rotation(jnp.pi/2), 0.5 * t_z)

    # Define the inertia assuming each body is a uniform density cylinder, twice
    # as long as it is wide
    radius = 0.25 * link_length
    inertia = SpatialInertiaTensor.from_I_m(inertia_of_cylinder(
        body_mass, radius, link_length), body_mass).transform(T_body_to_com)

    # Create the base body. The base joint is fixed, so it has no parent, and
    # does not require inertial properties.
    bodies = [Body(0,        # id
                   Fixed(),  # joint
                   -1,     # parent_id (no parent)
                   "base", inertia, body_mass, jnp.zeros(3))]  # name

    # Create the rest of the bodies
    for i in range(1, num_joints + 1):
        bodies.append(Body(i,            # id
                           joint,        # joint
                           i - 1,        # parent_id
                           f"body_{i}",  # name
                           inertia,      # inertia
                           body_mass))   # mass

    for body in bodies:
        body.visuals =[{"type": "cylinder",
                        "radius": radius,
                        "length": link_length,
                        "offset": T_body_to_com.homogenous()}]

    # Create the tree
    return RigidBodyTree(bodies)

def make_box(size, mass):
    """Make a box with a free joint"""
    inertia = SpatialInertiaTensor.from_I_m(inertia_of_box(mass, size), mass)
    body = Body(0, Free(), -1, "box", inertia, mass)
    body.visuals = [{"type": "box", "size": size}]
    return RigidBodyTree([body])

def make_pendulum(length, mass):
    """Make a pendulum with a revolute joint"""
    T_world_joint = SpatialTransform(x_rotation(jnp.pi/2), jnp.zeros(3))
    t_joint_com = jnp.array([0, length, 0])
    T_joint_com = SpatialTransform(jnp.eye(3), t_joint_com)
    inertia = SpatialInertiaTensor.from_I_m(jnp.zeros((3, 3)), mass).transform(T_joint_com)
    body = Body(0,
                Revolute(T_world_joint),
                -1,
                "pendulum",
                inertia,
                mass)

    T_joint_cylinder = SpatialTransform(jnp.eye(3), t_joint_com)
    body.visuals = [
        {"type": "cylinder", "radius": 0.1 * length, "length": length, "offset": T_joint_cylinder.homogenous()},
        {"type": "sphere", "radius": 0.25 * length, "offset": T_joint_com.homogenous()}
    ]
    return RigidBodyTree([body])


def run_and_print_dynamics(rbt):
    """Tries out various forward and inverse dynamics functions."""
    # Get a random configuration
    key_gen = prng_key_gen()
    q0 = make_q(rbt, next(key_gen))
    v0 = make_v(rbt, next(key_gen))
    a0 = make_v(rbt, next(key_gen))
    print("q0:", q0)
    print("v0:", v0)
    print("a0:", a0)

    # Do forward kinematics/dynamics to compute the poses, velocities, and
    # accelerations of each body given the joint positions, velocities, and
    # accelerations
    poses, velocities, accelerations = fk(rbt, q0, v0, a0)
    for b, p, v, a in zip(rbt.bodies, poses, velocities, accelerations):
        print(b.name)
        print("  t: ", p.t)
        print("  v: ", v)
        print("  a: ", a)

    # Now do inverse dynamics to compute the joint forces required to achieve
    # the desired accelerations
    tau = id(rbt, q0, v0, a0)
    print("tau:", tau)

    # Now do forward dynamics to compute the joint accelerations given the
    # joint positions, velocities, and forces
    a1 = fd_differential(rbt, q0, v0, tau)
    print("a0:", a0)
    print("a1:", a1)

if __name__ == "__main__":
    jnp.set_printoptions(precision=6, suppress=True)
    # rbt = make_simple_arm(5)
    # rbt = make_box(jnp.array([0.1, 0.2, 0.3]), 1.0)
    rbt = make_pendulum(0.4, 1.0)
    vis.add_rbt(rbt)

    q = make_q(rbt)
    v = make_v(rbt)
    tau = make_v(rbt)

    q = jnp.array([jnp.pi/2])

    gravity_forces = [SpatialForceVector(jnp.array([0,0,0,0,0,-9.81])) for _ in rbt.bodies]

    while True:
        vis.draw_rbt(rbt, q)
        a = fd_differential(rbt, q, v, tau, gravity_forces)
        print("fd_differential:", a, "\n\n")
        q, v = euler_step(rbt, q, v, a, 0.01)
