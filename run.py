import jax.numpy as jnp

from dynamics import id, fd_differential
from inertia import inertia_of_cylinder, inertia_of_box
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
    com = 0.5 * t_z

    # Define the inertia assuming each body is a uniform density cylinder, twice
    # as long as it is wide
    inertia = inertia_of_cylinder(body_mass, 0.25 * link_length, link_length)

    # Create the base body. The base joint is fixed, so it has no parent, and
    # does not require inertial properties.
    bodies = [Body(0,        # id
                   Fixed(),  # joint
                   None,     # parent_id
                   "base", inertia, body_mass, jnp.zeros(3))]  # name

    # Create the rest of the bodies
    for i in range(1, num_joints + 1):
        bodies.append(Body(i,            # id
                           joint,        # joint
                           i - 1,        # parent_id
                           f"body_{i}",  # name
                           inertia,      # inertia
                           body_mass,    # mass
                           com))         # com

    T_body_to_geom = SpatialTransform(x_rotation(jnp.pi/2), 0.5 * t_z)
    for body in bodies:
        body.visuals =[{"type": "cylinder",
                        "radius": 0.25 * link_length,
                        "length": link_length,
                        "offset": T_body_to_geom.homogenous()}]

    # Create the tree
    return RigidBodyTree(bodies)


def make_box(size, mass):
    """Make a box with a free joint"""
    bodies = [Body(0, Free(), None, "box", inertia_of_box(mass, size), mass)]
    bodies[0].visuals = [{"type": "box", "size": size}]
    return RigidBodyTree(bodies)


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
    # rbt = make_simple_arm(5)
    rbt = make_box(jnp.array([0.1, 0.2, 0.3]), 1.0)
    key_gen = prng_key_gen()
    q = make_q(rbt)
    v = make_v(rbt)
    tau = make_v(rbt)

    a = make_v(rbt)
    a = a.at[5].set(-9.81)

    v = v.at[:3].set(5)
    v = v.at[5].set(3)

    gravity_forces = [SpatialForceVector(jnp.array([0,0,0,0,0,-1])) for _ in rbt.bodies]

    vis.add_rbt(rbt)
    while True:
        vis.draw_rbt(rbt, q)
        # a = fd_differential(rbt, q, v, tau)
        print(f"q:\t{q}\nv:\t{v}\na:\t{a}\ntau:\t{tau}")
        q, v = euler_step(rbt, q, v, a, 0.01)
