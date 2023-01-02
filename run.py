import argparse

import jax.numpy as jnp

from dynamics import id, fd_differential, fd_composite
from inertia import inertia_of_cylinder, inertia_of_box, SpatialInertiaTensor
from integrate import euler_step
from kinematics import fk
from joint import Revolute, Fixed, Free
from misc_math import prng_key_gen, timer, stats
from rbt import RigidBodyTree, Body, make_q, make_v
from transforms import SpatialTransform, SpatialForceVector, x_rotation, y_rotation
from visualize import star_visualizer, add_rbt, draw_rbt


################################################################################
# Helper functions to create various RBTs
################################################################################

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
                   -1,       # parent_id (no parent)
                   "base", inertia, jnp.zeros(3))]  # name

    # Create the rest of the bodies
    for i in range(1, num_joints + 1):
        bodies.append(Body(i,            # id
                           joint,        # joint
                           i - 1,        # parent_id
                           f"body_{i}",  # name
                           inertia))     # inertia

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
    body = Body(0, Free(), -1, "box", inertia)
    body.visuals = [{"type": "box", "size": size}]
    return RigidBodyTree([body])

def make_pendulum(length, mass):
    """The pendulum rotates around the world x-axis and begins pointed directly
    along the world y-axis. There a point mass at the end of the pendulum."""
    T_w_j = SpatialTransform(y_rotation(jnp.pi/2), jnp.zeros(3))
    T_j_m = SpatialTransform(jnp.eye(3), jnp.array([0, length, 0]))

    rod = Body(0, Revolute(T_w_j), -1, "rod")
    end = Body(1, Fixed(T_j_m), 0, "end", SpatialInertiaTensor.from_m(mass))

    rod.visuals = [{
        "type": "cylinder",
        "radius": 0.03 * length,
        "length": length,
        "offset": SpatialTransform(jnp.eye(3), jnp.array([0, 0.5 * length, 0])).homogenous()
        }]
    end.visuals = [{"type": "sphere", "radius": 0.1 * length}]
    return RigidBodyTree([rod, end])

def make_free_arm(*args):
    """Make an arm with a free joint at its base."""
    rbt = make_simple_arm(*args)
    old_joint = rbt.root.joint
    rbt.root.joint = Free(old_joint.T_in)
    return RigidBodyTree(rbt.bodies)


################################################################################
# Various Experiments
################################################################################

def run_and_print_dynamics(rbt):
    """Tries out various forward and inverse dynamics functions."""
    # Get a random configuration
    key_gen = prng_key_gen()
    q0 = make_q(rbt, next(key_gen))
    v0 = make_v(rbt, next(key_gen))
    a0 = make_v(rbt, next(key_gen))
    f_ext = [SpatialForceVector() for _ in rbt.bodies]
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
    tau = id(rbt, q0, v0, a0, f_ext)
    print("tau:", tau)

    # Now do forward dynamics to compute the joint accelerations given the
    # joint positions, velocities, and forces
    a1 = fd_differential(rbt, q0, v0, tau, f_ext)
    a2 = fd_composite(rbt, q0, v0, tau, f_ext)
    print("a:\t", a0)
    print("a_fd_diff:\t", a1)
    print("a_fd_comp:\t", a2)


def timing_test(rbt):
    f_ext = [SpatialForceVector() for _ in rbt.bodies]

    key_gen = prng_key_gen()

    def rand_args():
        q = make_q(rbt, next(key_gen))
        v = make_v(rbt, next(key_gen))
        a = make_v(rbt, next(key_gen))
        return q, v, a


    r = 100
    n = 10

    # Get the amount of time it takes to get the args
    t_setup = sum(sorted([timer(rand_args) for _ in range(100)])[:n]) / n

    fns = {
        "FD_comp": lambda: fd_composite(rbt, *rand_args(), f_ext),
        "FD_diff": lambda: fd_differential(rbt, *rand_args(), f_ext),
        "ID": lambda: id(rbt, *rand_args(), f_ext),
        "FK": lambda: fk(rbt, *rand_args()),
    }

    for name, fn in fns.items():
        print(name)
        times = [timer(fn) - t_setup for _ in range(r)]
        for k, v in stats(times).items():
            print(f"\t{k}:\t{v * 1000 :.3f} ms")


def simulate_gravity(rbt):
    """Simulates the dynamics of a rigid body tree under gravity."""
    vis = star_visualizer()
    add_rbt(vis, rbt)

    q = make_q(rbt)
    v = make_v(rbt)
    tau = make_v(rbt)
    f_ext = [SpatialForceVector(jnp.array([0,0,0,0,0,0])) for _ in rbt.bodies]

    # v = v.at[:3].set(jnp.array([0, 5, 1e-6]))
    q = jnp.ones_like(q)

    while True:
        draw_rbt(vis, rbt, q)
        a = fd_differential(rbt, q, v, tau, f_ext)
        # a = fd_composite(rbt, q, v, tau, f_ext)
        print("fd_differential:", a, "\n\n")
        q, v = euler_step(rbt, q, v, a, 0.01)


################################################################################
# Main
################################################################################

if __name__ == "__main__":
    jnp.set_printoptions(precision=4, suppress=True)

    models = {
        "arm5": make_simple_arm(5),
        "freearm1": make_free_arm(1),
        "box": make_box(jnp.array([0.05, 0.2, 0.3]), 1.0),
        "pendulum": make_pendulum(0.4, 1.0),
    }

    experiments = {
        "print": run_and_print_dynamics,
        "timeit": timing_test,
        "sim": simulate_gravity,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="arm5", choices=models.keys())
    parser.add_argument("--experiment", type=str, default="sim", choices=experiments.keys())
    args = parser.parse_args()

    # Choose the model
    rbt = models[args.model]
    # Run the experiment
    experiments[args.experiment](rbt)