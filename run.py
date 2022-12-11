import jax.numpy as jnp
import matplotlib.pyplot as plt

from joint import Revolute, zero_q
from transforms import SpatialTransform, x_rotation
from rbt import RigidBodyTree, Body
from kinematics import fk


if __name__ == "__main__":

    t_z = jnp.array([0, 0, 0.1]);
    theta = jnp.pi / 6
    # T_body_i_joint is offset by t_z rotated around x by theta
    T_in = SpatialTransform(x_rotation(theta), t_z)
    # Create the revolute joint
    joint = Revolute(T_in)

    # Create the bodies
    bodies = [Body(0, Fixed(), None, "base")]
    for i in range(1, 7):
        bodies.append(Body(i, joint, i - 1, f"body_{i}"))

    # Create the tree
    rbt = RigidBodyTree(bodies)

    # Do forward kinematics
    q0 = zero_q(rbt)
    v0 = zero_v(rbt)
    a0 = zero_a(rbt)
    poses, velocities, accelerations = fk(rbt, q0, jnp.zeros_like(q0), jnp.zeros_like(q0))

    # Plot the positions of each body
    # for pose in poses:
    #     plt.scatter(pose.t[1], pose.t[2], s=100)
    # plt.show()


