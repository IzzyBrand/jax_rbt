import meshcat

import numpy as np
from rbt import RigidBodyTree, seg_q
from joint import joint_transform
from transforms import SpatialTransform


start = meshcat.Visualizer

def add_rbt(rbt: RigidBodyTree, draw_joints=True, draw_bodies=True):
    """Add a rigid body tree to the visualizer"""
    for body in rbt.bodies:
        if draw_joints:
            vis[body.name].set_object(meshcat.geometry.triad(0.1))
        if body.visuals is None or not draw_bodies:
            continue
        for i, geom in enumerate(body.visuals):
            if geom["type"] == "box":
                obj = meshcat.geometry.Box(np.array(geom["size"], dtype=float))
            elif geom["type"] == "sphere":
                obj = meshcat.geometry.Sphere(geom["radius"])
            elif geom["type"] == "cylinder":
                obj = meshcat.geometry.Cylinder(geom["length"], geom["radius"])
            else:
                raise ValueError(f"Unknown geometry type {geom['type']}")

            vis[body.name][str(i)].set_object(obj)

            if "offset" in geom:
                vis[body.name][str(i)].set_transform(np.array(geom["offset"], dtype=float))

def draw_rbt(rbt: RigidBodyTree, q):
    body_poses = []
    for body in rbt.bodies:
        q_joint = seg_q(body, q)
        X_joint = joint_transform(body.joint, q_joint)
        X_parent = body_poses[body.parent_idx] if body.parent else SpatialTransform()
        X_body = X_parent * X_joint
        body_poses.append(X_body)
        vis[body.name].set_transform(np.array(X_body.homogenous(), dtype=float))
