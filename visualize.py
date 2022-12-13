import meshcat

from rbt import RigidBodyTree, seg_q
from joint import joint_transform
from transforms import SpatialTransform


vis = meshcat.Visualizer().open()

def add_rbt(rbt: RigidBodyTree):
    """Add a rigid body tree to the visualizer"""
    for body in rbt.bodies:
        if body.visuals is None:
            continue
        for i, geom in enumerate(body.visuals):
            if geom["type"] == "box":
                obj = meshcat.geometry.Box(geom["size"])
            elif geom["type"] == "sphere":
                obj = meshcat.geometry.Sphere(geom["radius"])
            elif geom["type"] == "cylinder":
                obj = meshcat.geometry.Cylinder(geom["length"], geom["radius"])
            else:
                raise ValueError(f"Unknown geometry type {geom['type']}")

            vis[body.name][str(i)].set_object(obj)

            if "offset" in geom:
                vis[body.name][str(i)].set_transform(geom["offset"])

def draw_rbt(rbt: RigidBodyTree, q):
    body_poses = []
    for body in rbt.bodies:
        q_joint = seg_q(body, q)
        X_joint = joint_transform(body.joint, q_joint)
        X_parent = body_poses[body.parent.idx] if body.parent else SpatialTransform()
        X_body = X_parent * X_joint
        body_poses.append(X_body)
        vis[body.name].set_transform(X_body.homogenous_numpy())
