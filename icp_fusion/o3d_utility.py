'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''

import open3d as o3d
import numpy as np

flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


def make_point_cloud(points, normals=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_icp(source_points, target_points, T):
    pcd_source = make_point_cloud(source_points)
    pcd_source.paint_uniform_color([1, 0, 0])

    pcd_target = make_point_cloud(target_points)
    pcd_target.paint_uniform_color([0, 1, 0])

    pcd_source.transform(T)
    o3d.visualization.draw_geometries([
        pcd_source.transform(flip_transform),
        pcd_target.transform(flip_transform)
    ])


def visualize_correspondences(source_points, target_points, T):
    if len(source_points) != len(target_points):
        print(
            'Error! source points and target points has different length {} vs {}'
            .format(len(source_points), len(target_points)))
        return

    pcd_source = make_point_cloud(source_points)
    pcd_source.paint_uniform_color([1, 0, 0])
    pcd_source.transform(T)
    pcd_source.transform(flip_transform)

    pcd_target = make_point_cloud(target_points)
    pcd_target.paint_uniform_color([0, 1, 0])
    pcd_target.transform(flip_transform)

    corres = []
    for k in range(len(source_points)):
        corres.append((k, k))

    lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        pcd_source, pcd_target, corres)

    o3d.visualization.draw_geometries([pcd_source, pcd_target, lineset])
