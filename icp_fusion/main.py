'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''

import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import quaternion  # pip install numpy-quaternion

import transforms
import o3d_utility

from preprocess import load_gt_poses
from icp import icp
from fusion import Map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', help='path to the dataset folder containing rgb/ and depth/')
    parser.add_argument('--start_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=1)
    parser.add_argument('--end_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=200)
    parser.add_argument('--downsample_factor', type=int, default=2)
    args = parser.parse_args()

    intrinsic_struct = o3d.io.read_pinhole_camera_intrinsic('intrinsics.json')
    intrinsic = np.array(intrinsic_struct.intrinsic_matrix)
    indices, gt_poses = load_gt_poses(
        os.path.join(args.path, 'livingRoom2.gt.freiburg'))

    rgb_path = os.path.join(args.path, 'rgb')
    depth_path = os.path.join(args.path, 'depth')
    normal_path = os.path.join(args.path, 'normal')

    # TUM convention
    depth_scale = 5000.0

    m = Map()

    down_factor = args.downsample_factor
    intrinsic /= down_factor
    intrinsic[2, 2] = 1

    # Only use pose 0 for 1-th frame for alignment.
    # DO NOT use other gt poses here
    T_cam_to_world = gt_poses[0]

    T_gt = []
    T_est = []
    for i in range(args.start_idx, args.end_idx + 1):
        print('loading frame {}'.format(i))
        depth = o3d.io.read_image('{}/{}.png'.format(depth_path, i))
        depth = np.asarray(depth) / depth_scale
        depth = depth[::down_factor, ::down_factor]
        vertex_map = transforms.unproject(depth, intrinsic)

        color_map = np.asarray(
            o3d.io.read_image('{}/{}.png'.format(rgb_path,
                                                 i))).astype(float) / 255.0
        color_map = color_map[::down_factor, ::down_factor]

        normal_map = np.load('{}/{}.npy'.format(normal_path, i))
        normal_map = normal_map[::down_factor, ::down_factor]

        if i > 1:
            print('Frame-to-model icp')
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            T_world_to_cam = icp(m.points[::down_factor],
                                 m.normals[::down_factor],
                                 vertex_map,
                                 normal_map,
                                 intrinsic,
                                 T_world_to_cam,
                                 debug_association=False)
            T_cam_to_world = np.linalg.inv(T_world_to_cam)
        print('Point-based fusion')
        m.fuse(vertex_map, normal_map, color_map, intrinsic, T_cam_to_world)

        # A shift is required as gt starts from 1
        T_gt.append(gt_poses[i - 1])
        T_est.append(T_cam_to_world)

    global_pcd = o3d_utility.make_point_cloud(m.points,
                                              colors=m.colors,
                                              normals=m.normals)
    o3d.visualization.draw_geometries(
        [global_pcd.transform(o3d_utility.flip_transform)])

    # Visualize the trajectories
    T_gt = np.stack(T_gt)
    T_est = np.stack(T_est)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    pos_gt = T_gt[:, :3, 3]
    pos_est = T_est[:, :3, 3]
    ax.plot3D(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2])
    ax.plot3D(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2])
    plt.show()
