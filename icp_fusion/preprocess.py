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


def load_gt_poses(gt_filename):
    indices = []
    Ts = []

    # Camera to world
    # Dirty left 2 right coordinate transform
    # https://github.com/theNded/MeshHashing/blob/master/src/io/config_manager.cc#L88
    T_l2r = np.eye(4)
    T_l2r[1, 1] = -1

    with open(gt_filename) as f:
        content = f.readlines()
        for line in content:
            data = np.array(list(map(float, line.strip().split(' '))))
            indices.append(int(data[0]))

            data = data[1:]

            q = data[3:][[3, 0, 1, 2]]
            q = quaternion.from_float_array(q)
            R = quaternion.as_rotation_matrix(q)

            t = data[:3]
            T = np.eye(4)

            T[0:3, 0:3] = R
            T[0:3, 3] = t

            Ts.append(T_l2r @ T @ np.linalg.inv(T_l2r))

    return indices, Ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='path to the dataset folder containing rgb/ and depth/')
    args = parser.parse_args()

    # Load intrinsics and gt poses for evaluation
    intrinsic_struct = o3d.io.read_pinhole_camera_intrinsic('intrinsics.json')
    intrinsic = np.array(intrinsic_struct.intrinsic_matrix)
    indices, gt_poses = load_gt_poses(
        os.path.join(args.path, 'livingRoom2.gt.freiburg'))
    depth_scale = 5000.0

    depth_path = os.path.join(args.path, 'depth')
    normal_path = os.path.join(args.path, 'normal')
    os.makedirs(normal_path, exist_ok=True)

    # Generate normal maps
    # WARNING: please start from index 1, as ground truth poses are provided starting from index 1.
    for i in indices:
        print('Preprocessing frame {:03d}'.format(i))
        depth = np.asarray(o3d.io.read_image('{}/{}.png'.format(
            depth_path, i))) / depth_scale
        vertex_map = transforms.unproject(depth, intrinsic)

        pcd = o3d_utility.make_point_cloud(vertex_map.reshape((-1, 3)))
        pcd.estimate_normals()

        normal_map = np.asarray(pcd.normals).reshape(vertex_map.shape)
        np.save('{}/{}.npy'.format(normal_path, i), normal_map)
