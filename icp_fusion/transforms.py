'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''

import numpy as np


def unproject(depth_map, intrinsic):
    h, w = depth_map.shape
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    xs = (us - cx) / fx * depth_map
    ys = (vs - cy) / fy * depth_map

    vertex_map = np.zeros((h, w, 3))
    vertex_map[:, :, 0] = xs
    vertex_map[:, :, 1] = ys
    vertex_map[:, :, 2] = depth_map

    return vertex_map


def project(points, intrinsic):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    us = fx * points[:, 0] / points[:, 2] + cx
    vs = fy * points[:, 1] / points[:, 2] + cy
    ds = points[:, 2]

    return us, vs, ds
