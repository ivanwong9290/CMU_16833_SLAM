'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import matplotlib.pyplot as plt


def vectorize_state(traj, landmarks):
    x = np.concatenate((traj.flatten(), landmarks.flatten()))
    return x


def devectorize_state(x, n_poses):
    traj = x[:n_poses * 2].reshape((-1, 2))
    landmarks = x[n_poses * 2:].reshape((-1, 2))
    return traj, landmarks


def plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks):
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='gt poses')
    plt.scatter(gt_landmarks[:, 0],
                gt_landmarks[:, 1],
                c='b',
                marker='+',
                label='gt landmarks')

    plt.plot(traj[:, 0], traj[:, 1], 'r-', label='poses')
    plt.scatter(landmarks[:, 0],
                landmarks[:, 1],
                s=30,
                facecolors='none',
                edgecolors='r',
                label='landmarks')

    plt.legend()
    plt.show()
