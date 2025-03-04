'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math
from utils import wrap_to_pi
from map_reader import MapReader
import matplotlib.pyplot as plt

RESOLUTION_REDUCTION = 10
def init_particles_freespace(num_particles: int, occupancy_map: np.ndarray) -> np.ndarray:

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    # initialize random positions (x0_vals, y0_vals) in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    X_bar_init = np.hstack((x0_vals, y0_vals))

    # we only want [x0_vals, y0_vals] pairs that are not occupied, re-generate until all values are unoccupied
    X_bar_id = (X_bar_init // RESOLUTION_REDUCTION).astype(np.uint16)
    occupancy = occupancy_map[X_bar_id[:, 1], X_bar_id[:, 0]]

    occupied_id = np.argwhere(np.logical_or(occupancy > 0.1, occupancy <= 0)).squeeze()
    while occupied_id.size > 0:
        new_y0_vals = np.random.uniform(0, 7000, (occupied_id.size, 1))
        new_x0_vals = np.random.uniform(3000, 7000, (occupied_id.size, 1))
        X_bar_init[occupied_id] = np.hstack((new_x0_vals, new_y0_vals))

        X_bar_id = (X_bar_init // RESOLUTION_REDUCTION).astype(np.uint16)
        occupancy = occupancy_map[X_bar_id[:, 1], X_bar_id[:, 0]]

        # give a little threshold to consider something is occupied (5%)
        occupied_id = np.argwhere(np.logical_or(occupancy > 0.05, occupancy <= 0)).squeeze()

    # initialize random theta (theta0_vals) for all particles
    theta0_vals = np.random.uniform(-np.pi, np.pi, (num_particles, 1))

    # initialize weights (w0_vals) for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64) / num_particles

    # horizontal concantenate all vectors
    X_bar_init = np.hstack((X_bar_init, theta0_vals, w0_vals))

    return X_bar_init

class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self, num_particles):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.002  # rotation
        self._alpha2 = 0.002 # translation
        self._alpha3 = 0.005  # translation
        self._alpha4 = 0.005  # rotation
        self._num_particles = num_particles

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
            References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
            [Chapter 5.4, Table 5.6]
        """
        """
        TODO : Add your code here
        """

        # If no motion, return previous belief
        if u_t1[0] == u_t0[0] and u_t1[1] == u_t0[1] and u_t1[2] == u_t0[2]: 
            return x_t0
        
        d_rot1 = math.atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        d_trans = math.sqrt((u_t1[0]-u_t0[0])**2 + (u_t1[1]-u_t0[1])**2)
        d_rot2 = u_t1[2] - u_t0[2] - d_rot1

        sample_d_rot1 = self._alpha1 * (d_rot1 ** 2) + self._alpha2 * (d_trans** 2)
        sample_d_trans = self._alpha3 * (d_trans ** 2) + self._alpha4 * (d_rot1 ** 2 + d_rot2 ** 2)
        sample_d_rot2 = self._alpha1 * (d_rot2 ** 2) + self._alpha2 * (d_trans ** 2)

        d_rot1 -= np.random.normal(0, np.sqrt(sample_d_rot1), self._num_particles)
        d_trans -= np.random.normal(0, np.sqrt(sample_d_trans), self._num_particles)
        d_rot2 -= np.random.normal(0, np.sqrt(sample_d_rot2), self._num_particles)

        x_t1 = x_t0[:, 0] + d_trans * np.cos(x_t0[:, 2] + d_rot1)
        y_t1 = x_t0[:, 1] + d_trans * np.sin(x_t0[:, 2] + d_rot1)
        theta_t1 = wrap_to_pi(x_t0[:, 2] + d_rot1 + d_rot2)

        return np.array([x_t1, y_t1, theta_t1]).T

# if __name__ == "__main__":
#     src_path_map = '../data/map/wean.dat'
#     map_obj = MapReader(src_path_map)
#     occupancy_map = map_obj.get_map()
#     src_path_log = '../data/log/robotdata1.log'
#     logfile = open(src_path_log, 'r')

#     motion_model = MotionModel(num_particles=1)
#     X_bar = init_particles_freespace(num_particles=1, occupancy_map=occupancy_map)
#     xt, yt, Xt, Yt = [], [], [], []
#     first_time_idx = True
#     for time_idx, line in enumerate(logfile):
#         meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
#         odometry_robot = meas_vals[0:3]

#         if first_time_idx:
#             u_t0 = odometry_robot
#             first_time_idx = False
#             continue

#         u_t1 = odometry_robot
#         X_bar = motion_model.update(u_t0, u_t1, X_bar)
#         u_t0 = u_t1

#         xt.append(X_bar[:, 0])
#         yt.append(X_bar[:, 1])
#         Xt.append(u_t0[0])
#         Yt.append(u_t0[1])

#     plt.subplot(1, 2, 1)
#     plt.plot(xt, yt)
#     plt.scatter(xt[0], yt[0], c='r')
#     plt.subplot(1, 2, 2)
#     plt.plot(Xt, Yt)
#     plt.scatter(Xt[0], Yt[0], c='r')
#     plt.show()
