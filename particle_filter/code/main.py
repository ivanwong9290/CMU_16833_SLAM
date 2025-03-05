'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''
from tqdm import tqdm
import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

from multiprocessing import Pool
from itertools import product

MAP_SIZE = 800
RESOLUTION_REDUCTION = 10 # each grid cell has a 10cm resolution in x, y axes

def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, MAP_SIZE, 0, MAP_SIZE])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = (X_bar[:, 0] // RESOLUTION_REDUCTION).astype(np.uint16)
    y_locs = (X_bar[:, 1] // RESOLUTION_REDUCTION).astype(np.uint16)
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o',s=10)
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))

    plt.pause(0.00001)
    scat.remove()

def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init

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
        occupied_id = np.argwhere(np.logical_or(occupancy > 0.1, occupancy <= 0)).squeeze()

    # initialize random theta (theta0_vals) for all particles
    theta0_vals = np.random.uniform(-np.pi, np.pi, (num_particles, 1))

    # initialize weights (w0_vals) for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64) / num_particles

    # horizontal concantenate all vectors
    X_bar_init = np.hstack((X_bar_init, theta0_vals, w0_vals))

    return X_bar_init

class Simulation:
    def __init__(self, args):
        # os.makedirs(args.output, exist_ok=True)
        self._sensor_data = open(args.path_to_log, 'r')

        self._occupancy_map = MapReader(args.path_to_map).get_map()
        self._num_particles = args.num_particles
        self._X_bar = init_particles_freespace(self._num_particles, self._occupancy_map)

        if args.visualize:
            visualize_map(self._occupancy_map)

def run_monte_carlo_localization(args):
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, weight] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """

    """
    Initialize Parameters
    """
    sim = Simulation(args)

    motion_model = MotionModel(num_particles = sim._num_particles)
    sensor_model = SensorModel(occupancy_map = sim._occupancy_map, 
                               map_resolution_reduction_factor = RESOLUTION_REDUCTION
                              )
    resampler = Resampling()

    X_t0 = sim._X_bar[:, 0:3]

    first_time_idx = True

    """
    Monte Carlo Localization
    """
    for time_idx, line in tqdm(enumerate(sim._sensor_data)):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        if (meas_type == "L"):
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.

        """ MOTION MODEL """
        X_t0 = sim._X_bar[:, 0:3]
        X_t1 = motion_model.update(u_t0=u_t0, u_t1=u_t1, x_t0=X_t0)

        if meas_type == "L":
            """ SENSOR MODEL """
            w_t = sensor_model.beam_range_finder_model(z_t1_arr=ranges, x_t1=X_t1)
            X_bar_t1 = np.hstack((X_t1, w_t))

            """ RESAMPLING """
            X_bar_t1 = resampler.low_variance_sampler(X_bar_t1)

        u_t0 = u_t1

        if args.visualize and args.num_particles > 1:
            visualize_timestep(X_bar_t1, time_idx, args.output)

    print("Program time: %s seconds" % (time.time() - start_time))


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata2.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--ray', action='store_true')
    parser.add_argument('--belief', action='store_true')
    
    args = parser.parse_args()
    
    # Monte Carlo Localization Algorithm
    run_monte_carlo_localization(args)