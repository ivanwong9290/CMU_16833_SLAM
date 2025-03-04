'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from tqdm import tqdm
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader

# try:
#     import cupy as cp
#     use_cupy = True
#     np = cp
# except ImportError:
#     use_cupy = False

class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map: np.ndarray, map_resolution_reduction_factor: int):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """

        # Range measurement error parameters
        self._z_hit = 150 # Correct range with local measurement noise
        self._z_short = 17.5 # Unexpected objects
        self._z_max = 15 # Failure to sense due to max range reached
        self._z_rand = 100 # Random, unexplainable measurements

        # assert(self._z_hit + self._z_short + self._z_max + self._z_rand == 1)

        # Distribution parameters
        self._sigma_hit = 30
        self._lambda_short = 0.5
    
        # Occupancy map
        self._occupancy_map = occupancy_map
        self._mrrf = map_resolution_reduction_factor

        # Laser parameters
        self._laser_max_range = 8183
        self._min_occupancy_probability = 0.35 # p > 0.35 = occupied 
        self._sensor_offset = 25 # in centimeters
        self._angle_stride = 2
        self._laser_stride = 10

    def raycast_model(self, x_t1: np.ndarray) -> np.ndarray:
        """
            Use the current state of all the particles to map out the ray positions, 
            introducing strides make calculations faster as they lower the resolution of the model 

            args:
                x_t1: (M, 3), each row contains (x_m, y_m, theta_m), 
                    0 <= m <= M, M = num_particles
            
            returns:
                z_star : (M, K), each row contains the laser range value of each particle (m) for each angle (k) swept across 0-180 degrees
                    0 <= m <= M, M = num_particles
                    0 <= k <= K, K = 180 / angle_stride 
        """
        # if use_cupy:
        #     x_t1 = cp.asarray(x_t1)

        # prep our angle and range space
        theta_k = np.radians(np.arange(-90, 90, self._angle_stride))
        laser_r = np.arange(0, self._laser_max_range, self._laser_stride)

        # split x, y, theta into individual array 
        x_m, y_m, theta_m = np.hsplit(x_t1, 3)

        # get all combinations of theta_m, theta_k, laser_r
        theta_M, theta_K, laser_R = np.meshgrid(theta_m, theta_k, laser_r, indexing="ij")

        # ray cast map indices for all combinations of laser range increments, laser angle increments, and particles are calculated here
        raycast_x_id = ((laser_R * np.cos(theta_M + theta_K) + self._sensor_offset * np.cos(theta_M) + x_m[:, np.newaxis]) // self._mrrf).astype(int)
        raycast_y_id = ((laser_R * np.sin(theta_M + theta_K) + self._sensor_offset * np.sin(theta_M) + y_m[:, np.newaxis]) // self._mrrf).astype(int)

        # ensure all indices are in bound of map, wrap out-of-bounds value around x = (0, map_w), y = (0, map_h)
        map_h, map_w = self._occupancy_map.shape
        np.clip(raycast_x_id, 0, map_w - 1, out=raycast_x_id)
        np.clip(raycast_y_id, 0, map_h - 1, out=raycast_y_id)

        # check ray occupancies for each particle and each angle, assign true in its indices if the space it occupied is greater than the minimum
        # occupied: (M, K, R)
        occupied = self._occupancy_map[raycast_x_id, raycast_y_id] > self._min_occupancy_probability 

        # we need to find out which particle, at each angle, and at which laser range did it first encountered an obstacle,
        # then transform all indices back into distance (1 index spans 10 cm)
        z_star = np.argmax(occupied, axis=2) * self._mrrf # size (M, K)

        # for each particle and each angle, if no collision is found in that direction, then set the range from 0 to max laser range
        free_indices = ~np.any(occupied, axis=2) # flip the map so free = True instead of occupied
        z_star[free_indices] = self._laser_max_range # before, values at free_indices = 0
        return z_star

    def calculate_p_hit(self, z_star: np.ndarray, z_measured: np.ndarray) -> np.ndarray:
        """
            Get the probability distribution of the measured range accounting for measurement noise.

            args:
                z_star: (M, K), true range measurement from raycasting
                    0 <= m <= M, M = num_particles
                    0 <= k <= K, K = 180 / angle_stride 
                z_measured: (180, 1), measured range measurement
                    
            returns:
                p_hit: (M, K), gaussian distribution of true range measurement for each particle and angle
        """
        # downsample to ensure matching array size between true and measured
        z_measured = z_measured[0::self._angle_stride][:, np.newaxis]
        assert(z_measured.shape[0] == z_star.shape[1])

        # pdf calculation
        dz = z_measured.T - z_star
        p_hit = norm.pdf(dz, loc=z_star, scale=self._sigma_hit)

        # normalizer calculation
        cdf_upper = norm.cdf(self._laser_max_range, loc=z_star, scale=self._sigma_hit)
        cdf_lower = norm.cdf(0, loc=z_star, scale=self._sigma_hit)
        n = 1 / (cdf_upper - cdf_lower)
        
        # apply normalizer
        p_hit *= n

        # mask p_hit if it's not within 0 <= z_measured <= z_max
        out_of_bounds = (z_measured < 0) | (z_measured > self._laser_max_range)
        out_of_bounds = np.repeat(out_of_bounds, p_hit.shape[0], axis=1).T
        p_hit[out_of_bounds] = 0

        return p_hit

    def calculate_p_short(self, z_star: np.ndarray, z_measured: np.ndarray) -> np.ndarray:
        """
            Get the probability distribution of the measured range accounting for sensor noise due to unexpected objects (people walking by, etc.).

            args:
                z_star: (M, K), true range measurement from raycasting
                    0 <= m <= M, M = num_particles
                    0 <= k <= K, K = 180 / angle_stride 
                z_measured: (180, 1), measured range measurement
                    
            returns:
                p_short: (M, K), gaussian distribution of true range measurement for each particle and angle
        """
        # downsample to ensure matching array size between true and measured
        z_measured = z_measured[0::self._angle_stride][:, np.newaxis]
        assert(z_measured.shape[0] == z_star.shape[1])

        # pdf calculation
        p_short = self._lambda_short * np.exp(-self._lambda_short * z_measured)

        # normalizer calculation
        epsilon = 1e-10 # add a small value to avoid total 0 in denominator
        n = np.divide(1, 1 - np.exp(-self._lambda_short * z_star) + epsilon)

        # apply normalizer
        p_short *= n

        # mask p_short if it's not within 0 <= z_measured <= z_star
        out_of_bounds = (z_measured < 0) | (z_measured > z_star)
        out_of_bounds = np.repeat(out_of_bounds, p_short.shape[0], axis=1).T
        p_short[out_of_bounds] = 0

        return p_short

    def calculate_p_max(self, z_measured: np.ndarray) -> np.ndarray:
        """
            Get the probability distribution of the measured range accounting for missing obstacle due to specular reflections.
            args: 
                z_measured: (180, 1), measured range measurement
                    
            returns:
                p_max: (180, 1), uniform distribution of measured range measurement for each particle and angle.
        """
        p_max = (z_measured == self._laser_max_range).astype(int)
        return p_max

    def calculate_p_rand(self, z_measured: np.ndarray) -> np.ndarray:
        """
            Get the probability distribution of the measured range accounting for unexplainable measurements.
            args: 
                z_measured: (180, 1), measured range measurement
                    
            returns:
                p_max: (180, 1), uniform distribution of measured range measurement for each particle and angle.
        """
        p_rand = np.zeros(len(z_measured))
        p_rand[np.logical_and(0 < z_measured, z_measured < self._laser_max_range)] = self._laser_max_range
        
        return p_rand

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
            param[in] z_t1_arr : laser range readings [array of 180 values] at time t
            param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
            param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
            References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
            [Chapter 6.3, Table 6.1]
        """
        """
        TODO : Add your code here
        """
        q = np.ones(len(x_t1))

        z_t1_star = self.raycast_model(x_t1)
        p_hit = self._z_hit * self.calculate_p_hit(z_star=z_t1_star, z_measured=z_t1_arr)
        p_short = self._z_short * self.calculate_p_short(z_star=z_t1_star, z_measured=z_t1_arr)
        p_max = self._z_max * self.calculate_p_max(z_measured=z_t1_arr)[np.newaxis, :]
        p_rand = self._z_rand * self.calculate_p_rand(z_measured=z_t1_arr)[np.newaxis, :]
        p = p_hit + p_short + p_max + p_rand # p: (len(x_t1), 180 / self._angle_stride)

        