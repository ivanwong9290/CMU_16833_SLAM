'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import figure as fig


class MapReader:
    def __init__(self, src_path_map):

        self._occupancy_map = np.genfromtxt(src_path_map, skip_header=7)
        self._occupancy_map[self._occupancy_map < 0] = -1
        self._occupancy_map[self._occupancy_map > 0] = 1 - self._occupancy_map[
            self._occupancy_map > 0]
        self._occupancy_map = np.flipud(self._occupancy_map)

        self._resolution = 10  # each cell has a 10cm resolution in x,y axes
        self._size_x = self._occupancy_map.shape[0] * self._resolution
        self._size_y = self._occupancy_map.shape[1] * self._resolution

        print('Finished reading 2D map of size: ({}, {})'.format(
            self._size_x, self._size_y))

    def visualize_map(self):
        fig = plt.figure()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.ion()
        plt.imshow(self._occupancy_map, cmap='Greys')
        plt.axis([0, 800, 0, 800])
        plt.draw()
        plt.pause(0)

    def get_map(self):
        return self._occupancy_map

    def get_map_size_x(self):  # in cm
        return self._size_x

    def get_map_size_y(self):  # in cm
        return self._size_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    map1 = MapReader(src_path_map)
    map1.visualize_map()
