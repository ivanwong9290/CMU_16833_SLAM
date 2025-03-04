import numpy as np
import math

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# def occupancy(x, resolution, occupancy_map):
#     xMap = int(math.floor(x[0] // resolution))
#     yMap = int(math.floor(x[1] // resolution))
#     return occupancy_map[xMap, yMap]


# def inBound(x, resolution, mapSize):
#     xMap = math.floor(x[0] // resolution)
#     yMap = math.floor(x[1] // resolution)
#     if (xMap >= 0) and (xMap < mapSize) and (yMap >= 0) and (yMap < mapSize):
#         return True
#     else:
#         return False