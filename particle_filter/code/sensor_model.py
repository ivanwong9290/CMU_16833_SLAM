'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

# import cv2
from tqdm import tqdm
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


# variables :
# LaserReadings = [x, y, theta, xl, yl, thetal, r1......r180]
#
# parameters :
# zHit, zRand, zShort, zMax, sigmaHit, lambdaShort
# L = 25 , n = laser beam numbers


def occupancy(x, resolution, occupancy_map):
    xMap = int(math.floor(x[0] // resolution))
    yMap = int(math.floor(x[1] // resolution))
    return occupancy_map[xMap, yMap]


def inBound(x, resolution, mapSize):
    xMap = math.floor(x[0] // resolution)
    yMap = math.floor(x[1] // resolution)
    if (xMap >= 0) and (xMap < mapSize) and (yMap >= 0) and (yMap < mapSize):
        return True
    else:
        return False


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 150 # 5
        self._z_short = 17.5 # 0.5
        self._z_max = 15 # 0.5
        self._z_rand = 100 # 200
        self._sigma_hit = 100
        self._lambda_short = 15
        self._min_probability = 0.35
        self._subsampling = 2

        """ Occupancy map specs """
        self.OccMap = occupancy_map
        self.OccMapSize = np.size(occupancy_map)
        self.resolution = 10

        """ Laser specs """
        self.laserMax = 8183  # Laser max range
        self.nLaser = 30
        self.laserX = np.zeros((self.nLaser, 1))
        self.laserY = np.zeros((self.nLaser, 1))
        self.beamsRange = np.zeros((self.nLaser, 1))

        # print("OCCUPANCY MAP size : \n", np.shape(self.OccMap))
        # print("OccMapSize Initialized: ", self.OccMapSize)

    def WrapToPi(self, angle):
        angle_wrapped = angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
        return angle_wrapped

    # def WrapAngle(self,angle): # -2pi to 2pi
    #     if angle >= 2*np.pi:
    #         angle -=  (angle // (2*np.pi))*2*np.pi

    #     elif angle <= -2*np.pi:
    #         angle +=  (np.abs(angle) // (2*np.pi))*2*np.pi

    #     return angle

    # def ConvTo90(self,angle): # -pi/2 to pi/2
    #     if angle >= np.pi/2:
    #         angle -= (angle // (np.pi/2))*np.pi/2
    #     elif angle <= - np.pi/2:
    #         angle += (angle // (np.pi/2))*np.pi/2
    #     return angle

    # def getQuad(self,angle):
    #     if (angle>=0 and angle<np.pi/2) or (angle>=-2*np.pi and angle <-3*np.pi/2):
    #         quad = 1
    #     elif (angle>=np.pi/2 and angle <np.pi) or (angle >= -3*np.pi/2 and angle < -np.pi):
    #         quad = 2
    #     elif (angle >= 3*np.pi/2 and angle <2*np.pi) or (angle >= -np.pi/2 and angle <0):
    #         quad = 4
    #     else:
    #         quad = 3
    #     return quad

    # def getProbability_A(self, z_star, z_reading):
    #     # Hit
    #     if 0 < z_reading < self.laserMax:
    #         gauss_norm = self.norm.cdf(self.laserMax, loc=z_star, scale=self._sigma_hit) - self.norm.cdf(0,loc=z_star, scale=self._sigma_hit)
    #         gauss = self.norm.pdf(z_reading,loc=z_star, scale=self._sigma_hit) / gauss_norm
    #     else:
    #         gauss = 0
    #
    #     # short
    #     if 0 < z_reading < z_star:
    #         exp = self._lambda_short*np.exp(-self._lambda_short*z_reading)
    #         exp *= 1/(1-np.exp(-self._lambda_short*z_star))
    #     else:
    #         exp = 0
    #
    #     # Max
    #     if z_reading >= self.laserMax:
    #         p_max = 1
    #     else:
    #         p_max = 0
    #
    #     # random
    #     if (z_reading > 0 and z_reading < self.laserMax):
    #         p_rand = 1/self.laserMax
    #     else:
    #         p_rand = 0
    #     p = self._z_hit*gauss + self._z_short*exp + self._z_max*p_max + self._z_rand*p_rand
    #     p /= (self._z_hit + self._z_short + self._z_max + self._z_rand)

    # return p

    def getProbability(self, z_star, z_reading):
        # hit
        if 0 <= z_reading <= self.laserMax:
            pHit = np.exp(-1 / 2 * (z_reading - z_star) ** 2 / (self._sigma_hit ** 2))
            pHit = pHit / (np.sqrt(2 * np.pi * self._sigma_hit ** 2))

        else:
            pHit = 0

        # short
        if 0 <= z_reading <= z_star:
            # eta = 1.0/(1-np.exp(-lambdaShort*z_star))
            eta = 1
            pShort = eta * self._lambda_short * np.exp(-self._lambda_short * z_reading)

        else:
            pShort = 0

        # max
        if z_reading >= self.laserMax:
            pMax = self.laserMax
        else:
            pMax = 0

        # rand
        if 0 <= z_reading < self.laserMax:
            pRand = 1 / self.laserMax
        else:
            pRand = 0

        p = self._z_hit * pHit + self._z_short * pShort + self._z_max * pMax + self._z_rand * pRand
        p /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
        return p, pHit, pShort, pMax, pRand

    def rayCast(self, x_t1):

        '''
         vectorizing (mask) ---
        '''
        # beamsRange = np.zeros(self.nLaser)
        # laserX = np.zeros(self.nLaser)
        # laserY = np.zeros(self.nLaser)
        # angs   = np.zeros(self.nLaser)
        # L = 25

        # xc = x_t1[0]
        # yc = x_t1[1]
        # myPhi = x_t1[2]
        # ang = myPhi - np.pi/2
        # ang =self.WrapToPi(ang)
        # offSetX = xc + L* np.cos(ang)
        # offSetY = yc + L* np.sin(ang)

        # angStep = np.pi/self.nLaser
        # r = np.linspace(0,self.laserMax,500)

        # for i in range(self.nLaser):

        #     ang += angStep*i
        #     ang = self.WrapToPi(ang)
        #     # casting rays
        #     x = offSetX + r * np.cos(ang)
        #     y = offSetY + r * np.sin(ang)

        #     xInt = np.floor(x/self.resolution).astype(int)
        #     yInt = np.floor(y/self.resolution).astype(int)

        #     # mask = np.zeros_like(xInt).astype(bool)
        #     # mask1= np.zeros_like(xInt).astype(bool)
        #     # mask1[(xInt < 800) & (xInt>=0) & (yInt>=0) & (yInt < 800)] == True
        #     # print("x",xInt[mask1].shape,yInt[mask1].shape)
        #     # print("asdfsaf",self.OccMap[yInt[mask1],xInt[mask1]])

        #     xWithin = np.argwhere(xInt<800)
        #     yWithin = np.argwhere(yInt<800)
        #     within = np.intersect1d(xWithin,yWithin)
        #     hitInd = np.

        #     ii = 0
        #     for xx, yy in zip(xInt[mask1], yInt[mask1]):
        #         if((np.abs(self.OccMap[yInt[xx],xInt[yy]]) > 0.35)):
        #             idx = ii 
        #             break
        #         ii+=1

        #     idx = np.argwhere(mask1==True)[ii]
        #     mask[(np.abs(self.OccMap[yInt[mask1],xInt[mask1]]) > 0.35)] == True
        #     mask[((xInt < 800) & (yInt < 800)) & (np.abs(self.OccMap[yInt,xInt]) > 0.35)] == True
        #     laserX[idx] = x[idx]
        #     laserY[idx] = y[idx]

        #     beamsRange = r[idx]

        ''' 
        normal looping -----
        '''

        beamsRange = np.zeros(self.nLaser)
        laserX = np.zeros(self.nLaser)
        laserY = np.zeros(self.nLaser)
        angs = np.zeros(self.nLaser)
        L = 25

        xc = x_t1[0]
        yc = x_t1[1]
        myPhi = x_t1[2]
        ang = myPhi - np.pi / 2
        ang = self.WrapToPi(ang)
        offSetX = xc + L * np.cos(ang)
        offSetY = yc + L * np.sin(ang)

        angStep = np.pi / self.nLaser

        '''
        set ray step size
        '''
        r = np.linspace(0, self.laserMax, 800)

        for i in range(self.nLaser):

            ang += angStep
            # print(ang*180/np.pi)
            ang = self.WrapToPi(ang)
            # print("angle after wrapped: ",ang*180/np.pi)
            # print("current angle:", ang*180/np.pi)
            # for idx, rs in enumerate(r):
            for rs in r:

                x = offSetX + rs * np.cos(ang)
                y = offSetY + rs * np.sin(ang)

                xInt = np.floor(x / self.resolution).astype(int)
                yInt = np.floor(y / self.resolution).astype(int)

                if xInt < 800 and yInt < 800 and np.abs(self.OccMap[yInt, xInt]) > 0.35:
                    beamsRange[i] = rs
                    phi = np.arctan2((offSetY - yInt), (offSetX - xInt))  # phase
                    angs[i] = ang
                    laserX[i] = xInt
                    laserY[i] = yInt
                    break
                    # print(x,",",y)
        # print(np.abs(angs[0]*180/np.pi-angs[-1]*180/np.pi))

        # print(beamsRange)

        return beamsRange, laserX, laserY

        """ Test Method """

        # L = 25
        #
        # ang = np.linspace(x_t1[2] - np.pi / 2, x_t1[2] + np.pi / 2, 180)[:, np.newaxis]
        # r = np.linspace(0, self.laserMax, 200)[np.newaxis, :]
        #
        # x = x_t1[0] + (r + L) * np.cos(ang)
        # y = x_t1[1] + (r + L) * np.sin(ang)
        #
        # xInt = np.floor(x / self.resolution).astype(int)
        # yInt = np.floor(y / self.resolution).astype(int)
        #
        # for i in range(self.nLaser):
        #     for j in range(r.shape[1]):
        #         if xInt[i][j] < 800 and yInt[i][j] < 800 and np.abs(self.OccMap[yInt[i][j], xInt[i][j]]) > 0.35:
        #             self.laserX[i] = xInt[i][j]
        #             self.laserY[i] = yInt[i][j]
        #             self.beamsRange[i] = r[0][j]
        #             break
        #
        # return self.beamsRange, self.laserX, self.laserY

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        # print("\n ---------\nBEAM RANGE FINDER MODEL CALLED\n ---------\n")

        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        # q = 1

        '''
        q = 0
        # zt_star,self.laserX,self.laserY = rayCast(x_t1,resolution,nLaser,self.OccMap)
        
        # for i in range(nLaser):

        #     pHit   = pHitFun(z_t1_arr[i],zt_star[i],self._sigma_hit)
        #     pShort = pShortFun(z_t1_arr[i],zt_star[i],self._lambda_short)
        #     pMax   = pMaxFun(z_t1_arr[i],self._z_max)
        #     pRand  = pRandFun(z_t1_arr[i],self._z_max)

        #     p = self._z_hit*pHit + self._z_short*pShort + self._z_max*pMax + self._z_rand*pRand
        
        #     q = q*p
            
        #     if q==0:
        #         q = 1e-20
        # return q
        '''
        # q = 1
        q = 0

        step = int(180 / self.nLaser)
        z_reading = [z_t1_arr[n] for n in range(0, 180, step)]
        # print("measure----")
        # print(z_reading)
        zt_star, laserX, laserY = self.rayCast(x_t1)
        # print("my cast----")
        # print(zt_star)
        '''
        # diff = z_reading - zt_star
        # print("difference = ")
        # print(diff)
        '''
        # print(np.size(z_reading))
        probs = np.zeros(self.nLaser)
        for i in range(self.nLaser):
            probs[i], pHit, pShort, pMax, pRand = self.getProbability(zt_star[i], z_reading[i])
            q += np.log(probs[i])

        q = self.nLaser / np.abs(q)
        return q, probs, laserX, laserY

        # prob_zt1 = 1.0
        # return prob_zt1
