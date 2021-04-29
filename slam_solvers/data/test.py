import numpy as np

data = np.load("2d_linear.npz")
gt_traj = data['gt_traj']
gt_landmarks = data['gt_landmarks']
odoms = data['odom']
observations = data['observations']
sigma_odom = data['sigma_odom']
sigma_landmark = data['sigma_landmark']

print(gt_traj.shape)
print(gt_landmarks.shape)
print(odoms.shape)
print(observations.shape)

print(odoms[2].shape)
print(observations[0, 2::])