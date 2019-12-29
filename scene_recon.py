import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sift_detection as sd
import camera_calib as cc
import conv_2D3D as conv
cam_no = int(input("Choose camera [0 for primary, 1 for secondary]: "))
mtx, dist = cc.camera_calibration(cam_no)  # Camera calibration
F, pts1, pts2 = sd.find_fund()  # Finding fundamental matrix using SIFT features

E_ = np.matmul(np.transpose(mtx), F)
E = np.matmul(E_, mtx)

Points3D = np.zeros((3, len(pts1)))

conv.Convert2DPointsTo3DPoints(pts1, pts2, E, mtx, Points3D)

Points3D_ = np.transpose(Points3D)

file = open("points3D.txt", "r+")

for x, y, z in Points3D_:
    file.write("%f %f %f \n" % (x, y, z))

file.close()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter3D(Points3D_[:, 0], Points3D_[:, 1], Points3D_[:, 2], zdir='z', s=15, c=None, depthshade=True)
ax.set_title('Scatter plot')
plt.show()