#!/usr/bin/env python
"""
This is a long, multiline description
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import collections
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!

import pcl
import numpy as np


# RansacIteration = collections.namedtuple('RansacIteration', 'num_inliers A B C D')

class RansacIteration:
    def __init__(self, num_inliers, A, B, C, D):
        self.num_inliers = num_inliers
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.refA = self.A
        self.refB = self.B
        self.refC = self.C
        self.refD = self.D

    def plot(self, inliers, refined=True, color='g'):
        if refined == True:
            A, B, C, D = (self.refA, self.refB, self.refC, self.refD)
        else:
            A, B, C, D = (self.A, self.B, self.C, self.D)

        maxx = np.max(inliers[:, 0])
        maxy = np.max(inliers[:, 1])
        minx = np.min(inliers[:, 0])
        miny = np.min(inliers[:, 1])

        xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
        z = (-A * xx - B * yy - D) * 1. / C
        ax.plot_surface(xx, yy, z, alpha=0.3, color=color)


    def __str__(self):
        return 'num_inliers= ' + str(self.num_inliers) + '\norg A=' + str(self.A) + ' B=' + str(self.B) + ' C=' + str(self.C) + ' D=' + str(self.D) + '\nref A=' + str(self.refA) + ' B=' + str(self.refB) + ' C=' + str(self.refC) + ' D=' + str(self.refD)

# -------------------------------------------------------------------------------
# --- HEADER
# -------------------------------------------------------------------------------
__author__ = "Miguel Riem de Oliveira"
__date__ = "2019"
__copyright__ = "Miguel Riem de Oliveira"
__credits__ = ["Miguel Riem de Oliveira"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Miguel Oliveira"
__email__ = "m.riem.oliveira@gmail.com"
__status__ = "Development"


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------
def fitPlaneLTSQ(XYZ):
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  # X
    G[:, 1] = XYZ[:, 1]  # Y
    Z = XYZ[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return (c, normal)


import pcl

# p = pcl.load("./table_scene_lms400.pcd")
p = pcl.load("./table_scene_lms400_voxelized.pcd")

print(p)
pts = np.asarray(p)
print(pts)

number_iterations = 100
number_points = pts.shape[0]
distance_threshold = 0.02

selected = RansacIteration(0, 0, 0, 0, 0)

# Ransac iterations
for i in range(0, number_iterations):

    # Randomly select three points that connot be cohincident 
    # and colinear
    while True:
        idx1 = random.randint(0, number_points)
        idx2 = random.randint(0, number_points)
        idx3 = random.randint(0, number_points)
        pt1, pt2, pt3 = pts[[idx1, idx2, idx3], :]

        # Compute the norm of position vectors
        ab = np.linalg.norm(pt2 - pt1)
        bc = np.linalg.norm(pt3 - pt2)
        ac = np.linalg.norm(pt3 - pt1)

        # Check if points are colinear
        if (ab + bc) == ac:
            continue
        # Check if are coincident
        if idx2 == idx1:
            continue
        if idx3 == idx1 or idx3 == idx2:
            continue

        # If all the conditions are satisfied, we can end the loop
        break

    # ABC Hessian coefficients and given by the external product between two vectors lying on hte plane
    A, B, C = np.cross(pt2 - pt1, pt3 - pt1)
    # Hessian parameter D is computed using one point that lies on the plane
    D = - (A * pt1[0] + B * pt1[1] + C * pt1[2])

    # Compute the distance from all points to the plane
    # from https://www.geeksforgeeks.org/distance-between-a-point-and-a-plane-in-3-d/
    distances = abs((A * pts[:, 0] + B * pts[:, 1] + C * pts[:, 2] + D)) / (math.sqrt(A * A + B * B + C * C))

    # Compute number of inliers for this plane hypothesis.
    # Inliers are points which have distance to the plane less than a distance_threshold
    num_inliers = (distances < distance_threshold).sum()

    # Store this as the best hypothesis if the number of inliers is larger than the previous max
    if num_inliers > selected.num_inliers:
        selected = RansacIteration(num_inliers, A, B, C, D)

# End of Ransac iterations

# Extract the inliers and the outliers
distances = abs((selected.A * pts[:, 0] + selected.B * pts[:, 1] + selected.C * pts[:, 2] + selected.D)) / \
            (math.sqrt(selected.A * selected.A + selected.B * selected.B + selected.C * selected.C))
inliers = pts[np.where(distances < distance_threshold)]
outliers = pts[np.where(distances >= distance_threshold)]

# Refine the plane model by fitting a plane to all inliers
c, normal = fitPlaneLTSQ(inliers)
point = np.array([0.0, 0.0, c])
d = -point.dot(normal)
selected.refA = normal[0]
selected.refB = normal[1]
selected.refC = normal[2]
selected.refD = d

print('Selected plane: ' + str(selected))



fig = plt.figure()
ax = fig.gca(projection='3d')
# plot original points
ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], color='g')
ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color='r')


selected.plot(inliers,refined=False, color='b')
selected.plot(inliers,refined=True, color='r')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
