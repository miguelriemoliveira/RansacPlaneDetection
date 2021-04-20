#!/usr/bin/env python
"""
Segments pointcloud into multiple planes, extracts their information and allows their visualisation
"""

import os
import collections
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
# from open3d import *
import open3d as o3d


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

    def __str__(self):
        return 'num_inliers= ' + str(self.num_inliers) + '\norg A=' + str(self.A) + ' B=' + str(self.B) + ' C=' + str(self.C) + ' D=' + str(self.D) + '\nref A=' + str(self.refA) + ' B=' + str(self.refB) + ' C=' + str(self.refC) + ' D=' + str(self.refD)


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


def ransacPointSeparation(pts, niter, npoints, distance_threshold):

        selected = RansacIteration(0, 0, 0, 0, 0)

        # Ransac iterations
        for i in range(0, niter):

            # Randomly select three points that connot be coincident
            # TODO missing check: the points also cannot be colinear
            idx1 = random.randint(0, npoints-1)
            while True:
                idx2 = random.randint(0, npoints-1)
                if not idx2 == idx1:
                    break
            while True:
                idx3 = random.randint(0, npoints-1)
                if not idx3 == idx1 and not idx3 == idx2:
                    break

            pt1, pt2, pt3 = pts[[idx1, idx2, idx3], :]

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

        return {'plane': selected, 'inliers': inliers, 'outliers': outliers}


if __name__ == "__main__":

    # Load point cloud
    pc_path = "/home/negativespade/datasets/meetingRoom5/00000050.ply"
    if not os.path.isfile(pc_path):
        print('[Err]: invalid point cloud file path.')
        exit(-1)
        
    p = o3d.io.read_point_cloud(pc_path)
    # o3d.visualization.draw_geometries([p])

    # Func args
    points = np.asarray(p.points)
    number_iterations = 100
    number_points = points.shape[0]
    distance_threshold = 0.08

    # List of dicts with results for all planes detected
    ransac_results = []

    cnt = 1
    while cnt <= 4:
        # Separate the points which fall within one plane from all other points
        ransac_result = ransacPointSeparation(points, number_iterations, number_points, distance_threshold)
        
        # Break if no plane was found     
        if ransac_result['inliers'] == []:
            break

        # Save the obtained results
        ransac_results.append(ransac_result)

        # Break if there are no ouliers anymore
        if len(ransac_result['outliers']) < 3:
            break

        print(ransac_result['outliers'])

        # The new points are the outliers of the previous run
        points = ransac_result['outliers']
        number_points = points.shape[0]
        cnt = cnt + 1

    current_palette = sns.color_palette()
    separated_clouds = []
    for i, result in enumerate(ransac_results):
        # Found plane information
        print('\nSelected plane: ' + str(result['plane']))

        # Visualization of segments
        inlier_cloud = o3d.geometry.PointCloud()
        
        # Get the segment points
        inlier_cloud.points = o3d.utility.Vector3dVector(result['inliers'])
        
        # Paint them a different colour
        inlier_cloud.paint_uniform_color(current_palette[i])
        
        # Add to the list of clouds to visualise
        separated_clouds.append(inlier_cloud)
    
    p.paint_uniform_color([0, 0, 0])
    
    # Show the original cloud
    separated_clouds.append(p)

    # Draw the clouds
    o3d.visualization.draw_geometries(separated_clouds)
