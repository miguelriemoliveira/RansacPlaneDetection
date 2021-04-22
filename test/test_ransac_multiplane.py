# encoding=utf-8

"""
Segments pointcloud into multiple planes, extracts their information and allows their visualisation
"""

import math
import os
import random

import numpy as np
# from open3d import *
import open3d as o3d
import seaborn as sns
from tqdm import tqdm


class RansacIteration:
    def __init__(self, num_inliers, A, B, C, D):
        """
        :param num_inliers:
        :param A:
        :param B:
        :param C:
        :param D:
        """
        self.num_inliers = num_inliers

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.ref_A = self.A
        self.ref_B = self.B
        self.ref_C = self.C
        self.ref_D = self.D

    def __str__(self):
        """
        :return:
        """
        return 'num_inliers= ' + \
               str(self.num_inliers) + \
               '\norg A=' + str(self.A) + \
               ' B=' + str(self.B) + \
               ' C=' + str(self.C) + \
               ' D=' + str(self.D) + \
               '\nref A=' + str(self.ref_A) + ' B=' + str(self.ref_B) + ' C=' \
               + str(self.ref_C) + ' D=' + str(self.ref_D)


def fitPlaneLTSQ(XYZ):
    """
    aX + bY + c = Z
    :param XYZ:
    :return:
    """
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


## TODO: other methods for 3D plane fitting...

def ransac_point_separation(pts, n_iter, n_points, distance_threshold):
    """
    :param pts:
    :param n_iter:
    :param n_points:
    :param distance_threshold:
    :return:
    """
    selected = RansacIteration(0, 0, 0, 0, 0)

    # ----- Ransac iterations
    for i in range(0, n_iter):
        # Randomly select 3 3D points that connot be coincident
        while True:
            ids_tri = np.random.choice(np.arange(n_points), 3)
            if ids_tri[0] != ids_tri[1] and ids_tri[1] != ids_tri[2] and ids_tri[2] != ids_tri[0]:

                idx1, idx2, idx3 = ids_tri
                pt1, pt2, pt3 = pts[[idx1, idx2, idx3], :]

                # ABC Hessian coefficients and given by the external product between two vectors lying on hte plane
                A, B, C = np.cross(pt2 - pt1, pt3 - pt1)

                ## colinear check
                if np.linalg.norm(np.array([A, B, C]), ord=2) < 1e-8:
                    continue
                else:
                    break

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

        # print('Iter {:d} done, number of inliers: {:d}.'.format(i + 1, num_inliers))

    # ----- End of Ransac iterations

    # Extract the inliers and the outliers
    distances = abs((selected.A * pts[:, 0] + selected.B * pts[:, 1] + selected.C * pts[:, 2] + selected.D)) / \
                (math.sqrt(selected.A * selected.A + selected.B * selected.B + selected.C * selected.C))
    inliers = pts[np.where(distances < distance_threshold)]
    outliers = pts[np.where(distances >= distance_threshold)]

    # ----- Refine the plane model by fitting a plane to all inliers
    c, normal = fitPlaneLTSQ(inliers)
    # -----

    ## ----- Get final plane parameters
    point = np.array([0.0, 0.0, c])
    d = -point.dot(normal)
    selected.ref_A = normal[0]
    selected.ref_B = normal[1]
    selected.ref_C = normal[2]
    selected.ref_D = d

    # calculate mean distacne of inliers to the plane: using numpy mat multiply
    plane = np.array([selected.ref_A, selected.ref_B, selected.ref_C, selected.ref_D],
                     dtype=np.float32)
    inlier_pts = np.concatenate([inliers, np.ones((inliers.shape[0], 1))], axis=1)
    dists = np.abs(np.dot(inlier_pts, plane)) \
            / np.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
    mena_dist = np.mean(dists)
    print('Mean distance of inlier point to the plane: {:.3f}m'.format(mena_dist))

    ## calculate camera centerpoint distance to the plane
    cam_dist = np.abs(np.dot(plane, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))) \
               / np.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
    print('Camera distance to plane: {:.3f}m'.format(cam_dist))

    return {'plane': selected, 'inliers': inliers, 'outliers': outliers}


if __name__ == "__main__":
    # Load point cloud
    ply_f_path = "../point_clouds/ply_apollo_train_0.ply"
    if not os.path.isfile(ply_f_path):
        print('[Err]: invalid point cloud file path.')
    p = o3d.io.read_point_cloud(ply_f_path)

    # Visualization
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
        ransac_result = ransac_point_separation(points, number_iterations, number_points, distance_threshold)

        # Break if no plane was found     
        if ransac_result['inliers'] is []:
            break

        # Save the obtained results
        ransac_results.append(ransac_result)

        # Break if there are no ouliers anymore
        if len(ransac_result['outliers']) < 3:
            break
        # print('Outliers:\n', ransac_result['outliers'])

        # The new points are the outliers of the previous run
        points = ransac_result['outliers']
        number_points = points.shape[0]
        cnt += 1

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
