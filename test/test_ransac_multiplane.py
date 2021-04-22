# encoding=utf-8

"""
Segments pointcloud into multiple planes, extracts their information and allows their visualisation
"""

import math
import os

import numpy as np
# from open3d import *
import open3d as o3d
import seaborn as sns


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


## solve a over-determined non-homogenous equation
def fit_plane_solve_overdetermined(XYZ):
    """
    aX + bY + c = Z
    :param XYZ:
    :return:
    """
    (rows, cols) = XYZ.shape

    G = np.ones((rows, 3), dtype=np.float32)
    G[:, 0] = XYZ[:, 0]  # X
    G[:, 1] = XYZ[:, 1]  # Y

    Z = XYZ[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn

    # return (c, normal)
    return np.array([normal[0], normal[1], normal[2], np.abs(c) / nn])


## TODO: other methods for 3D plane fitting...
# solve a least square probelm
## Ax=B: aX + bY + c = Z <=> aX + bY - Z + c = 0
## https://blog.csdn.net/konglingshneg/article/details/82585868
def fit_plane_least_square(XYZ):
    """
    :param XYZ:
    :return:
    """
    sum_x = np.sum(XYZ[:, 0])
    sum_y = np.sum(XYZ[:, 1])
    sum_z = np.sum(XYZ[:, 2])

    sum_xx = np.sum(np.square(XYZ[:, 0]))
    sum_yy = np.sum(np.square(XYZ[:, 1]))
    sum_xy = np.sum(np.multiply(XYZ[:, 0], XYZ[:, 1]))
    sum_xz = np.sum(np.multiply(XYZ[:, 0], XYZ[:, 2]))
    sum_yz = np.sum(np.multiply(XYZ[:, 1], XYZ[:, 2]))

    avg_x = sum_x / float(XYZ.shape[0])
    avg_y = sum_y / float(XYZ.shape[0])
    avg_z = sum_z / float(XYZ.shape[0])

    A = np.zeros((3, 3), dtype=np.float32)
    B = np.zeros((3, 1), dtype=np.float32)
    A[0, 0] = sum_xx
    A[0, 1] = sum_xy
    A[0, 2] = sum_x
    A[1, 0] = sum_xy
    A[1, 1] = sum_yy
    A[1, 2] = sum_y
    A[2, 0] = sum_x
    A[2, 1] = sum_y
    A[2, 2] = XYZ.shape[0]

    B[0, 0] = sum_xz
    B[1, 0] = sum_yz
    B[2, 0] = sum_z

    # solve non-homogeneous linear equations
    (a, b, c), resid, rank, s = np.linalg.lstsq(A, B, rcond=None)

    # make sure the 3d points are in front of the camera
    normal = [a, b, -1.0]
    if np.dot(normal, np.array([avg_x, avg_y, avg_z])) > 0.0:
        normal *= -1.0

    nn = np.linalg.norm(normal, ord=2)
    normal = normal / nn

    c = np.squeeze(c)
    normal = normal.reshape((1, 3)).squeeze()

    # return (c, normal)
    return np.array([normal[0], normal[1], normal[2], np.abs(c) / nn])


## solve a PCA probel by constructing a lesat square problem
## Can not solve a "big" matrix
# plane equation: AX + BY + CZ + D = 0
# https://www.jianshu.com/p/faa9953213dd
def fit_plane_PCA(XYZ):
    """
    :param XYZ:
    :return:
    """
    # calculate the covariance matrix
    XYZ = XYZ.astype(np.float32)
    new_XYZ = XYZ - np.mean(XYZ, axis=0)
    cov_mat = np.cov(new_XYZ.T)

    # calculate eigen values and eigen vector
    e, v = np.linalg.eig(cov_mat)
    normal = v[:, -1]  # Get the last column of eigen matrix
    # print(normal)
    normal = normal / np.linalg.norm(normal, ord=2)
    # print(normal)

    avg_x = np.mean(XYZ[:, 0])
    avg_y = np.mean(XYZ[:, 1])
    avg_z = np.mean(XYZ[:, 2])

    # make sure the 3d points are in front of the camera
    if np.dot(normal, np.array([avg_x, avg_y, avg_z])) > 0.0:
        normal *= -1.0

    # calculate d
    normal = normal.squeeze()
    d = -(normal[0] * avg_x + normal[1] * avg_y + normal[2] * avg_z)

    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)


def fit_plane_SVD(XYZ):
    """
    :param XYZ:
    :return:
    """
    # avg_x = np.mean(XYZ[:, 0])
    # avg_y = np.mean(XYZ[:, 1])
    # avg_z = np.mean(XYZ[:, 2])

    mean_xyz = np.mean(XYZ, axis=0)
    new_XYZ = XYZ - mean_xyz

    U, S, V = np.linalg.svd(new_XYZ, full_matrices=False, compute_uv=True)
    normal = V[-1] / np.linalg.norm(V[-1], ord=2)

    # make sure the 3d points are in front of the camera
    if np.dot(normal, mean_xyz) > 0.0:
        normal *= -1.0

    # calculate d
    normal = normal.squeeze()
    d = -(normal[0] * mean_xyz[0] + normal[1] * mean_xyz[1] + normal[2] * mean_xyz[2])

    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

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
                if np.linalg.norm(np.array([A, B, C]), ord=2) < 1e-5:
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
    # Assume plane equation: AX + BY + CZ + D = 0

    # plane_arr = fit_plane_solve_overdetermined(inliers)
    # plane_arr = fit_plane_least_square(inliers)
    # plane_arr = fit_plane_PCA(inliers)
    plane_arr = fit_plane_SVD(inliers)
    # -----

    ## ----- Get final plane parameters
    selected.ref_A = plane_arr[0]
    selected.ref_B = plane_arr[1]
    selected.ref_C = plane_arr[2]
    selected.ref_D = plane_arr[3]

    # calculate mean distacne of inliers to the plane: using numpy mat multiply
    # plane = np.array([selected.ref_A, selected.ref_B, selected.ref_C, selected.ref_D],
    #                  dtype=np.float32)
    inlier_pts = np.concatenate([inliers, np.ones((inliers.shape[0], 1))], axis=1)
    dists = np.abs(np.dot(inlier_pts, plane_arr)) / np.linalg.norm(plane_arr[:-1], ord=2)
    mean_dist = np.mean(dists)
    print('Mean distance of inlier point to the plane: {:.3f}m'.format(float(mean_dist)))

    ## calculate camera centerpoint distance to the plane
    cam_dist = np.abs(np.dot(plane_arr, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))) \
               / np.linalg.norm(plane_arr[:-1], ord=2)
    print('Camera distance to plane: {:.3f}m'.format(float(cam_dist)))

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
    number_iterations = 100  # 100
    number_points = points.shape[0]
    distance_threshold = 0.01  # 0.08

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
