#!/usr/bin/env python
"""
This is a long, multiline description
"""

#-------------------------------------------------------------------------------
#--- IMPORTS (standard, then third party, then my own modules)
#-------------------------------------------------------------------------------
import pcl
import numpy as np

#-------------------------------------------------------------------------------
#--- HEADER
#-------------------------------------------------------------------------------
__author__ = "Miguel Riem de Oliveira"
__date__ = "2019"
__copyright__ = "Miguel Riem de Oliveira"
__credits__ = ["Miguel Riem de Oliveira"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Miguel Oliveira"
__email__ = "m.riem.oliveira@gmail.com"
__status__ = "Development"

#-------------------------------------------------------------------------------
#--- FUNCTIONS
#-------------------------------------------------------------------------------

import pcl
p = pcl.load("./table_scene_lms400.pcd")
fil = p.make_statistical_outlier_filter()
fil.set_mean_k (50)
fil.set_std_dev_mul_thresh (1.0)
fil.filter().to_file("inliers.pcd")

p = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32))
p = pcl.load("/home/mike/workingcopy/OptimizationUtils/test/test_cloud.pcd")
# p = pcl.load("./table_scene_lms400.pcd")
seg = p.make_segmenter()
# seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_model_type(pcl.SACMODEL_LINE)
seg.set_method_type(pcl.SAC_RANSAC)
print("asdasd")
indices, model = seg.segment()
