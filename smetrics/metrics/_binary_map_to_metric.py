# -*- coding: utf-8 -*-
"""Algorithms to estimate a image quality metric from an error map

Classes
-------
ClusterMapMetric
AreaPerimeterMetric
PercentageAreaMetric
IsingMetric

"""

import math
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import binary_erosion


class ClusterMapMetric:
    """Calculate a metric from a binary map using the number of neighbors

    Parameters
    ----------
    threshold: float
        Neighbor count to identify clustered pixel

    Attributes
    ----------
    metric_: float
        Obtained metric
    count_map_: ndarray
        cluster map (this is a intermediate variable)

    """

    def __init__(self, threshold=4):
        self.threshold = threshold
        self.metric_ = None
        self.count_map_ = None

    def run(self, detection_map):
        """Do the calculation"""
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        self.count_map_ = convolve(detection_map, kernel) * detection_map
        n = np.count_nonzero(detection_map > 0)
        self.metric_ = np.count_nonzero(self.count_map_ > self.threshold) / n
        return self.metric_


class AreaPerimeterMetric:
    """Calculate the ratio area/perimeter

    Attributes
    ----------
    area_: float
        Measured area
    perimeter_: float
        Measured parameter
    metric_: float
        Obtained metric

    """

    def __init__(self):
        self.area_ = -1
        self.perimeter_ = -1
        self.metric_ = None

    def run(self, detection_map):
        """Do the calculation"""
        self.area_ = np.sum(detection_map == 1)
        contour = detection_map - binary_erosion(detection_map)
        self.perimeter_ = np.sum(contour == 1)
        self.metric_ = self.perimeter_ / self.area_
        return self.metric_


class PercentageAreaMetric:
    """Calculate the ratio area/N

    N is image size

    Attributes
    ----------
    area_: float
        Measured area
    metric_: float
        Obtained metric

    """

    def __init__(self):
        self.area_ = -1
        self.metric_ = None

    def run(self, detection_map):
        """Do the calculation"""
        self.area_ = np.sum(detection_map == 1)
        self.metric_ = self.area_ / detection_map.size
        return self.metric_


class IsingMetric:
    """Calculate the Ising MRF binary Energy

    Parameters
    ----------
    weight: int
        Ising model weight (beta parameter)

    Attributes
    ----------
    metric_: float
        Obtained metric
    """

    def __init__(self, weight=1):
        self.weight = weight
        self.metric_ = None

    def run(self, detection_map):
        """Do the calculation"""
        # h_term = number of positive pixels
        h_term = np.sum(detection_map == 1)
        s_x = detection_map.shape[0]
        s_y = detection_map.shape[1]
        l_term = 0
        for x in range(1, s_x - 1):
            for y in range(1, s_y - 1):
                tmp = pow(detection_map[x, y] - detection_map[x - 1, y], 2) + \
                      pow(detection_map[x, y] - detection_map[x, y - 1], 2) + \
                      pow(detection_map[x, y] - detection_map[x, y + 1], 2) + \
                      pow(detection_map[x, y] - detection_map[x - 1, y], 2)
                l_term += tmp
        self.metric_ = math.exp((-h_term - self.weight * l_term) /
                                detection_map.size)
        return self.metric_
