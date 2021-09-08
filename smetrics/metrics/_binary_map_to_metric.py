import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import binary_erosion
import math


class ClusterMapMetric:
    """Calculate a metric from a binary map using the number of neighbors

    """

    def __init__(self, threshold=4):
        self.threshold = threshold
        self.metric_ = None
        self.count_map_ = None

    def run(self, detection_map):
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        self.count_map_ = convolve(detection_map, kernel) * detection_map
        n = np.count_nonzero(detection_map > 0)
        self.metric_ = np.count_nonzero(self.count_map_ > self.threshold) / n
        return self.metric_


class AreaPerimeterMetric:
    """Calculate the ratio area/perimeter

    """

    def __init__(self):
        self.area_ = -1
        self.perimeter_ = -1
        self.metric_ = None

    def run(self, detection_map):
        self.area_ = np.sum(detection_map == 1)
        contour = detection_map - binary_erosion(detection_map)
        self.perimeter_ = np.sum(contour == 1)
        self.metric_ = self.perimeter_ / self.area_
        return self.metric_


class PercentageAreaMetric:
    """Calculate the ratio area/N

    N is image size

    """

    def __init__(self):
        self.area_ = -1
        self.metric_ = None

    def run(self, detection_map):
        self.area_ = np.sum(detection_map == 1)
        self.metric_ = self.area_ / detection_map.size
        return self.metric_


class IsingMetric:
    """Calculate the Ising MRF binary Energy

    """

    def __init__(self, weight=1):
        self.weight = weight
        self.metric_ = None

    def run(self, detection_map):
        # h_term = number of positive pixels
        h_term = np.sum(detection_map == 1)
        sx = detection_map.shape[0]
        sy = detection_map.shape[1]
        l_term = 0
        for x in range(1, sx - 1):
            for y in range(1, sy - 1):
                v = pow(detection_map[x, y] - detection_map[x - 1, y], 2) + \
                    pow(detection_map[x, y] - detection_map[x, y - 1], 2) + \
                    pow(detection_map[x, y] - detection_map[x, y + 1], 2) + \
                    pow(detection_map[x, y] - detection_map[x - 1, y], 2)
                l_term += v
        #self.metric_ =  math.exp(-h_term-self.weight*l_term)
        self.metric_ = math.exp((-h_term - self.weight * l_term) / detection_map.size)
        return self.metric_
