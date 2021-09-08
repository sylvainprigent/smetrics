import math
import numpy as np


def my_disk(center, radius):
    sr = radius * radius
    cc = []
    rr = []
    for x in range(center[0] - radius, center[0] + radius + 1):
        for y in range(center[1] - radius, center[1] + radius + 1):
            if (x - center[0]) * (x - center[0]) + (y - center[1]) * (
                    y - center[1]) <= sr:
                rr.append(x)
                cc.append(y)
    return np.array(cc), np.array(rr)


def my_disk_ring(center, radius, alpha):
    sr = radius * radius
    sa = (radius + alpha) * (radius + alpha)
    ro = radius + alpha + 1
    cc = []
    rr = []
    for x in range(center[0] - ro, center[0] + ro + 1):
        for y in range(center[1] - ro, center[1] + ro + 1):
            value = (x - center[0]) * (x - center[0]) + (y - center[1]) * (
                        y - center[1])
            if sr < value <= sa:
                rr.append(x)
                cc.append(y)
    return np.array(cc), np.array(rr)


class AContrario:
    """Implements the a-contrario anomalies detection

    This method is implemented from B. Grosjean, L. Moisan, "A-contrario
    Detectability of Spots in Textured Backgrounds". J. Math. Imaging Vis, 2009

    """

    def __init__(self, radius=5, alpha=7, epsilon=0.5):
        self.radius = radius
        self.alpha = alpha
        self.epsilon = epsilon
        self.detection_map_ = None
        self.acontrario_map_ = None

    def run(self, image1, image2):
        map1 = self.calculate_map(image1, image2)
        map2 = self.calculate_map(image2, image1)
        self.acontrario_map_ = np.maximum(map1, map2)

        self.detection_map_ = np.zeros(image1.shape)
        self.detection_map_[self.acontrario_map_ > self.epsilon] = 1

    def calculate_map(self, image1, image2):
        error_map = image1 - image2

        # measure contrast map in a neighborhood and erf
        acontrario_map_ = np.zeros(error_map.shape)
        coefficient = math.sqrt(math.pi) * \
                      math.sqrt(1 - 1 / (self.alpha * self.alpha)) * self.radius
        sigma = math.sqrt(np.var(error_map))
        n = 1 #error_map.shape[0] * error_map.shape[1]
        irr, icc = my_disk((0, 0), self.radius)
        i_size = irr.shape[0]
        orr, occ = my_disk_ring((0, 0), self.radius, self.alpha)
        o_size = orr.shape[0]

        sqrt2 = math.sqrt(2)
        border = self.radius+self.alpha
        for x in range(border, error_map.shape[0]-border):
            for y in range(border, error_map.shape[1]-border):
                m1 = np.sum(error_map[irr + x, icc + y]) / i_size
                m2 = np.sum(error_map[orr + x, occ + y]) / o_size
                t = coefficient*(m1-m2)/sigma
                acontrario_map_[x, y] = n*0.5*math.erfc(t/sqrt2)
        return acontrario_map_
