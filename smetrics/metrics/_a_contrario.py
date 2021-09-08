# -*- coding: utf-8 -*-
"""Detect anomalies between two images using A-Contrario method.

Classes
-------
AContrario

Methods
-------
my_disk
my_disk_ring


"""

import math
import numpy as np


def my_disk(center, radius):
    """Calculate the coordinates of pixels inside a circle

    Parameters
    ----------
    center: tuple
        X, Y coordinates of the circle
    radius: float
        Radius of the circle (in pixels unit)

    """
    s_r = radius * radius
    c_c = []
    r_r = []
    for x in range(center[0] - radius, center[0] + radius + 1):
        for y in range(center[1] - radius, center[1] + radius + 1):
            if (x - center[0]) * (x - center[0]) + (y - center[1]) * (
                    y - center[1]) <= s_r:
                r_r.append(x)
                c_c.append(y)
    return np.array(c_c), np.array(r_r)


def my_disk_ring(center, radius, alpha):
    """Calculate the coordinates of pixels in a disk ring

    Parameters
    ----------
    center: tuple
        X, Y coordinates of the circle
    radius: float
        Radius of the circle (in pixels unit)
    alpha: float
        Width of the ring

    """
    s_r = radius * radius
    s_a = (radius + alpha) * (radius + alpha)
    r_o = radius + alpha + 1
    c_c = []
    r_r = []
    for x in range(center[0] - r_o, center[0] + r_o + 1):
        for y in range(center[1] - r_o, center[1] + r_o + 1):
            value = (x - center[0]) * (x - center[0]) + (y - center[1]) * (
                        y - center[1])
            if s_r < value <= s_a:
                r_r.append(x)
                c_c.append(y)
    return np.array(c_c), np.array(r_r)


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
        """Do the calculation"""
        map1 = self.calculate_map(image1, image2)
        map2 = self.calculate_map(image2, image1)
        self.acontrario_map_ = np.maximum(map1, map2)

        self.detection_map_ = np.zeros(image1.shape)
        self.detection_map_[self.acontrario_map_ > self.epsilon] = 1

    def calculate_map(self, image1, image2):
        """Calculate the a-contrario map

        Parameters
        ----------
        image1: ndarray
            Reference image
        image2: ndarray
            Test image
        """
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
                m_1 = np.sum(error_map[irr + x, icc + y]) / i_size
                m_2 = np.sum(error_map[orr + x, occ + y]) / o_size
                stat = coefficient*(m_1-m_2)/sigma
                acontrario_map_[x, y] = n*0.5*math.erfc(stat/sqrt2)
        return acontrario_map_
