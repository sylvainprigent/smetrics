# -*- coding: utf-8 -*-
"""Calculate the fourier ring correlation between two images.

Classes
-------
FRC

Methods
-------
frc
"""

import math
import numpy as np
from skimage import draw


def frc(image1, image2, resolution=1):
    """Calculate the Fourier Ring Correlation of a full image

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare
    resolution: float
        Size of one pixel

    Returns
    -------
    (curve, frequency, metric)
    """

    fft1 = np.fft.fftshift(np.fft.fft2(image1))
    fft2 = np.fft.fftshift(np.fft.fft2(image2))

    s_x = fft1.shape[0]
    s_y = fft1.shape[1]
    r_max = min(int(s_x / 2), int(s_y / 2))
    curve_ = np.zeros(r_max - 1)
    curve_[0] = 1
    frequencies_ = np.zeros(r_max - 1)
    frequencies_[0] = 0
    metric_ = None
    for radius in range(1, r_max - 1):
        r_r, c_c = draw.circle_perimeter(int(s_x / 2), int(s_y / 2), radius)
        p_1 = fft1[r_r, c_c]
        p_2 = fft2[r_r, c_c]

        num = np.abs(np.sum(p_1 * np.conjugate(p_2)))
        den = np.sum(np.square(np.abs(p_1))) * np.sum(np.square(np.abs(p_2)))

        curve_[radius] = num / math.sqrt(den)
        frequencies_[radius] = float(radius * resolution)

        if curve_[radius] < 1.0 / 7.0 and not metric_:
            metric_ = frequencies_[radius]
    return curve_, frequencies_, metric_


class FRC:
    """Calculate the fourier ring correlation between two 2D images

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare
    resolution: float
        Size of one pixel
    patches : list
        list of path center coordinates [(x, y), (x, y), ...]
        A patch list can be generated using the Patch class
    radius : int
        Patch radius

    Attributes
    ----------
    curve_ : np.array
        1D array containing the coefficients for each frequency
    frequencies_ : np.array
        1D array containing the frequencies list
    metric_ : float
        resolution measurement

    """

    def __init__(self, image1: np.array, image2: np.array,
                 resolution: float = 1, patches: list = None, radius: int = -1):
        self.image1 = image1
        self.image2 = image2
        self.resolution = resolution
        self.patches = patches
        self.radius = radius
        self.curves_ = None
        self.curve_ = None
        self.metric_ = None
        self.frequencies_ = None

    def run(self):
        """Do the calculation"""
        if self.image1.shape[0] != self.image2.shape[0] or \
                self.image2.shape[0] != self.image2.shape[0]:
            raise Exception("FRC: image1 and image2 must have the same shape")
        if not self.patches:
            print('run frc global')
            # run global
            (self.curve_, self.frequencies_, self.metric_) = \
                frc(self.image1, self.image2, self.resolution)
        else:
            # run on patches
            self.curve_ = np.zeros(self.radius - 1)
            self.curves_ = np.zeros((len(self.patches), self.radius - 1))
            i = -1
            for patch in self.patches:
                x_1 = patch[0] - self.radius
                x_2 = patch[0] + self.radius
                y_1 = patch[1] - self.radius
                y_2 = patch[1] + self.radius
                (curve, self.frequencies_, _) = frc(self.image1[x_1:x_2,
                                                    y_1:y_2],
                                                    self.image2[x_1:x_2,
                                                    y_1:y_2],
                                                    self.resolution)
                i += 1
                self.curves_[i, :] = curve
                self.curve_ += curve
            count = len(self.patches)
            if count > 0:
                self.curve_ /= count


class FSC:
    """Calculate the fourier shell correlation between two 3D images

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare

    Attributes
    ----------
    metric_ : float
        Value of the calculated RSP

    TODO
    ----
    Current implementation does not take no-isotropic data

    """

    def __init__(self, image1: np.array, image2: np.array):
        self.image1 = image1
        self.image2 = image2
        self.metric_ = None
        self.frequencies_ = None

    def run(self):
        """Do the calculation"""
        fft1 = np.fft.fftshift(np.fft.fftn(self.image1))
        fft2 = np.fft.fftshift(np.fft.fftn(self.image2))
        s_x = fft1.shape[0]
        s_y = fft1.shape[1]
        s_z = fft1.shape[2]
        r_max = min(int(s_x / 2), int(s_y / 2), int(s_z / 2))
        self.metric_ = np.zeros(r_max - 1)
        self.metric_[0] = 1
        self.frequencies_ = np.zeros(r_max - 1)
        self.frequencies_[0] = 0
        for radius in range(1, r_max - 1):
            x_x, y_y, z_z = calculate_sphere_border(int(s_x / 2),
                                                   int(s_y / 2),
                                                   int(s_z / 2),
                                                   radius)
            p_1 = fft1[x_x, y_y, z_z]
            p_2 = fft2[x_x, y_y, z_z]

            num = np.abs(np.sum(p_1 * np.conjugate(p_2)))
            den = np.sum(np.square(np.abs(p_1))) * \
                  np.sum(np.square(np.abs(p_2)))

            self.metric_[radius] = num / math.sqrt(den)
            self.frequencies_ = float(1 / radius)


def calculate_sphere_border(c_x: int, c_y: int, c_z: int, radius: int):
    """Calculate the coordinates of the points in the border of a sphere

    Parameters
    ----------
    c_x: int
        X position of the sphere center
    c_y: int
        Y position of the sphere center
    c_z: int
        Z position of the sphere center
    radius: int
        Radius of the sphere (in pixels)

    """
    p_x = []
    p_y = []
    p_z = []
    r_1 = pow(radius - 1, 2)
    r_2 = pow(radius, 2)
    for x in range(c_x - radius, c_x + radius + 1):
        for y in range(c_y - radius, c_y + radius + 1):
            for z in range(c_z - radius, c_z + radius + 1):
                euclid = pow(x - c_x, 2) + pow(y - c_y, 2) + pow(z - c_z, 2)
                if r_1 < euclid <= r_2:
                    p_x.append(x)
                    p_y.append(y)
                    p_z.append(z)
    return p_x, p_y, p_z
