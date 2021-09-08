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
import random

from .patch import Patch


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

    sx = fft1.shape[0]
    sy = fft1.shape[1]
    r_max = min(int(sx / 2), int(sy / 2))
    curve_ = np.zeros(r_max - 1)
    curve_[0] = 1
    frequencies_ = np.zeros(r_max - 1)
    frequencies_[0] = 0
    metric_ = None
    for r in range(1, r_max - 1):
        rr, cc = draw.circle_perimeter(int(sx / 2), int(sy / 2), r)
        p1 = fft1[rr, cc]
        p2 = fft2[rr, cc]

        num = np.abs(np.sum(p1 * np.conjugate(p2)))
        den = np.sum(np.square(np.abs(p1))) * np.sum(np.square(np.abs(p2)))

        curve_[r] = num / math.sqrt(den)
        frequencies_[r] = float(r * resolution)

        if curve_[r] < 1.0 / 7.0 and not metric_:
            metric_ = frequencies_[r]
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
            self.curve_ = np.zeros(self.radius-1)
            self.curves_ = np.zeros((len(self.patches), self.radius-1))
            i = -1
            for patch in self.patches:
                x1 = patch[0] - self.radius
                x2 = patch[0] + self.radius
                y1 = patch[1] - self.radius
                y2 = patch[1] + self.radius
                (curve, self.frequencies_, _) = frc(self.image1[x1:x2, y1:y2],
                                                    self.image2[x1:x2, y1:y2],
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
        fft1 = np.fft.fftshift(np.fft.fftn(self.image1))
        fft2 = np.fft.fftshift(np.fft.fftn(self.image2))
        sx = fft1.shape[0]
        sy = fft1.shape[1]
        sz = fft1.shape[2]
        r_max = min(int(sx / 2), int(sy / 2), int(sz / 2))
        self.metric_ = np.zeros(r_max - 1)
        self.metric_[0] = 1
        self.frequencies_ = np.zeros(r_max - 1)
        self.frequencies_[0] = 0
        for r in range(1, r_max - 1):
            xx, yy, zz = calculate_sphere_border(int(sx / 2), int(sy / 2),
                                                 int(sz / 2), r)
            p1 = fft1[xx, yy, zz]
            p2 = fft2[xx, yy, zz]

            num = np.abs(np.sum(p1 * np.conjugate(p2)))
            den = np.sum(np.square(np.abs(p1))) * np.sum(np.square(np.abs(p2)))

            self.metric_[r] = num / math.sqrt(den)
            self.frequencies_ = float(1 / r)


def calculate_sphere_border(cx: int, cy: int, cz: int, r: int):
    px = []
    py = []
    pz = []
    r1 = pow(r - 1, 2)
    r2 = pow(r, 2)
    for x in range(cx - r, cx + r + 1):
        for y in range(cy - r, cy + r + 1):
            for z in range(cz - r, cz + r + 1):
                euclid = pow(x - cx, 2) + pow(y - cy, 2) + pow(z - cz, 2)
                if r1 < euclid <= r2:
                    px.append(x)
                    py.append(y)
                    pz.append(z)
    return px, py, pz
