# -*- coding: utf-8 -*-
"""Calculate the visual PSNR between two images.

Classes
-------
VPSNR

"""

import math
import numpy as np


class VPSNR:
    """Visual PSNR error map

    Implement the Visual-PSNR from A. Tanchenko, "Visual-PSNR measure of
    image quality", Journal of Visual Communication and Image Representation.
    July 2014

    Parameters
    ----------
    image1 : array (2D)
        First image to compare
    image2 : array (2D)
        Second image to compare

    Attributes
    ----------
    patch_radius: int
        Radius of the local patch (square)
    error_map_ : array (2D)
        Visual-PSNR map

    """
    def __init__(self, image1: np.array, image2: np.array):
        if image1 is not None:
            self.image1 = image1.astype(np.float64)
        if image2 is not None:
            self.image2 = image2.astype(np.float64)
        self.patch_radius = 5
        self.error_map_ = None
        self.metric_ = None

    def run(self):
        """Do the calculation"""
        n_l = self.image1.shape[0]
        n_c = self.image1.shape[1]
        self.error_map_ = np.zeros((n_l, n_c))

        for x in range(self.patch_radius, n_l-self.patch_radius):
            for y in range(self.patch_radius, n_c-self.patch_radius):
                b_x = self.image1[x-self.patch_radius:x+self.patch_radius,
                                  y-self.patch_radius:y+self.patch_radius]
                b_y = self.image2[x - self.patch_radius:x + self.patch_radius,
                                  y - self.patch_radius:y + self.patch_radius]
                b_mse = np.mean((self.image1 - self.image2) ** 2)
                sig_x = np.sqrt(np.var(b_x))
                sig_y = np.sqrt(np.var(b_y))
                self.error_map_[x, y] = b_mse / (1+0.5*math.sqrt(sig_x*sig_y))

        range_ = np.amax(self.image1) - np.amin(self.image1)
        self.metric_ = 10*math.log10(range_*range_ / np.mean(self.error_map_))
