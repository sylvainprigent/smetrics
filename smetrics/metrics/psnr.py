# -*- coding: utf-8 -*-
"""Calculate the Peak signal-to-noise ratio (PSNR) between two images.

Classes
-------
PSNR

Methods
-------
psnr
"""

import numpy as np
from .mse import mse


class PSNR:
    """Calculate the peak signal-to-noise ratio between two images

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare
    data_range : float
        Range of the intensity. if None estimated using the data type or the
        real min and max value for float data type.

    Attributes
    ----------
    metric_ : float
        Value of the calculated PSNR
    """

    def __init__(self, image1: np.array, image2: np.array, data_range=None):
        self.image1 = image1
        self.image2 = image2
        self.data_range = data_range
        self.metric_ = None

    def run(self):
        """Do the calculation"""
        data_range = self.data_range
        if not self.data_range:
            if self.image1.dtype.type == np.uint8:
                data_range = 255
            elif self.image1.dtype.type == np.uint16:
                data_range = 2 ** 16 - 1
            else:
                data_range = np.max(self.image1) - np.min(self.image1)

        self.metric_ = 10 * np.log10((data_range ** 2)
                                     / mse(self.image1, self.image2))


def psnr(image1: np.array, image2: np.array, data_range=None) -> float:
    """Calculate the mean squared error

    Convenient function to call MSE class

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare
    data_range : float
        Range of the intensity. if None estimated using the data type or the
        real min and max value for float data type.

    Returns
    -------
    metric_ : float
        Value of the calculated MSE
    """

    obj = PSNR(image1, image2, data_range)
    obj.run()
    return obj.metric_
