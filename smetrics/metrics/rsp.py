# -*- coding: utf-8 -*-
"""Calculate the resolution scaled pearson coefficient between two images.

Classes
-------
RSP

Methods
-------
rsp
"""

import math
import numpy as np


class RSP:
    """Calculate the resolution scaled pearson coefficient between two images

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
    """

    def __init__(self, image1: np.array, image2: np.array):
        self.image1 = image1.astype(np.float64)
        self.image2 = image2.astype(np.float64)
        self.metric_ = None

    def run(self):
        """Do the calculation"""
        mean1 = np.mean(self.image1)
        mean2 = np.mean(self.image2)
        num = np.sum(np.multiply((self.image1 - mean1), (self.image2 - mean2)))
        den = math.sqrt(np.sum(np.square(self.image1 - mean1)) *
                        np.sum(np.square(self.image2 - mean2)))
        self.metric_ = num / den


def rsp(image1: np.array, image2: np.array) -> float:
    """Calculate the resolution scaled pearson coefficient between two images

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare

    Returns
    -------
    metric_ : float
        Value of the calculated RSP
    """

    obj = RSP(image1, image2)
    obj.run()
    return obj.metric_
