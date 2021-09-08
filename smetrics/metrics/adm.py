# -*- coding: utf-8 -*-
"""Calculate the absolute difference map between two images.

Classes
-------
AbsoluteDifferenceMap

Methods
-------
adm
"""

import numpy as np


class AbsoluteDifferenceMap:
    """Calculate the absolute difference map between two images

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare

    Attributes
    ----------
    map_ : np.array
        Absolute difference map
    """

    def __init__(self, image1: np.array, image2: np.array):
        self.image1 = image1.astype(np.float64)
        self.image2 = image2.astype(np.float64)
        self.map_ = None

    def run(self):
        """Do the calculation"""
        self.map_ = np.abs(self.image1 - self.image2)


def adm(image1: np.array, image2: np.array) -> np.array:
    """Calculate the absolute difference map

    Convenient function to call AbsoluteDifferenceMap class

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare

    Returns
    -------
    map_ : float
        Value of the calculated MSE
    """

    obj = AbsoluteDifferenceMap(image1, image2)
    obj.run()
    return obj.map_
