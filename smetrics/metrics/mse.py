# -*- coding: utf-8 -*-
"""Calculate the mean squared error between two images.

Classes
-------
MSE
NRMSE

Methods
-------
mse
nrmse
"""

import numpy as np


class MSE:
    """Calculate the mean squared error between two images

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare

    Attributes
    ----------
    metric_ : float
        Value of the calculated MSE
    """

    def __init__(self, image1: np.array, image2: np.array):
        self.image1 = image1.astype(np.float64)
        self.image2 = image2.astype(np.float64)
        self.metric_ = None
        self.error_map_ = None

    def run(self):
        """Do the calculation"""
        self.error_map_ = np.square(self.image1 - self.image2)
        self.metric_ = self.error_map_.mean(axis=None)


class NRMSE:
    """Calculate the normalized mean squared error between two images

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare
    norm : str
        Name of the normalisation norm (euclidean, min-max, mean)
        - 'euclidean' : normalize by the averaged Euclidean norm of
          ``image1``
        - 'min-max'   : normalize by the intensity range of ``image1``.
        - 'mean'      : normalize by the mean of ``image1``

    Attributes
    ----------
    metric_ : float
        Value of the calculated NRMSE
    """

    def __init__(self, image1: np.array, image2: np.array,
                 norm: str = 'euclidean'):
        self.image1 = image1.astype(np.float64)
        self.image2 = image2.astype(np.float64)
        self.norm = norm
        self.metric_ = None

    def run(self):
        normalization = self.norm.lower()
        if normalization == 'euclidean':
            denom = np.sqrt(
                np.mean((self.image1 * self.image1), dtype=np.float64))
        elif normalization == 'min-max':
            denom = self.image1.max() - self.image1.min()
        elif normalization == 'mean':
            denom = self.image1.mean()
        else:
            raise ValueError("Unsupported norm_type")
        self.metric_ = np.sqrt(np.mean((self.image1 - self.image2) ** 2,
                               dtype=np.float64)) / denom


def mse(image1: np.array, image2: np.array) -> float:
    """Calculate the mean squared error

    Convenient function to call MSE class

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare

    Returns
    -------
    metric_ : float
        Value of the calculated MSE
    """

    obj = MSE(image1, image2)
    obj.run()
    return obj.metric_


def nrmse(image1: np.array, image2: np.array, norm: str = 'euclidean') -> float:
    """Calculate the normalized mean squared error between two images

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare
    norm : str
        Name of the normalisation norm (euclidean, min-max, mean)
        - 'euclidean' : normalize by the averaged Euclidean norm of
          ``image1``
        - 'min-max'   : normalize by the intensity range of ``image1``.
        - 'mean'      : normalize by the mean of ``image1``

    Returns
    -------
    metric_ : float
        Value of the calculated NRMSE
    """

    obj = NRMSE(image1, image2, norm)
    obj.run()
    return obj.metric_