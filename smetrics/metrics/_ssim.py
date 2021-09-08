# -*- coding: utf-8 -*-
"""Calculate the structural similarity index between two images.

Classes
-------
SSIM

Methods
-------
ssim
"""

import numpy as np
from skimage.metrics import structural_similarity


class SSIM:
    """Calculate the peak signal-to-noise ratio between two images

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare

    Attributes
    ----------
    metric_ : float
        Value of the calculated mean similarity index
    map_ : np.array
        SSIM map
    """

    def __init__(self, image1: np.array, image2: np.array):
        self.image1 = image1
        self.image2 = image2
        self.metric_ = None
        self.map_ = None

    def run(self):
        """Do the calculation"""
        # settings
        [self.metric_, self.map_] = structural_similarity(self.image1,
                                        self.image2, win_size=None,
                                        gradient=False, data_range=None,
                                        multichannel=False,
                                        gaussian_weights=True, full=True)


def ssim(image1: np.array, image2: np.array) -> float:
    """Calculate the peak signal-to-noise ratio between two images

    Parameters
    ----------
    image1 : np.array
        First image to compare
    image2 : np.array
        Second image to compare

    Returns
    -------
    metric_ : mean SSIM value
    """
    obj = SSIM(image1, image2)
    obj.run()
    return obj.metric_
