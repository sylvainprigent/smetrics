# -*- coding: utf-8 -*-
"""Patch functionnalities.

Classes
-------
Patch

Methods
-------
draw_patches

"""
import random
import numpy as np


class Patch:
    """Generate a configuration of patch

    Parameters
    ----------
    image : np.array
        image where to generate the patches
    """

    def __init__(self, image: np.array):
        self.image = image

    def grid(self, radius: int, threshold: float = None):
        """ Create a list of patches distributed along a grid

        Parameters
        ----------
        radius: int
            Radius of each patch
        threshold: float
            Minimum mean intensity of the patch to be kept
        """
        patches = []
        x = range(0, self.image.shape[0], 2*radius+1)
        y = range(0, self.image.shape[1], 2*radius+1)
        for i in range(len(x) - 1):
            for j in range(len(y) - 1):
                if not threshold:
                    patches.append((x[i]+radius, y[j]+radius))
                else:
                    if np.mean(self.image[x[i]:x[i + 1], y[j]:y[j + 1]]) \
                            > threshold:
                        patches.append((x[i]+radius, y[j]+radius))
        return patches

    def rand(self, radius: int, threshold: float = None, count: int = 100):
        """Generate a list of randomly distributed patches

        Parameters
        ----------
        radius: int
            Radius of each patch
        threshold: float
            Minimum mean intensity of the patch to be kept
        count: int
            Number of patches

        """
        patches = []
        for _ in range(count):
            p_x = random.randint(radius, self.image.shape[0] - radius)
            p_y = random.randint(radius, self.image.shape[1] - radius)

            if not threshold:
                patches.append((p_x, p_y))
            else:
                patch1 = self.image[p_x - radius:p_x + radius,
                                    p_y - radius:p_y + radius]
                if np.mean(patch1) > threshold:
                    patches.append((p_x, p_y))
        return patches


def draw_patches(image, patches, radius, colors=None):
    """Draw a patch on an image

    Parameters
    ----------
    image: ndarray
        Image where the patch are drawn
    patches: list
        List of the patch positions
    radius: int
        Radius of the patches
    colors: list
        List of colors (one color per patch)

    """
    im_out = np.zeros((image.shape[0], image.shape[1], 3))
    maxi = np.amax(image)
    mini = np.amin(image)
    im_out[:, :, 0] = 255*(image - mini)/(maxi - mini)
    im_out[:, :, 1] = 255*(image - mini)/(maxi - mini)
    im_out[:, :, 2] = 255*(image - mini)/(maxi - mini)
    count = -1
    for patch in patches:
        count += 1
        x_1 = patch[0] - radius+1
        y_1 = patch[1] - radius+1
        x_2 = patch[0] + radius-1
        y_2 = patch[1] + radius-1
        if colors:
            im_out[x_1:x_2, y_1, 0] = 255*colors[count][0]
            im_out[x_1:x_2, y_1, 1] = 255*colors[count][1]
            im_out[x_1:x_2, y_1, 2] = 255*colors[count][2]

            im_out[x_1:x_2, y_2, 0] = 255*colors[count][0]
            im_out[x_1:x_2, y_2, 1] = 255*colors[count][1]
            im_out[x_1:x_2, y_2, 2] = 255*colors[count][2]

            im_out[x_1, y_1:y_2, 0] = 255*colors[count][0]
            im_out[x_1, y_1:y_2, 1] = 255*colors[count][1]
            im_out[x_1, y_1:y_2, 2] = 255*colors[count][2]

            im_out[x_2, y_1:y_2, 0] = 255*colors[count][0]
            im_out[x_2, y_1:y_2, 1] = 255*colors[count][1]
            im_out[x_2, y_1:y_2, 2] = 255*colors[count][2]
        else:
            im_out[x_1:x_2, y_1, 2] = 255
            im_out[x_1:x_2, y_2, 2] = 255
            im_out[x_1, y_1:y_2, 2] = 255
            im_out[x_2, y_1:y_2, 2] = 255
    return im_out
