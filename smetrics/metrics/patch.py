# -*- coding: utf-8 -*-
"""Patch functionnalities.

Classes
-------
Patch

Methods
-------
"""

import numpy as np
import random


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
        patches = []
        for p in range(count):
            px = random.randint(radius, self.image.shape[0] - radius)
            py = random.randint(radius, self.image.shape[1] - radius)

            if not threshold:
                patches.append((px, py))
            else:
                patch1 = self.image[px - radius:px + radius,
                                    py - radius:py + radius]
                if np.mean(patch1) > threshold:
                    patches.append((px, py))
        return patches


def draw_patches(image, patches, radius, colors=None):
    im_out = np.zeros((image.shape[0], image.shape[1], 3))
    maxi = np.amax(image)
    mini = np.amin(image)
    im_out[:, :, 0] = 255*(image - mini)/(maxi - mini)
    im_out[:, :, 1] = 255*(image - mini)/(maxi - mini)
    im_out[:, :, 2] = 255*(image - mini)/(maxi - mini)
    count = -1
    for patch in patches:
        count += 1
        x1 = patch[0] - radius+1
        y1 = patch[1] - radius+1
        x2 = patch[0] + radius-1
        y2 = patch[1] + radius-1
        if colors:
            im_out[x1:x2, y1, 0] = 255*colors[count][0]
            im_out[x1:x2, y1, 1] = 255*colors[count][1]
            im_out[x1:x2, y1, 2] = 255*colors[count][2]

            im_out[x1:x2, y2, 0] = 255*colors[count][0]
            im_out[x1:x2, y2, 1] = 255*colors[count][1]
            im_out[x1:x2, y2, 2] = 255*colors[count][2]

            im_out[x1, y1:y2, 0] = 255*colors[count][0]
            im_out[x1, y1:y2, 1] = 255*colors[count][1]
            im_out[x1, y1:y2, 2] = 255*colors[count][2]

            im_out[x2, y1:y2, 0] = 255*colors[count][0]
            im_out[x2, y1:y2, 1] = 255*colors[count][1]
            im_out[x2, y1:y2, 2] = 255*colors[count][2]
        else:
            im_out[x1:x2, y1, 2] = 255
            im_out[x1:x2, y2, 2] = 255
            im_out[x1, y1:y2, 2] = 255
            im_out[x2, y1:y2, 2] = 255
    return im_out
