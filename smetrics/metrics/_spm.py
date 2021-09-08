# -*- coding: utf-8 -*-
"""Calculate Statistical Parametric Mapping between two images.

Classes
-------
SPM

"""

import math
import numpy as np
from skimage import measure
from skimage import filters


class SPM:
    """Calculate the Statistical Parametric Mapping between two images

    Parameters
    ----------
    image1 : array (2D)
        First image to compare
    image2 : array (2D)
        Second image to compare

    Attributes
    ----------
    inference_map_ : array (2D)
        Value of the calculated p-value map
    level_map_ : array (2D)
        Threshold level map
    detection_map_ : array (2D)
        binary map of significant detected anomalies

    """
    def __init__(self, image1: np.array, image2: np.array):
        if image1 is not None:
            self.image1 = image1.astype(np.float64)
        if image2 is not None:
            self.image2 = image2.astype(np.float64)
        self.thresholds = [1, 1.5, 2, 2.5, 3, 3.5, 4]
        self.alpha = 0.0005
        self.stat_type = 'minHS'
        self.sigma = 0  # smoothing radius
        self.error_map_ = None
        self.inference_map_ = None
        self.level_map_ = None
        self.detection_map_ = None

    def run_error(self, error_map):
        """Calculate two sided SPM on the error map

        Parameters
        ----------
        error_map = ndarray
            Error map between two images

        """
        inference_map, level_map, detection_map = \
            self._inference_map(error_map)
        inference_map_neg, level_map_neg, detection_map_neg = \
            self._inference_map(-error_map)

        # merge
        self.inference_map_ = np.minimum(inference_map, inference_map_neg)
        self.level_map_ = np.maximum(level_map, level_map_neg)
        self.detection_map_ = np.maximum(detection_map, detection_map_neg)

    def run(self):
        """Do the calculation"""
        self.error_map_ = self.image2 - self.image1
        if self.sigma > 0:
            data = filters.gaussian(self.error_map_, sigma=self.sigma)
        else:
            data = self.error_map_
        mean_val = np.mean(data)
        data = data - mean_val
        data = data / math.sqrt(np.var(data))
        data += mean_val
        self.run_error(data)

    def _inference_map(self, input_image):
        """INFERENCE_MAP: Statistical Parametric Mapping

        Function that threshold a Gaussian field with the threshold values
        listed in the vector 'thresholds'. for each threshold, a pvalue map of
        probability for a pixel to be in the gaussian field is estimated with
        the method proposed in J.B.Poline article.

        Parameters
        ----------
        input_image: array (2D)
            input Gaussian field (2D image matrix)

        Return
        ------
        map : array (2D)
            Probability map(inference map)
        detection : array (2D)
            Detection map (threshold of inference map at 0.05)
        """
        # Parameters
        n_l = input_image.shape[0]
        n_c = input_image.shape[1]
        coeff = self._field_roughness(input_image)

        print('field coeff=', coeff)

        # Compute p - value(pick and spatial extent)
        p_values_image = np.zeros((n_l, n_c, len(self.thresholds)))
        idx = -1
        for threshold in self.thresholds:
            idx += 1
            p_values_image[:, :, idx] = self._field_properties_at_t(input_image,
                                                                    threshold,
                                                                    coeff)

        inference_map = np.ones((n_l, n_c))
        for m in range(n_l):
            for n in range(n_c):
                for nth in range(len(self.thresholds)):
                    if p_values_image[m, n, nth] >= 0:
                        inference_map[m, n] = p_values_image[m, n, nth]

        # Threshold the field
        detection_map = np.zeros((n_l, n_c))
        detection_map[inference_map < self.alpha] = 1

        # level map for visualization
        level_map = self._level_map_calc(input_image)

        return inference_map, level_map, detection_map

    @staticmethod
    def _field_roughness(field_image):
        """Calculate the field roughness

        The roughness is the determinant of the covariance matrix of the field
        derivatives

        Parameters
        ----------
        field_image: array (2D)
            Input Gaussian field

        Return
        ------
        coeff: float
            Roughness coefficient multiplied by power(2 * pi, 3 / 2)

        """
        n_l = field_image.shape[0]
        n_c = field_image.shape[1]
        grad_x = np.gradient(field_image, axis=0)
        grad_y = np.gradient(field_image, axis=1)

        f_x = grad_x.reshape((1, n_l * n_c))
        f_y = grad_y.reshape((1, n_l * n_c))

        det_val = np.linalg.det(np.cov(f_x, f_y))
        return pow(2 * math.pi, -3 / 2) * pow(det_val, 0.5)

    def _field_properties_at_t(self, field_image, threshold, coeff):
        """calculate intensity and surface p-value

        Compute the field p-value map for a given threshold t with the method
        described in the F. Lafarge article.

        Parameters
        ----------
        field_image: array (2D)
            Input Gaussian field
        threshold: float
            Threshold value
        coeff: float
            Roughness coefficient of the field

        """
        # 1 - parameters
        n_l = field_image.shape[0]
        n_c = field_image.shape[1]

        # 2 - threshold the field
        t_h = np.zeros((n_l, n_c))
        t_h[field_image > threshold] = 1

        # 3 - Get clusters
        clusters, num_clusters = measure.label(t_h, background=0,
                                               return_num=True)

        print('num clusters:', num_clusters)

        # 4 - compute surface and max of each cluster
        x_0 = np.zeros((num_clusters, 1))
        s_0 = np.zeros((num_clusters, 1))
        for m in range(n_l):
            for n in range(n_c):
                for cluster in range(1, num_clusters+1):
                    if clusters[m, n] == cluster:
                        s_0[cluster-1] = s_0[cluster-1] + 1
                        if field_image[m, n] > x_0[cluster-1]:
                            x_0[cluster-1] = field_image[m, n]

        # 5 - compute p_value of each cluster
        p_value = np.zeros((num_clusters, 1))

        if self.stat_type == 'H':
            for cluster in range(1, num_clusters+1):
                p_value[cluster-1] = (x_0[cluster-1] / threshold) * \
                                     math.exp((threshold * threshold -
                                               x_0[cluster-1] *
                                              x_0[cluster-1]) / 2)
        elif self.stat_type == 'S':
            for cluster in range(1, num_clusters+1):
                p_value[cluster-1] = math.exp(-((coeff * s_0[cluster-1] *
                                                 threshold *
                                              math.exp(-threshold *
                                                       threshold / 2))
                                              / (self._phi(threshold))))
        elif self.stat_type == 'minHS':
            for cluster in range(1, num_clusters+1):
                p_h = (x_0[cluster-1] / threshold) * math.exp((threshold *
                                                              threshold -
                                                              x_0[cluster-1] *
                                                              x_0[cluster-1]) /
                                                              2)
                p_s = math.exp(-((coeff * s_0[cluster-1] * threshold *
                                  math.exp(-threshold * threshold / 2)) /
                                 (self._phi(threshold))))
                p_value[cluster-1] = min(p_h, p_s)
        else:
            raise Exception('enter a valid statistic type for SPM')

        # 6 - replace the p_values in the image
        p_values_image = -np.ones((n_l, n_c))
        for m in range(n_l):
            for n in range(n_c):
                for cluster in range(1, num_clusters+1):
                    if clusters[m, n] == cluster:
                        p_values_image[m, n] = p_value[cluster-1]

        return p_values_image

    @staticmethod
    def _phi(x):
        """Phi function

        Parameters
        ----------
        x: float
            abscissa

        """
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _level_map_calc(self, field_image):
        n_l = field_image.shape[0]
        n_c = field_image.shape[1]
        level_map = np.zeros((n_l, n_c))
        for threshold in self.thresholds:
            level_map[field_image > threshold] = threshold
        return level_map
