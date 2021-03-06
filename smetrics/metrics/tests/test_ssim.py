# -*- coding: utf-8 -*-
"""Unit testing the SSIM metric."""

import pytest

from smetrics import data
from smetrics.metrics._ssim import SSIM


def test_ssim_pollen():
    """Test SSIM on pollen 3D image"""
    image1 = data.pollen()
    image2 = data.pollen_poison_noise_blurred()

    # MSE
    ssim = SSIM(image1, image2)
    ssim.run()

    assert ssim.metric_ == pytest.approx(0.43367, rel=1e-3)
