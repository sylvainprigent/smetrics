"""Unit testing the PSNR metric."""
import pytest

from smetrics import data
from smetrics.metrics.psnr import PSNR


def test_psnr_pollen():
    """Test PSNR on pollen 3D image"""
    image1 = data.pollen()
    image2 = data.pollen_poison_noise_blurred()

    # MSE
    mse = PSNR(image1, image2)
    mse.run()

    assert mse.metric_ == pytest.approx(23.9016, rel=1e-3)
