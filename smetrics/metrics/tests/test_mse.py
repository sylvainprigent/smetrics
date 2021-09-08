import pytest

from smetrics import data
from smetrics.metrics.mse import MSE, NRMSE


def test_mse_pollen():
    image1 = data.pollen()
    image2 = data.pollen_poison_noise_blurred()

    # MSE
    mse = MSE(image1, image2)
    mse.run()

    assert mse.metric_ == pytest.approx( 17489539.14017, rel=1e-3)


def test_nrmse_pollen():
    image1 = data.pollen()
    image2 = data.pollen_poison_noise_blurred()

    # MSE
    nr_mse = NRMSE(image1, image2)
    nr_mse.run()

    assert nr_mse.metric_ == pytest.approx(0.84927136, rel=1e-5)
