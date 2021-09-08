import pytest

from smetrics import data
from smetrics.metrics.rsp import RSP


def test_rsp_pollen():
    image1 = data.pollen()
    image2 = data.pollen_poison_noise_blurred()

    # MSE
    rsp = RSP(image1, image2)
    rsp.run()

    assert rsp.metric_ == pytest.approx(0.40601269, rel=1e-3)
