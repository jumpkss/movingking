import numpy as np
import pytest

from ebl_gap.orientation import InsufficientEdgesError, estimate_angle_deg
from ebl_gap.types import Roi
from tests.synth import synth_gap_image


@pytest.mark.parametrize("angle_deg", [0.0, 2.0, 5.0, 10.0, -6.0])
def test_recovers_the_true_angle(angle_deg):
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0,
                          angle_deg=angle_deg)
    roi = Roi(106, 106, 405, 405)
    estimated, n_rows = estimate_angle_deg(img, roi)
    assert estimated == pytest.approx(angle_deg, abs=0.2)
    assert n_rows == 300


def test_survives_noise_that_breaks_some_rows():
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0,
                          angle_deg=7.0, noise_sigma=8.0, seed=5)
    roi = Roi(106, 106, 405, 405)
    estimated, _ = estimate_angle_deg(img, roi)
    assert estimated == pytest.approx(7.0, abs=0.5)


def test_theil_sen_is_not_dragged_by_a_few_corrupted_rows():
    """최소자승이라면 끌려갈 상황에서 Theil-Sen은 버텨야 한다."""
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0,
                          angle_deg=3.0)
    img[150:160, :] = 200.0  # 10개 행을 통째로 금속으로 덮는다
    img[300:310, 300:] = 40.0  # 10개 행의 오른쪽 절반을 어둡게 만든다
    roi = Roi(106, 106, 405, 405)
    estimated, _ = estimate_angle_deg(img, roi)
    assert estimated == pytest.approx(3.0, abs=0.5)


def test_raises_when_there_are_too_few_usable_rows():
    img = np.full((256, 256), 180.0)
    roi = Roi(28, 28, 227, 227)
    with pytest.raises(InsufficientEdgesError, match="에지"):
        estimate_angle_deg(img, roi)


def test_reports_how_many_rows_were_usable():
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0)
    img[106:156, :] = 200.0  # ROI 상단 50행을 못 쓰게 만든다
    roi = Roi(106, 106, 405, 405)
    _, n_rows = estimate_angle_deg(img, roi)
    assert 240 <= n_rows <= 260
