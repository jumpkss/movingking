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
    """최소자승이라면 끌려갈 상황에서 Theil-Sen은 버텨야 한다.

    손상된 행이 analyze_profile 단계에서 걸러지면 회귀에 도달하지 못하고, 그러면
    최소자승으로 바꿔도 이 테스트가 통과한다 — 아무것도 검증하지 못한다는 뜻이다.
    그래서 대비를 유지한 채 갭 위치만 60픽셀 옮긴다. 에지는 정상적으로 검출되고
    회귀에 진짜 이상치로 들어간다. ROI 맨 위 20행을 옮기는 것은 지렛대 효과를
    키우기 위해서다 — 최소자승이라면 기울기가 3도에서 약 -1.3도까지 끌려간다.
    """
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0,
                          angle_deg=3.0)
    img[106:126] = np.roll(img[106:126], 60, axis=1)
    roi = Roi(106, 106, 405, 405)
    estimated, n_rows = estimate_angle_deg(img, roi)
    # 손상 행이 실제로 회귀에 들어갔는지 확인한다. 걸러졌다면 이 테스트는 무의미하다.
    assert n_rows >= 290
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
