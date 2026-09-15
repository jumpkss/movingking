import numpy as np
import pytest

from tests.synth import gap_center_x_at_row, half_max_centre, synth_gap_image


def test_gap_center_is_dark_and_far_field_is_bright():
    img = synth_gap_image(width=256, height=64, gap_nm=40.0, nm_per_px=1.0)
    assert img[32, 128] == pytest.approx(40.0, abs=0.5)
    assert img[32, 5] == pytest.approx(200.0, abs=0.5)
    assert img[32, 250] == pytest.approx(200.0, abs=0.5)


def test_fifty_percent_crossing_sits_exactly_at_half_gap():
    """참값 정의: |u| == gap_px/2 인 지점의 밝기가 정확히 중간값이다."""
    img = synth_gap_image(width=256, height=8, gap_nm=40.0, nm_per_px=1.0,
                          edge_sigma_px=2.0)
    cx = (256 - 1) / 2.0
    half = 20.0
    mid = (200.0 + 40.0) / 2.0
    row = img[4]
    # 서브픽셀 위치이므로 선형보간으로 읽는다.
    left_value = np.interp(cx - half, np.arange(256), row)
    right_value = np.interp(cx + half, np.arange(256), row)
    assert left_value == pytest.approx(mid, abs=1.0)
    assert right_value == pytest.approx(mid, abs=1.0)


@pytest.mark.parametrize("angle_deg", [0.0, 2.0, 5.0, 10.0])
def test_rotation_shifts_gap_center_by_tangent(angle_deg):
    img = synth_gap_image(width=512, height=512, gap_nm=30.0, nm_per_px=1.0,
                          angle_deg=angle_deg)
    for row in (180, 256, 330):
        observed = half_max_centre(img[row])
        expected = gap_center_x_at_row(row, width=512, height=512,
                                       angle_deg=angle_deg)
        assert observed == pytest.approx(expected, abs=0.05)


def test_noise_is_reproducible_by_seed():
    a = synth_gap_image(width=64, height=16, noise_sigma=5.0, seed=7)
    b = synth_gap_image(width=64, height=16, noise_sigma=5.0, seed=7)
    c = synth_gap_image(width=64, height=16, noise_sigma=5.0, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_edge_brightening_raises_intensity_just_outside_the_gap():
    plain = synth_gap_image(width=256, height=8, gap_nm=40.0, nm_per_px=1.0,
                            edge_sigma_px=2.0, edge_bright=0.0)
    bright = synth_gap_image(width=256, height=8, gap_nm=40.0, nm_per_px=1.0,
                             edge_sigma_px=2.0, edge_bright=30.0)
    cx = int((256 - 1) / 2.0)
    just_outside = cx + 20 + 3
    assert bright[4, just_outside] > plain[4, just_outside] + 5.0
    # 먼 평탄부는 영향을 받지 않아야 문턱 계산이 흔들리지 않는다.
    assert bright[4, 5] == pytest.approx(plain[4, 5], abs=0.5)
