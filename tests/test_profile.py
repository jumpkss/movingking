import numpy as np
import pytest

from ebl_gap.profile import extract_profiles
from ebl_gap.types import Roi
from tests.synth import gap_center_x_at_row, half_max_centre, synth_gap_image


def test_zero_angle_full_roi_is_the_identity():
    img = np.arange(40 * 60, dtype=float).reshape(40, 60)
    roi = Roi(0, 0, 59, 39)
    out = extract_profiles(img, roi, 0.0)
    assert out.shape == (40, 60)
    assert np.allclose(out, img, atol=1e-9)


def test_zero_angle_sub_roi_is_a_plain_crop():
    img = np.arange(40 * 60, dtype=float).reshape(40, 60)
    roi = Roi(10, 5, 39, 24)
    out = extract_profiles(img, roi, 0.0)
    assert out.shape == (20, 30)
    assert np.allclose(out, img[5:25, 10:40], atol=1e-9)


@pytest.mark.parametrize("angle_deg", [2.0, 5.0, 10.0])
def test_correct_angle_makes_every_row_share_one_gap_column(angle_deg):
    """정렬이 제대로 되면 기울어진 갭이 모든 행에서 같은 열에 온다.

    갭 중심은 argmin이 아니라 50% 문턱 중점으로 잡는다. 회전 정렬된 프로파일은
    행마다 샘플링 위상이 달라 평탄한 갭 바닥의 argmin이 4픽셀까지 흔들리는데,
    그것은 정렬 오차가 아니라 argmin의 동점 처리 방식일 뿐이다.
    """
    img = synth_gap_image(width=512, height=512, gap_nm=30.0, nm_per_px=1.0,
                          angle_deg=angle_deg)
    roi = Roi(106, 106, 405, 405)
    centres = np.array([half_max_centre(row)
                        for row in extract_profiles(img, roi, angle_deg)])
    assert np.ptp(centres) < 0.1


def test_wrong_angle_leaves_the_gap_drifting_across_columns():
    img = synth_gap_image(width=512, height=512, gap_nm=30.0, nm_per_px=1.0,
                          angle_deg=10.0)
    roi = Roi(106, 106, 405, 405)
    out = extract_profiles(img, roi, 0.0)
    minima = np.argmin(out, axis=1)
    assert minima.max() - minima.min() > 40


def test_along_axis_averaging_reduces_noise():
    rng = np.random.default_rng(0)
    img = np.full((200, 200), 100.0) + rng.normal(0.0, 10.0, (200, 200))
    roi = Roi(20, 20, 179, 179)
    plain = extract_profiles(img, roi, 0.0, along_average=1)
    smoothed = extract_profiles(img, roi, 0.0, along_average=9)
    assert smoothed.std() < plain.std() * 0.6


def test_sampling_outside_the_image_clamps_instead_of_raising():
    img = np.full((50, 50), 7.0)
    roi = Roi(0, 0, 49, 49)
    out = extract_profiles(img, roi, 20.0)
    assert np.isfinite(out).all()
    assert out.min() == pytest.approx(7.0)


def test_angle_sign_matches_the_project_convention():
    """양의 각도는 행이 내려갈수록 갭이 오른쪽으로 가는 경우다."""
    img = synth_gap_image(width=512, height=512, gap_nm=30.0, nm_per_px=1.0,
                          angle_deg=8.0)
    assert gap_center_x_at_row(400, width=512, height=512, angle_deg=8.0) > \
           gap_center_x_at_row(100, width=512, height=512, angle_deg=8.0)
    roi = Roi(106, 106, 405, 405)
    centres = np.array([half_max_centre(row)
                        for row in extract_profiles(img, roi, 8.0)])
    assert np.ptp(centres) < 0.1
