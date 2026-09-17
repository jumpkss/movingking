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


def test_empty_image_is_rejected_with_a_korean_error():
    """빈 이미지는 조용히 쓰레기 값을 돌려주는 대신 명확히 거부해야 한다.

    ndim만 검사하면 (0, 0) 배열이 통과하고 map_coordinates가 초기화되지 않은
    메모리를 담은 배열을 돌려준다. 예외보다 나쁜 실패 방식이다.
    """
    with pytest.raises(ValueError, match="이미지"):
        extract_profiles(np.empty((0, 0)), Roi(0, 0, 8, 4), 0.0)


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


def test_aligned_to_image_inverts_the_sampling_transform():
    from ebl_gap.profile import aligned_to_image

    roi = Roi(100, 50, 299, 249)
    for angle_deg in (0.0, 7.0, -12.0):
        prof = extract_profiles(np.arange(300 * 400, dtype=float).reshape(300, 400),
                                roi, angle_deg)
        # 정렬 좌표 (u, v)에서 뽑은 값과 역변환한 이미지 좌표에서 읽은 값이 같아야 한다.
        u_px, v_px = 37, 91
        x, y = aligned_to_image(roi, angle_deg, u_px, v_px)
        img = np.arange(300 * 400, dtype=float).reshape(300, 400)
        from scipy.ndimage import map_coordinates
        direct = map_coordinates(img, [[y], [x]], order=1, mode="nearest")[0]
        assert prof[v_px, u_px] == pytest.approx(direct, rel=1e-9)


def test_aligned_to_image_centre_maps_to_roi_centre():
    from ebl_gap.profile import aligned_to_image

    roi = Roi(100, 50, 299, 249)
    x, y = aligned_to_image(roi, 15.0, (roi.width - 1) / 2.0,
                            (roi.height - 1) / 2.0)
    assert (x, y) == pytest.approx((roi.cx, roi.cy))


# --- 측정 방향이 세로일 때 (Task 28) ---------------------------------------
#
# 사용자의 실제 ROI는 가로 500 x 세로 75였다. 각도 90도에서 ROI의 가로를 측정
# 범위로 쓰면 세로로 ±250픽셀을 훑어 데이터바까지 내려간다 — 화면의 상자와
# 표본을 뜬 자리가 갈라진다.

FLAT_ROI = Roi(10, 200, 509, 274)  # 가로 500 x 세로 75


def test_a_vertical_measurement_direction_swaps_the_roi_box():
    img = np.arange(600 * 600, dtype=float).reshape(600, 600)
    out = extract_profiles(img, FLAT_ROI, 90.0)
    assert out.shape == (FLAT_ROI.width, FLAT_ROI.height) == (500, 75)


def test_a_vertical_measurement_direction_stays_inside_the_roi_box():
    """상자 밖을 훑으면 NaN이 섞여 나온다 — 실제로는 데이터바 밝기가 섞였다.

    금지 구역은 상자에서 한 픽셀 물려 놓는다. 쌍선형 보간이 좌표가 정확히
    정수여도 가중치 0으로 옆 픽셀을 읽고, 0 * NaN은 NaN이라 테두리 한 줄은
    상자 안을 훑어도 NaN이 된다. 이 테스트가 잡으려는 것은 그 한 픽셀이
    아니라 ±250픽셀짜리 탈출이다.
    """
    img = np.full((600, 600), 100.0)
    img[: FLAT_ROI.y0 - 1, :] = np.nan
    img[FLAT_ROI.y1 + 2 :, :] = np.nan
    img[:, : FLAT_ROI.x0 - 1] = np.nan
    img[:, FLAT_ROI.x1 + 2 :] = np.nan
    out = extract_profiles(img, FLAT_ROI, 90.0)
    assert np.isfinite(out).all()


def test_aligned_to_image_follows_the_same_swap():
    """오버레이의 초록 에지는 실제로 잰 자리에 찍혀야 한다.

    `extract_profiles`와 `aligned_to_image`가 갈라지면 CSV의 숫자와 그림이 서로
    다른 자리를 가리킨다 — 계측 툴에서 가장 나쁜 종류의 버그다.
    """
    from scipy.ndimage import map_coordinates

    from ebl_gap.profile import aligned_to_image

    img = np.arange(600 * 600, dtype=float).reshape(600, 600)
    prof = extract_profiles(img, FLAT_ROI, 90.0)
    for u_px, v_px in ((0, 0), (37, 480), (74, 499)):
        x, y = aligned_to_image(FLAT_ROI, 90.0, u_px, v_px)
        direct = map_coordinates(img, [[y], [x]], order=1, mode="nearest")[0]
        assert prof[v_px, u_px] == pytest.approx(direct, rel=1e-9)


def test_uv_extent_keeps_the_roi_box_for_a_horizontal_measurement():
    from ebl_gap.profile import uv_extent

    assert uv_extent(FLAT_ROI, 0.0) == (FLAT_ROI.width, FLAT_ROI.height)
    assert uv_extent(FLAT_ROI, 12.0) == (FLAT_ROI.width, FLAT_ROI.height)
    assert uv_extent(FLAT_ROI, 90.0) == (FLAT_ROI.height, FLAT_ROI.width)
    assert uv_extent(FLAT_ROI, -80.0) == (FLAT_ROI.height, FLAT_ROI.width)
    # 45도는 동률이다. uv_extent와 detect_base_angle_deg가 같은 쪽(가로 측정)을
    # 고르지 않으면 45도 부근에서 상자와 표본이 갈라진다.
    assert uv_extent(FLAT_ROI, 45.0) == (FLAT_ROI.width, FLAT_ROI.height)
