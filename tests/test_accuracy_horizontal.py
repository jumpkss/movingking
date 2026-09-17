"""가로로 누운 갭에 대한 정확도 게이트.

`tests/test_accuracy.py`와 같은 격자를 돌되 이미지를 `np.rot90`으로 눕힌다.
회전은 참값을 보존하므로 허용 오차도 세로와 같은 ±1픽셀이다. 허용치를 늘려
통과시키면 이 파일의 의미가 없어진다.

이 파일이 존재하는 이유: 이 프로젝트의 합성 이미지가 전부 세로 갭이라, 403개
테스트가 전부 통과하는 동안 가로 갭은 한 번도 시험되지 않았다. 사용자의 실제
FEI 이미지(100,000x, 3.125 nm/px)에서 참값 약 70 nm가 137.39 nm로 나왔다.

`np.rot90`은 각도 a인 갭을 90 + a로 만든다(갭 축 (sin a, cos a)가 (cos a, -sin a)
로 옮겨간다). ROI는 회전 후 좌표계 기준이며, 정사각 대칭이라 같은 상자를 쓴다.
"""

import numpy as np
import pytest

from ebl_gap.measure import measure_roi
from ebl_gap.types import Roi, ScaleInfo
from tests.synth import synth_gap_image

ROI = Roi(106, 106, 405, 405)
NM_PER_PX = 1.0
SCALE = ScaleInfo(nm_per_px=NM_PER_PX, source="fei_metadata")
TOLERANCE_NM = 1.0 * NM_PER_PX


@pytest.mark.parametrize("gap_nm", [20.0, 30.0, 50.0, 80.0, 100.0])
@pytest.mark.parametrize("angle_deg", [0.0, 2.0, 5.0, 10.0])
@pytest.mark.parametrize("noise_sigma", [0.0, 3.0, 8.0])
def test_a_horizontal_gap_is_within_one_pixel_of_truth(gap_nm, angle_deg,
                                                       noise_sigma):
    img = np.rot90(synth_gap_image(
        width=512, height=512, gap_nm=gap_nm, nm_per_px=NM_PER_PX,
        angle_deg=angle_deg, edge_sigma_px=1.5, noise_sigma=noise_sigma,
        edge_bright=15.0, seed=int(gap_nm * 100 + angle_deg * 10 + noise_sigma),
    ))
    result = measure_roi(img, ROI, SCALE)
    assert result.mean_nm is not None, f"측정 실패: {result.warnings}"
    assert result.angle_deg == pytest.approx(90.0 + angle_deg, abs=0.5)
    assert result.mean_nm == pytest.approx(gap_nm, abs=TOLERANCE_NM)


@pytest.mark.parametrize("gap_nm", [30.0, 50.0, 100.0])
def test_a_horizontal_gap_at_coarse_pixels_still_lands_within_one_pixel(gap_nm):
    """100k배 촬영에 해당하는 3 nm/px — 사용자가 실제로 쓴 배율대다."""
    nm_per_px = 3.0
    img = np.rot90(synth_gap_image(
        width=512, height=512, gap_nm=gap_nm, nm_per_px=nm_per_px,
        angle_deg=3.0, edge_sigma_px=1.2, noise_sigma=4.0, edge_bright=15.0,
        seed=int(gap_nm),
    ))
    result = measure_roi(img, ROI, ScaleInfo(nm_per_px, "fei_metadata"))
    assert result.mean_nm == pytest.approx(gap_nm, abs=1.0 * nm_per_px)


# --- 사용자 제보 사례 -------------------------------------------------------
#
# FEI, 100,000x, 3.125 nm/px, 가로로 누운 S/D 갭, 참값 약 70 nm.
# 툴이 내놓은 값: **137.39 nm**.
#
#   137.39 / 70.0 = 1.963배
#   자동 추정 31.52도, 가로 갭의 정답은 90도 -> 오차 58.48도
#   1 / cos(58.48도) = 1.913배,  70 / cos(58.48도) = 133.9 nm
#
# ROI는 사용자가 실제로 잡은 모양(가로 500 x 세로 75)이다. 이 납작한 상자가
# 중요하다 — 정사각 ROI에서는 `uv_extent`의 맞바꿈이 항등이라 위쪽 정확도
# 격자로는 상자 회전 버그를 잡을 수 없다.

USER_NM_PER_PX = 3.125
USER_SCALE = ScaleInfo(nm_per_px=USER_NM_PER_PX, source="fei_metadata")
USER_GAP_NM = 70.0
USER_ROI = Roi(262, 434, 761, 508)  # 가로 500 x 세로 75, 갭 중심(471행)에 맞춤


def _user_image():
    """사용자 이미지와 같은 모양(1024x943)의 가로 갭. 갭 중심은 471행.

    이 합성 이미지는 수정 전 엔진에서 제보와 거의 같은 자리에 떨어졌다:
    각도 32.45도(제보 31.52도), 평균 134.92 nm(제보 137.39 nm), 참값의 1.927배
    (제보 1.963배). 기울기 1.5도와 seed 1은 그 재현을 위해 고른 값이다.
    """
    return np.rot90(synth_gap_image(
        width=943, height=1024, gap_nm=USER_GAP_NM, nm_per_px=USER_NM_PER_PX,
        angle_deg=1.5, edge_sigma_px=1.2, noise_sigma=8.0, edge_bright=15.0,
        seed=1,
    ))


def test_the_reported_field_case_lands_within_one_pixel():
    result = measure_roi(_user_image(), USER_ROI, USER_SCALE)
    assert result.mean_nm is not None, f"측정 실패: {result.warnings}"
    assert result.angle_deg == pytest.approx(91.5, abs=0.5)
    assert result.mean_nm == pytest.approx(USER_GAP_NM, abs=USER_NM_PER_PX)
    # 부풀던 값과는 1픽셀이 아니라 20픽셀 넘게 떨어져 있어야 한다.
    assert abs(result.mean_nm - 137.39) > 20 * USER_NM_PER_PX


def test_the_flat_user_roi_samples_only_its_own_75_rows():
    """납작한 ROI가 90도에서 세로로 ±250픽셀을 훑으면 데이터바까지 내려간다.

    라인 수가 ROI 가로(500)와 같고 프로파일 길이가 ROI 세로(75)와 같아야 한다.
    """
    result = measure_roi(_user_image(), USER_ROI, USER_SCALE)
    assert len(result.lines) == USER_ROI.width == 500
    for line in result.lines:
        if line.left_px is not None:
            assert 0.0 <= line.left_px <= USER_ROI.height - 1
            assert 0.0 <= line.right_px <= USER_ROI.height - 1
