import numpy as np
import pytest

from ebl_gap.measure import MeasureParams, measure_roi
from ebl_gap.types import Roi, ScaleInfo
from tests.synth import synth_gap_image

SCALE = ScaleInfo(nm_per_px=1.0, source="fei_metadata")
ROI = Roi(106, 106, 405, 405)


def test_measures_every_row_of_the_roi():
    """이미지 최대 해상도를 그대로 쓴다: ROI 세로 픽셀 수만큼 라인이 나온다."""
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    result = measure_roi(img, ROI, SCALE)
    assert len(result.lines) == ROI.height == 300
    assert [ln.row for ln in result.lines] == list(range(300))


def test_estimates_the_angle_when_not_given():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0, angle_deg=6.0)
    result = measure_roi(img, ROI, SCALE)
    assert result.angle_deg == pytest.approx(6.0, abs=0.2)


def test_uses_the_supplied_angle_without_estimating():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0, angle_deg=6.0)
    result = measure_roi(img, ROI, SCALE, angle_deg=0.0)
    assert result.angle_deg == pytest.approx(0.0)
    # 각도를 무시하면 폭이 1/cos(6도) = 0.55% 과대평가된다.
    assert result.mean_nm > 40.0


def test_falls_back_to_zero_angle_with_a_warning_when_estimation_fails():
    img = np.full((512, 512), 180.0)
    result = measure_roi(img, ROI, SCALE)
    assert result.angle_deg == pytest.approx(0.0)
    assert any("각도" in w for w in result.warnings)


def test_a_closed_gap_is_reported_as_short_not_as_a_measurement():
    img = np.full((512, 512), 200.0)
    rng = np.random.default_rng(0)
    img = img + rng.normal(0.0, 3.0, img.shape)
    result = measure_roi(img, ROI, SCALE)
    assert result.mean_nm is None
    assert result.n_short == 300
    assert any("short" in w for w in result.warnings)


def test_widths_are_converted_to_nanometres_with_the_given_scale():
    img = synth_gap_image(gap_nm=60.0, nm_per_px=1.0)
    coarse = ScaleInfo(nm_per_px=2.0, source="manual")
    result = measure_roi(img, ROI, coarse)
    # 이미지의 갭은 60픽셀이고 스케일이 2 nm/px이므로 120 nm로 읽혀야 한다.
    assert result.mean_nm == pytest.approx(120.0, abs=2.0)


def test_params_are_threaded_through_to_the_edge_detector():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0, edge_sigma_px=3.0)
    narrow = measure_roi(img, ROI, SCALE,
                         params=MeasureParams(threshold_fraction=0.3)).mean_nm
    wide = measure_roi(img, ROI, SCALE,
                       params=MeasureParams(threshold_fraction=0.7)).mean_nm
    assert narrow < 40.0 < wide


def test_locally_filled_rows_do_not_drag_the_mean():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    img[200:205, 240:280] = 200.0  # 5개 행의 갭 일부를 금속으로 메운다
    result = measure_roi(img, ROI, SCALE)
    assert result.mean_nm == pytest.approx(40.0, abs=1.0)
    # 부분적으로 메워진 행은 에지 검출 실패로 uncertain이 된다
    assert result.n_uncertain == 5
    assert result.n_valid + result.n_short + result.n_uncertain == 300


def test_a_widened_row_is_rejected_as_an_outlier():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    img[200:205, 200:312] = 40.0  # 5개 행의 갭만 세 배 가까이 넓힌다
    result = measure_roi(img, ROI, SCALE)
    assert result.mean_nm == pytest.approx(40.0, abs=1.0)
    assert any(ln.status == "outlier" for ln in result.lines)


def test_result_carries_the_scale_it_was_measured_with():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    result = measure_roi(img, ROI, SCALE)
    assert result.scale is SCALE


# 리뷰어가 실측한 각도 붕괴 조건의 재현. 1024x943, 70 nm 갭, 3.0517578125 nm/px,
# 실제 기울기 2도. ROI를 x0=460에 놓으면 갭이 ROI 왼쪽 평탄부 구간(양 끝 20%)에
# 걸쳐 행마다 에지가 엉뚱하게 잡히고, Theil-Sen 적합이 -18.5도로 접힌다.
FEI_SCALE = ScaleInfo(nm_per_px=3.0517578125, source="fei_metadata")
COLLAPSING_ROI = Roi(460, 100, 760, 400)
HEALTHY_ROI = Roi(362, 100, 662, 400)


def _angle_collapse_image():
    return synth_gap_image(width=1024, height=943, gap_nm=70.0,
                           nm_per_px=3.0517578125, angle_deg=2.0,
                           edge_sigma_px=1.2, noise_sigma=8.0,
                           edge_bright=15.0, seed=7)


def test_an_implausible_fitted_angle_gets_a_warning_that_names_the_cause():
    """각도가 무너진 측정은 그 사실을 말해야 한다.

    이 ROI는 평균 74 nm(참값 70.0, +1.3픽셀)를 유효 라인 85줄과 표준편차
    0.4 nm로 내놓는다 — 숫자만 보면 정밀하다. 유일하게 뜨던 경고는
    `short 발생 구간 있음`이라 원인을 잘못 짚었다.
    """
    result = measure_roi(_angle_collapse_image(), COLLAPSING_ROI, FEI_SCALE)
    assert result.angle_deg < -15.0
    assert result.mean_nm > 71.0  # 참값 70.0에서 1픽셀(3.05 nm) 넘게 부풀었다
    assert any("갭 축 각도" in w for w in result.warnings), result.warnings


def test_a_normal_tilt_does_not_trigger_the_angle_warning():
    """스펙이 상정한 0~10도 범위 안의 기울기는 경고 없이 지나가야 한다."""
    result = measure_roi(_angle_collapse_image(), HEALTHY_ROI, FEI_SCALE)
    assert result.angle_deg == pytest.approx(2.0, abs=0.3)
    assert not any("갭 축 각도" in w for w in result.warnings), result.warnings


def test_an_roi_that_reaches_into_the_databar_is_refused():
    """데이터바를 걸친 ROI는 측정하지 않는다 (스펙 4.1).

    측정하면 데이터바의 균일한 띠가 라인마다 short로 판정돼 날조된 이상 비율이
    나온다. 이 툴이 보고하려고 존재하는 바로 그 신호에 가짜가 섞인다.
    """
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    img[380:, :] = 10.0  # 아래쪽 데이터바
    with pytest.raises(ValueError, match="데이터바"):
        measure_roi(img, ROI, SCALE, databar_top=380)


def test_an_roi_above_the_databar_is_measured_normally():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    result = measure_roi(img, ROI, SCALE, databar_top=460)
    assert result.mean_nm == pytest.approx(40.0, abs=1.0)


def test_databar_top_defaults_to_none_for_callers_that_cannot_know_it():
    """노트북 사용처럼 데이터바 위치를 모르는 호출은 그대로 동작한다."""
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    assert measure_roi(img, ROI, SCALE).mean_nm == pytest.approx(40.0, abs=1.0)


def test_the_first_databar_row_already_counts_as_intrusion():
    """경계는 '데이터바 첫 줄에 닿으면 거부'다. ROI는 양 끝 픽셀을 포함한다."""
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    with pytest.raises(ValueError, match="데이터바"):
        measure_roi(img, ROI, SCALE, databar_top=ROI.y1)
    assert measure_roi(img, ROI, SCALE, databar_top=ROI.y1 + 1) is not None
