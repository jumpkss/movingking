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
