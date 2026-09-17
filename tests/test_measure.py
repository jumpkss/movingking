from pathlib import Path

import numpy as np
import pytest

from ebl_gap.loader import LoadedImage
from ebl_gap.measure import (
    DatabarOverlapError,
    MeasureParams,
    measure_loaded,
    measure_roi,
)
from ebl_gap.types import ImageRecord, Roi, ScaleInfo
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


def test_a_locked_angle_outside_the_plausible_range_still_warns():
    """고정은 검사 면제가 아니다. -70도를 잘못 넣은 사용자도 알아야 한다.

    고정 자체를 되돌리라는 경고는 무의미하지만, 범위를 벗어난 값은 고정했든
    추정했든 똑같이 틀린 갭 폭을 낸다.
    """
    result = measure_roi(_angle_collapse_image(), HEALTHY_ROI, FEI_SCALE,
                         angle_deg=-70.0)
    warning = next((w for w in result.warnings if "고정한 각도" in w), None)
    assert warning is not None, result.warnings
    assert "-70.0도" in warning
    # 이미 고정한 사람에게 고정을 권하면 따를 수 있는 조언이 남지 않는다.
    assert "고정하세요" not in warning


def test_a_locked_angle_inside_the_range_passes_quietly():
    """범위 안의 고정값까지 경고하면 경고가 잡음이 된다."""
    result = measure_roi(_angle_collapse_image(), HEALTHY_ROI, FEI_SCALE,
                         angle_deg=-12.0)
    assert not any("각도" in w for w in result.warnings), result.warnings


def test_the_locked_and_estimated_angle_warnings_read_differently():
    """두 경고가 권하는 다음 행동이 다르다 — 문구도 달라야 한다.

    추정이 무너졌으면 ROI를 고쳐야 하고, 고정값이 이상하면 입력한 숫자를
    고쳐야 한다. 한 문구로 합치면 둘 중 한쪽은 엉뚱한 곳을 보게 된다.
    """
    estimated = measure_roi(_angle_collapse_image(), COLLAPSING_ROI, FEI_SCALE)
    locked = measure_roi(_angle_collapse_image(), HEALTHY_ROI, FEI_SCALE,
                         angle_deg=-70.0)
    estimated_warning = next(w for w in estimated.warnings if "각도" in w)
    locked_warning = next(w for w in locked.warnings if "각도" in w)

    assert "ROI" in estimated_warning and "고정하세요" in estimated_warning
    assert "ROI" not in locked_warning and "의도한" in locked_warning


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


def test_a_tilted_roi_that_swings_into_the_databar_is_refused():
    """회전 전 상자만 보면 통과하지만, 실제로 훑는 행은 데이터바 안이다.

    `roi.y1 >= databar_top`은 회전 **전** 상자로 검사하는데 `extract_profiles`는
    `angle_deg`만큼 돌린 사각형을 훑는다. 모서리는 최대 (width/2)*|sin θ| 행만큼
    아래로 내려간다. 리뷰어 실측: y1=451, databar_top=452에서 0도면 short 0%,
    12도면 8.9% — 5% 문턱을 넘어 dose 곡선에 빨간 X가 날조된 데이터로 붙는다.
    엔진은 각도를 아는 첫 번째 자리이므로 각도가 정해진 뒤에 검사한다.
    """
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0)
    img[452:, :] = 10.0
    roi = Roi(106, 151, 406, 451)  # 가로 301px, y1은 데이터바 바로 위 한 줄

    # 0도에서는 지금과 똑같이 통과해야 한다.
    assert measure_roi(img, roi, SCALE, angle_deg=0.0,
                       databar_top=452) is not None

    with pytest.raises(DatabarOverlapError, match="데이터바") as caught:
        measure_roi(img, roi, SCALE, angle_deg=12.0, databar_top=452)
    assert "483" in str(caught.value), "몇 행까지 훑는지를 말해 줘야 한다"


def _loaded(pixels, databar_top, scale=SCALE):
    """load_image가 돌려주는 것과 같은 모양의 LoadedImage."""
    record = ImageRecord(path=Path("pattern_300uC.tif"), scale=scale, dose=300.0)
    return LoadedImage(record=record, pixels=pixels, databar_top=databar_top)


def test_measure_loaded_passes_the_databar_row_the_caller_never_typed():
    """노트북 경로가 데이터바 보호를 자동으로 받는다.

    `measure_roi`의 `databar_top`은 기본 None이라 호출자가 잊으면 거부가 통째로
    꺼진다. 리뷰어 실측: README 형태로 부르면 26.9%의 날조된 short 비율이 그대로
    나왔다. `LoadedImage`는 그 행을 이미 들고 있으므로, 잊을 수 있는 자리를
    아예 없앤다 — 예제만 고치면 다음 사람이 또 잊는다.
    """
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    img[380:, :] = 10.0
    with pytest.raises(DatabarOverlapError, match="데이터바"):
        measure_loaded(_loaded(img, databar_top=380), ROI)


def test_measure_loaded_measures_exactly_what_measure_roi_would():
    """편의 함수가 다른 답을 내면 두 경로가 갈라진다."""
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    direct = measure_roi(img, ROI, SCALE)
    convenience = measure_loaded(_loaded(img, databar_top=None), ROI)
    assert convenience.mean_nm == pytest.approx(direct.mean_nm)
    assert convenience.scale is SCALE


def test_measure_loaded_threads_params_and_a_locked_angle():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0, angle_deg=6.0,
                          edge_sigma_px=3.0)
    loaded = _loaded(img, databar_top=None)
    narrow = measure_loaded(loaded, ROI, angle_deg=6.0,
                            params=MeasureParams(threshold_fraction=0.3))
    wide = measure_loaded(loaded, ROI, angle_deg=6.0,
                          params=MeasureParams(threshold_fraction=0.7))
    assert narrow.angle_deg == pytest.approx(6.0)
    assert narrow.mean_nm < 40.0 < wide.mean_nm


def test_measure_loaded_says_so_when_the_scale_is_not_settled():
    """스케일을 못 읽은 이미지(PNG 크롭 등)에 AttributeError 대신 할 말을 준다."""
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    with pytest.raises(ValueError, match="스케일"):
        measure_loaded(_loaded(img, databar_top=None, scale=None), ROI)


# --- 가로 갭 (Task 28) ------------------------------------------------------

FLAT_ROI = Roi(10, 262, 509, 336)  # 가로 500 x 세로 75 — 사용자의 실제 ROI 모양
#: 600x600 이미지를 눕히면 갭 중심이 y=299.5에 온다. FLAT_ROI는 거기에 맞춰 놓았다.


def _horizontal_gap_image(**kwargs):
    """세로 갭 합성 이미지를 눕힌 것. rot90은 참값을 보존한다."""
    return np.rot90(synth_gap_image(**kwargs))


def test_a_vertical_measurement_reaches_exactly_the_roi_bottom():
    """각도 90도에서 도달 범위는 상자 그대로여야 한다.

    `roi.width`를 측정 방향 표본 수로 가정한 계산은 90도에서 가로 500짜리 ROI가
    세로로 250행 더 내려간다고 본다. 그러면 데이터바 근처의 멀쩡한 ROI가 거부된다.
    """
    from ebl_gap.measure import _lowest_scanned_row

    assert _lowest_scanned_row(FLAT_ROI, 90.0) == FLAT_ROI.y1
    assert _lowest_scanned_row(FLAT_ROI, 0.0) == FLAT_ROI.y1
    # 88도: 측정 방향이 세로이므로 갭 축(=ROI 가로 500) 쪽이 cos만큼 내려간다.
    assert _lowest_scanned_row(FLAT_ROI, 88.0) == FLAT_ROI.y1 + 9


def test_a_horizontal_gap_just_above_the_databar_is_not_refused():
    img = _horizontal_gap_image(width=600, height=600, gap_nm=40.0,
                                nm_per_px=1.0)
    result = measure_roi(img, FLAT_ROI, SCALE, angle_deg=90.0,
                         databar_top=FLAT_ROI.y1 + 1)
    assert result.mean_nm is not None, result.warnings


def test_a_horizontal_gap_is_measured_without_an_angle_warning():
    """가로 갭의 정답인 90도가 타당성 문턱에 걸리면 안 된다.

    문턱을 절댓값으로 보면 90도는 언제나 경고다. 기준에서 얼마나 벗어났는지로
    봐야 한다.
    """
    img = _horizontal_gap_image(width=600, height=600, gap_nm=40.0,
                                nm_per_px=1.0)
    result = measure_roi(img, FLAT_ROI, SCALE)
    assert result.angle_deg == pytest.approx(90.0, abs=0.5)
    assert not any("각도" in w for w in result.warnings), result.warnings


def test_a_locked_angle_far_from_the_horizontal_base_still_warns():
    img = _horizontal_gap_image(width=600, height=600, gap_nm=40.0,
                                nm_per_px=1.0)
    result = measure_roi(img, FLAT_ROI, SCALE, angle_deg=150.0)
    warning = next((w for w in result.warnings if "고정한 각도" in w), None)
    assert warning is not None, result.warnings
    assert "가로 갭" in warning and "기준 90도" in warning


def test_a_small_tilt_on_the_vertical_base_stays_quiet():
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0)
    result = measure_roi(img, ROI, SCALE, angle_deg=2.0)
    assert not any("각도" in w for w in result.warnings), result.warnings


def test_a_large_tilt_on_the_vertical_base_warns_and_names_the_base():
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0)
    result = measure_roi(img, ROI, SCALE, angle_deg=40.0)
    warning = next((w for w in result.warnings if "고정한 각도" in w), None)
    assert warning is not None, result.warnings
    assert "세로 갭" in warning and "기준 0도" in warning
