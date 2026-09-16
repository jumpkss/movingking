import pytest

from ebl_gap.classify import classify_line
from ebl_gap.edges import ProfileAnalysis


def make_analysis(**overrides) -> ProfileAnalysis:
    """판정 테스트용 ProfileAnalysis를 직접 만든다.

    이미지를 거치지 않고 레벨 값을 직접 지정해야 경계값을 정확히 찌를 수 있다.
    """
    fields = dict(i_hi_left=200.0, i_hi_right=200.0, i_lo=40.0, sigma_noise=4.0,
                  min_index=100, left_px=90.0, right_px=110.0,
                  n_cross_left=1, n_cross_right=1, threshold_fraction=0.5)
    fields.update(overrides)
    return ProfileAnalysis(**fields)


def test_healthy_line_is_valid_with_no_flags():
    status, flags, reason = classify_line(make_analysis())
    assert status == "valid"
    assert flags == frozenset()
    assert reason == ""


def test_low_contrast_is_short_even_when_edges_were_found():
    """대비가 노이즈 수준이면 찾았다는 에지는 잡음이다. short가 먼저다."""
    analysis = make_analysis(i_hi_left=50.0, i_hi_right=50.0, i_lo=40.0,
                             sigma_noise=4.0)
    status, _, reason = classify_line(analysis)
    assert status == "short"
    assert "대비" in reason


def test_zero_contrast_with_zero_noise_is_still_short():
    """노이즈 없는 합성 이미지에서 0 <= 0 이 short로 잡혀야 한다."""
    analysis = make_analysis(i_hi_left=100.0, i_hi_right=100.0, i_lo=100.0,
                             sigma_noise=0.0)
    assert classify_line(analysis)[0] == "short"


def test_missing_edge_is_no_edge():
    assert classify_line(make_analysis(left_px=None))[0] == "no_edge"
    assert classify_line(make_analysis(right_px=None))[0] == "no_edge"


def test_extra_crossings_are_multi_edge():
    assert classify_line(make_analysis(n_cross_left=2))[0] == "multi_edge"
    assert classify_line(make_analysis(n_cross_right=3))[0] == "multi_edge"


def test_width_below_three_pixels_is_sub_resolution():
    analysis = make_analysis(left_px=100.0, right_px=102.9)
    status, flags, reason = classify_line(analysis)
    assert status == "sub_resolution"
    assert flags == frozenset()
    assert "3" in reason


def test_width_exactly_three_pixels_is_valid_not_sub_resolution():
    """경계는 포함이다: min_width_px 이상이면 valid."""
    analysis = make_analysis(left_px=100.0, right_px=103.0)
    assert classify_line(analysis)[0] == "valid"


def test_width_between_three_and_ten_pixels_gets_low_confidence_flag():
    status, flags, _ = classify_line(make_analysis(left_px=100.0, right_px=105.0))
    assert status == "valid"
    assert flags == frozenset({"low_confidence"})


def test_width_exactly_ten_pixels_is_not_low_confidence():
    """경계는 배제다: low_confidence_px 이상이면 플래그가 붙지 않는다."""
    _, flags, _ = classify_line(make_analysis(left_px=100.0, right_px=110.0))
    assert flags == frozenset()


def test_thresholds_are_tunable():
    analysis = make_analysis(left_px=100.0, right_px=105.0)
    assert classify_line(analysis, min_width_px=6.0)[0] == "sub_resolution"
    assert classify_line(analysis, low_confidence_px=4.0)[1] == frozenset()


def test_short_check_runs_before_multi_edge():
    analysis = make_analysis(i_hi_left=50.0, i_hi_right=50.0, i_lo=40.0,
                             sigma_noise=4.0, n_cross_left=5)
    assert classify_line(analysis)[0] == "short"
