import pytest

from ebl_gap.stats import mark_outliers, representative_line, summarize
from ebl_gap.types import LineResult, ScaleInfo

SCALE = ScaleInfo(nm_per_px=1.0, source="fei_metadata")


def line(row, width_nm, status="valid", flags=frozenset()):
    return LineResult(row=row, left_px=100.0,
                      right_px=None if width_nm is None else 100.0 + width_nm,
                      width_px=width_nm, width_nm=width_nm, status=status,
                      flags=flags, reason="")


def valid_lines(widths):
    return [line(i, w) for i, w in enumerate(widths)]


def test_mean_and_std_use_valid_lines_only():
    lines = valid_lines([50.0, 52.0, 48.0]) + [
        line(3, None, status="short"),
        line(4, 999.0, status="no_edge"),
    ]
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert result.mean_nm == pytest.approx(50.0)
    assert result.std_nm == pytest.approx(2.0)
    assert result.n_valid == 3
    assert result.n_short == 1
    assert result.n_uncertain == 1


def test_low_confidence_lines_are_included_in_the_mean():
    """플래그는 통계 포함 여부를 바꾸지 않는다."""
    lines = valid_lines([50.0, 50.0]) + [
        line(2, 50.0, flags=frozenset({"low_confidence"}))
    ]
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert result.n_valid == 3
    assert result.n_low_confidence == 1
    assert result.mean_nm == pytest.approx(50.0)


def test_mean_is_none_when_no_line_is_valid():
    result = summarize([line(0, None, status="short")], scale=SCALE, angle_deg=0.0)
    assert result.mean_nm is None
    assert result.std_nm is None


def test_std_is_none_with_a_single_valid_line():
    result = summarize(valid_lines([50.0]), scale=SCALE, angle_deg=0.0)
    assert result.mean_nm == pytest.approx(50.0)
    assert result.std_nm is None


def test_mark_outliers_relabels_far_lines_and_leaves_the_rest():
    """값이 전부 같아 MAD가 0이어도, 분해능이 척도가 되어 판정이 살아 있어야 한다.

    여기서 판정을 포기하면 이상치 제거가 가장 필요한 상황에서 기능이 꺼진다.
    """
    lines = valid_lines([50.0] * 20 + [200.0])
    marked = mark_outliers(lines, resolution_nm=1.0)
    assert marked[-1].status == "outlier"
    assert "척도" in marked[-1].reason
    assert all(m.status == "valid" for m in marked[:-1])


def test_mark_outliers_does_nothing_without_any_scale():
    """MAD도 0이고 분해능도 모르면 판정 기준 자체가 없다. 전부 남겨야 한다."""
    marked = mark_outliers(valid_lines([50.0] * 10))
    assert all(m.status == "valid" for m in marked)


def test_mark_outliers_does_nothing_when_most_values_tie():
    """MAD가 0이면 편차 판정이 불가능하다. 근소한 차이를 이상치로 만들면 안 된다.

    값이 전부 같은 경우만 테스트하면 가드를 지워도 통과한다 — 모든 편차가 정확히
    0이라 limit=0과 비교해도 걸리지 않기 때문이다. 일부만 같은 경우가 진짜 함정이고,
    픽셀 양자화 때문에 여러 라인이 같은 nm 값으로 떨어지는 것은 실측에서 흔하다.
    """
    marked = mark_outliers(valid_lines([50.0] * 9 + [51.0]), resolution_nm=1.0)
    assert all(m.status == "valid" for m in marked)


def test_mark_outliers_ignores_non_valid_lines():
    lines = valid_lines([50.0] * 10) + [line(10, 999.0, status="no_edge")]
    marked = mark_outliers(lines)
    assert marked[-1].status == "no_edge"


def test_short_ratio_at_or_above_five_percent_warns():
    lines = valid_lines([50.0] * 19) + [line(19, None, status="short")]
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert any("short" in w for w in result.warnings)


def test_uncertain_ratio_at_or_above_twenty_percent_warns():
    lines = valid_lines([50.0] * 8) + [line(8 + i, None, status="no_edge")
                                       for i in range(2)]
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert any("ROI 재설정" in w for w in result.warnings)


def test_fewer_than_ten_valid_lines_warns():
    result = summarize(valid_lines([50.0] * 9), scale=SCALE, angle_deg=0.0)
    assert any("유효 라인 부족" in w for w in result.warnings)


def test_high_relative_spread_warns():
    lines = valid_lines([30.0, 50.0, 70.0] * 5)
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert any("편차" in w for w in result.warnings)


def test_non_metadata_scale_source_warns():
    manual = ScaleInfo(nm_per_px=1.0, source="manual")
    result = summarize(valid_lines([50.0] * 20), scale=manual, angle_deg=0.0)
    assert any("스케일" in w for w in result.warnings)


def test_low_confidence_lines_produce_an_actionable_warning():
    lines = [line(i, 6.0, flags=frozenset({"low_confidence"})) for i in range(20)]
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert any("배율" in w for w in result.warnings)


def test_extra_warnings_are_appended():
    result = summarize(valid_lines([50.0] * 20), scale=SCALE, angle_deg=0.0,
                       extra_warnings=("각도 추정 실패",))
    assert "각도 추정 실패" in result.warnings


def test_clean_measurement_produces_no_warnings():
    result = summarize(valid_lines([50.0, 50.5, 49.5] * 10), scale=SCALE,
                       angle_deg=1.2)
    assert result.warnings == ()
    assert result.angle_deg == pytest.approx(1.2)


def test_representative_line_is_the_median_width_valid_line():
    """대표 라인은 폭이 중앙값에 가장 가까운 valid 라인이다."""
    lines = [
        LineResult(row=0, left_px=0.0, right_px=10.0, width_px=10.0,
                   width_nm=10.0, status="valid", flags=frozenset(), reason=""),
        LineResult(row=1, left_px=0.0, right_px=30.0, width_px=30.0,
                   width_nm=30.0, status="valid", flags=frozenset(), reason=""),
        LineResult(row=2, left_px=0.0, right_px=20.0, width_px=20.0,
                   width_nm=20.0, status="valid", flags=frozenset(), reason=""),
    ]
    assert representative_line(lines).row == 2


def test_representative_line_falls_back_to_the_first_line():
    """valid가 하나도 없으면 첫 줄을 돌려준다.

    전부 short인 이미지에서도 '왜 short인지'를 보여줄 프로파일이 필요하다.
    """
    lines = [LineResult(row=7, left_px=None, right_px=None, width_px=None,
                        width_nm=None, status="short", flags=frozenset(),
                        reason="대비 없음")]
    assert representative_line(lines).row == 7
    assert representative_line([]) is None
