import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from ebl_gap.edges import analyze_profile  # noqa: E402
from ebl_gap.types import LineResult  # noqa: E402
from ebl_gap_gui.profile_plot import ProfilePlot  # noqa: E402


def sample_profile(n=201, gap=20.0):
    x = np.arange(n, dtype=float)
    d = np.abs(x - (n - 1) / 2.0)
    return np.where(d < gap / 2.0, 40.0, 200.0)


def valid_line(analysis, row=7):
    return LineResult(row=row, left_px=analysis.left_px,
                      right_px=analysis.right_px, width_px=analysis.width_px,
                      width_nm=analysis.width_px * 3.0, status="valid",
                      flags=frozenset(), reason="")


def test_starts_empty(qapp):
    assert ProfilePlot().has_curve() is False


def test_showing_a_line_draws_the_profile(qapp):
    profile = sample_profile()
    analysis = analyze_profile(profile)
    plot = ProfilePlot()
    plot.show_line(profile, valid_line(analysis), analysis)
    assert plot.has_curve() is True


def test_title_reports_the_row_width_and_status(qapp):
    profile = sample_profile()
    analysis = analyze_profile(profile)
    plot = ProfilePlot()
    plot.show_line(profile, valid_line(analysis, row=42), analysis)
    title = plot.title_text()
    assert "42" in title
    assert "valid" in title
    assert "nm" in title


def test_title_says_so_for_an_unmeasured_line(qapp):
    profile = np.full(201, 180.0)
    analysis = analyze_profile(profile)
    line = LineResult(row=3, left_px=None, right_px=None, width_px=None,
                      width_nm=None, status="short", flags=frozenset(),
                      reason="대비 없음")
    plot = ProfilePlot()
    plot.show_line(profile, line, analysis)
    assert "short" in plot.title_text()
    assert plot.has_curve() is True


def test_clear_removes_the_curve(qapp):
    profile = sample_profile()
    analysis = analyze_profile(profile)
    plot = ProfilePlot()
    plot.show_line(profile, valid_line(analysis), analysis)
    plot.clear()
    assert plot.has_curve() is False
    assert plot.title_text() == ""


def test_threshold_lines_follow_the_engine_fraction(qapp):
    """문턱선은 엔진이 쓴 값을 따라간다. 위젯이 0.5를 박아두면 여기서 갈라진다."""
    profile = np.array([200.0] * 20 + [40.0] * 10 + [160.0] * 20)
    analysis = analyze_profile(profile, threshold_fraction=0.3)
    plot = ProfilePlot()

    plot.show_line(profile, valid_line(analysis), analysis)

    levels = sorted(item.value() for item in plot.threshold_lines())
    assert levels == pytest.approx([76.0, 88.0])
