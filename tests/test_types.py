import pytest

from ebl_gap.types import (
    LINE_STATUSES,
    UNCERTAIN_STATUSES,
    LineResult,
    Roi,
    RoiResult,
    ScaleInfo,
)


def test_scale_info_rejects_unknown_source():
    with pytest.raises(ValueError, match="source"):
        ScaleInfo(nm_per_px=3.0, source="guess")


def test_scale_info_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="nm_per_px"):
        ScaleInfo(nm_per_px=0.0, source="manual")


def test_roi_normalizes_reversed_drag():
    roi = Roi(x0=400, y0=300, x1=100, y1=50)
    assert (roi.x0, roi.y0, roi.x1, roi.y1) == (100, 50, 400, 300)
    assert roi.width == 301
    assert roi.height == 251
    assert roi.cx == pytest.approx(250.0)
    assert roi.cy == pytest.approx(175.0)


def test_roi_rejects_degenerate_box():
    with pytest.raises(ValueError, match="ROI"):
        Roi(x0=10, y0=10, x1=10, y1=40)


def test_line_result_counts_in_stats_only_when_valid():
    base = dict(row=0, left_px=10.0, right_px=30.0, width_px=20.0,
                width_nm=60.0, flags=frozenset(), reason="")
    assert LineResult(status="valid", **base).counts_in_stats is True
    for status in ("short", "no_edge", "multi_edge", "sub_resolution", "outlier"):
        assert LineResult(status=status, **base).counts_in_stats is False


def test_line_result_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        LineResult(row=0, left_px=None, right_px=None, width_px=None,
                   width_nm=None, status="weird", flags=frozenset(), reason="")


def test_uncertain_statuses_is_subset_of_line_statuses():
    assert set(UNCERTAIN_STATUSES) < set(LINE_STATUSES)
    assert "valid" not in UNCERTAIN_STATUSES
    assert "short" not in UNCERTAIN_STATUSES


def test_roi_result_holds_lines_and_warnings():
    scale = ScaleInfo(nm_per_px=3.0, source="fei_metadata")
    result = RoiResult(mean_nm=60.0, std_nm=2.0, n_valid=100, n_short=3,
                       n_uncertain=5, n_low_confidence=12, angle_deg=1.5,
                       lines=(), warnings=("확인 필요",), scale=scale)
    assert result.n_total == 108
    assert result.warnings == ("확인 필요",)
