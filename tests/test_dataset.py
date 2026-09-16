from pathlib import Path

import pytest

from ebl_gap.dataset import DEFAULT_DOSE_PATTERN, ClosedDose, Session, parse_dose
from ebl_gap.types import ImageRecord, RoiResult, ScaleInfo

SCALE = ScaleInfo(nm_per_px=3.0, source="fei_metadata")


def roi_result(mean_nm, n_valid=100, n_short=0, n_uncertain=0, scale=SCALE):
    return RoiResult(mean_nm=mean_nm, std_nm=1.0, n_valid=n_valid,
                     n_short=n_short, n_uncertain=n_uncertain, n_low_confidence=0,
                     angle_deg=0.0, lines=(), warnings=(), scale=scale)


def record(name, dose, results, scale=SCALE):
    return ImageRecord(path=Path(name), scale=scale, dose=dose,
                       roi_results=list(results))


@pytest.mark.parametrize("name,expected", [
    ("pattern_320uC_01.tif", 320.0),
    ("SD_gap_280uc.tif", 280.0),
    ("dose-test-412.5uC-run2.tif", 412.5),
    ("300 uC scan.tif", 300.0),
    ("no_dose_here.tif", None),
    ("run_01.tif", None),
])
def test_parse_dose_reads_the_microcoulomb_figure(name, expected):
    assert parse_dose(name) == expected


def test_parse_dose_accepts_a_custom_pattern():
    assert parse_dose("d0450_scan.tif", pattern=r"d(\d+)") == 450.0


def test_parse_dose_returns_none_on_an_unmatched_custom_pattern():
    assert parse_dose("pattern_320uC.tif", pattern=r"z(\d+)") is None


def test_default_pattern_is_case_insensitive():
    assert "(?i)" in DEFAULT_DOSE_PATTERN


def test_dose_curve_is_sorted_by_dose():
    session = Session()
    session.add(record("c.tif", 400.0, [roi_result(30.0)]))
    session.add(record("a.tif", 200.0, [roi_result(80.0)]))
    session.add(record("b.tif", 300.0, [roi_result(55.0)]))
    assert [p.dose for p in session.dose_curve()] == [200.0, 300.0, 400.0]
    assert [p.mean_nm for p in session.dose_curve()] == [80.0, 55.0, 30.0]


def test_dose_curve_skips_records_without_a_dose():
    session = Session()
    session.add(record("a.tif", None, [roi_result(80.0)]))
    session.add(record("b.tif", 300.0, [roi_result(55.0)]))
    assert [p.dose for p in session.dose_curve()] == [300.0]


def test_dose_curve_skips_records_with_no_measurement():
    session = Session()
    session.add(record("a.tif", 300.0, []))
    session.add(record("b.tif", 400.0, [roi_result(None, n_valid=0, n_short=300)]))
    assert session.dose_curve() == []


def test_multiple_rois_are_averaged_weighted_by_valid_line_count():
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(40.0, n_valid=100),
                                        roi_result(60.0, n_valid=300)]))
    point = session.dose_curve()[0]
    assert point.mean_nm == pytest.approx(55.0)  # (40*100 + 60*300) / 400
    assert point.n_valid == 400


def test_short_counts_are_summed_across_rois():
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(40.0, n_short=5),
                                        roi_result(60.0, n_short=7)]))
    assert session.dose_curve()[0].n_short == 12


def test_mixed_pixel_sizes_produce_a_warning():
    coarse = ScaleInfo(nm_per_px=6.0, source="fei_metadata")
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(40.0)], scale=SCALE))
    session.add(record("b.tif", 400.0, [roi_result(60.0, scale=coarse)],
                       scale=coarse))
    assert any("배율" in w for w in session.scale_warnings())


def test_consistent_pixel_sizes_produce_no_warning():
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(40.0)]))
    session.add(record("b.tif", 400.0, [roi_result(60.0)]))
    assert session.scale_warnings() == []


def test_a_fully_shorted_dose_is_reported_as_a_closed_dose():
    """갭이 닫힌 dose는 곡선에서 사라지면 안 된다.

    "이 dose에서 갭이 닫힌다"가 dose test의 답이다. 그 점이 빠진 곡선은 답의
    절반을 지운 것이고, 사용자가 dose를 고르는 곳이 바로 그 곡선이다.
    측정된 갭 폭이 없으므로 `DosePoint`로는 돌려주지 않는다 — mean_nm에 0.0을
    끼워 넣으면 그 0이 평균과 기울기 계산에 조용히 섞여 들어간다.
    """
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(55.0)]))
    session.add(record("b.tif", 400.0,
                       [roi_result(None, n_valid=0, n_short=300)]))

    assert [p.dose for p in session.dose_curve()] == [300.0]
    closed = session.closed_doses()
    assert [c.dose for c in closed] == [400.0]
    assert closed[0] == ClosedDose(dose=400.0, n_short=300, n_total=300,
                                   path=Path("b.tif"))


def test_closed_doses_are_sorted_by_dose():
    session = Session()
    for dose in (500.0, 300.0, 400.0):
        session.add(record(f"{dose:g}.tif", dose,
                           [roi_result(None, n_valid=0, n_short=300)]))
    assert [c.dose for c in session.closed_doses()] == [300.0, 400.0, 500.0]


def test_an_unmeasurable_roi_is_not_a_closed_dose():
    """판정보류뿐인 이미지는 "갭이 닫혔다"가 아니라 "못 쟀다"이다.

    ROI를 엉뚱한 데 놓아 전부 no_edge가 난 것을 갭 0으로 찍으면, 화면이
    측정하지 않은 결론을 대신 말하게 된다. 이 태스크가 막으려는 바로 그 종류다.
    """
    session = Session()
    session.add(record("a.tif", 400.0,
                       [roi_result(None, n_valid=0, n_short=0, n_uncertain=300)]))
    assert session.closed_doses() == []


def test_a_measured_dose_is_not_also_a_closed_dose():
    """short가 섞여 있어도 유효 라인이 있으면 갭이 닫힌 것이 아니다."""
    session = Session()
    session.add(record("a.tif", 400.0, [roi_result(30.0, n_valid=50, n_short=250)]))
    assert session.closed_doses() == []
    assert [p.dose for p in session.dose_curve()] == [400.0]


def test_closed_doses_skip_records_without_a_dose():
    session = Session()
    session.add(record("a.tif", None, [roi_result(None, n_valid=0, n_short=300)]))
    assert session.closed_doses() == []
