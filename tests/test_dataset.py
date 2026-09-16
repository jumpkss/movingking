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


def test_uncertain_counts_are_summed_across_rois_onto_the_point():
    """판정보류 라인 수가 dose 점까지 따라와야 한다.

    곡선이 short 비율을 계산하는 분모가 이것을 포함한다. `DosePoint`가 들고
    오지 않으면 곡선은 결과 패널과 다른 분모를 쓸 수밖에 없다.
    """
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(40.0, n_uncertain=5),
                                        roi_result(60.0, n_uncertain=7)]))
    point = session.dose_curve()[0]
    assert point.n_uncertain == 12
    assert point.n_total == point.n_valid + point.n_short + point.n_uncertain


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


def test_a_closed_dose_counts_uncertain_lines_in_its_total():
    """전 구간 short 줄의 분모는 short + 판정보류다.

    리포트가 "280/300 라인"으로 찍는 그 분모다. 판정보류를 빼면 "280/280"이
    되어 20줄이 조용히 사라지고, 읽는 사람은 그 ROI가 한 줄도 남김없이
    short였다고 믿게 된다.
    """
    session = Session()
    session.add(record("b.tif", 400.0,
                       [roi_result(None, n_valid=0, n_short=280, n_uncertain=20)]))

    closed = session.closed_doses()

    assert [(c.n_short, c.n_total) for c in closed] == [(280, 300)]


def test_closed_doses_are_sorted_by_dose():
    session = Session()
    for dose in (500.0, 300.0, 400.0):
        session.add(record(f"{dose:g}.tif", dose,
                           [roi_result(None, n_valid=0, n_short=300)]))
    assert [c.dose for c in session.closed_doses()] == [300.0, 400.0, 500.0]


def test_a_dose_with_only_uncertain_lines_is_not_a_closed_dose():
    """판정보류(no_edge 등)뿐인 이미지는 "전 구간 short"가 아니라 "못 쟀다"이다.

    이 검사가 거르는 것은 딱 그것뿐이다. 빗나간 ROI는 걸러 주지 못한다 —
    평탄한 금속 위의 ROI는 no_edge가 아니라 전 구간 short를 내고, 엔진은
    그것을 닫힌 갭과 구별할 수 없다. 그 몫은 리포트와 곡선의 문구가 진다.
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


def test_a_session_with_no_measured_dose_at_all_warns_about_the_roi():
    """모든 dose가 전 구간 short면 dose test가 아니라 설정 문제일 가능성이 높다.

    한 dose가 닫히는 것은 정상이다. **모든** dose가 닫히는 dose test는 말이
    되지 않는다 — ROI가 패턴을 벗어났거나 스케일/문턱이 틀렸을 쪽이 훨씬 그럴듯
    하다. 엔진은 한 장만 봐서는 둘을 구별할 수 없지만, 세션 전체는 볼 수 있다.
    """
    session = Session()
    for dose in (300.0, 400.0, 500.0):
        session.add(record(f"{dose:g}.tif", dose,
                           [roi_result(None, n_valid=0, n_short=300)]))

    warnings = session.session_warnings()

    assert len(warnings) == 1
    assert "ROI" in warnings[0]


def test_one_measured_dose_is_enough_to_silence_the_session_warning():
    """측정된 점이 하나라도 있으면 설정은 멀쩡하다. 진짜로 닫힌 dose에 매번
    "ROI를 확인하세요"가 붙으면 그 문장이 무시된다."""
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(55.0)]))
    session.add(record("b.tif", 400.0, [roi_result(None, n_valid=0, n_short=300)]))

    assert session.session_warnings() == []


def test_an_empty_session_says_nothing():
    """아직 아무것도 측정하지 않은 세션에 설정 경고를 붙이면 안 된다."""
    assert Session().session_warnings() == []
