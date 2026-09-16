from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from ebl_gap.dataset import Session  # noqa: E402
from ebl_gap.types import ImageRecord, RoiResult, ScaleInfo  # noqa: E402
from ebl_gap_gui.panels import FilePanel, ResultPanel, ResultTable  # noqa: E402

SCALE = ScaleInfo(nm_per_px=3.0, source="fei_metadata")


def make_result(mean_nm=42.0, warnings=()):
    return RoiResult(mean_nm=mean_nm, std_nm=1.5, n_valid=280, n_short=12,
                     n_uncertain=8, n_low_confidence=30, angle_deg=2.4,
                     lines=(), warnings=tuple(warnings), scale=SCALE)


def make_records():
    return [
        ImageRecord(path=Path("a_300uC.tif"), scale=SCALE, dose=300.0,
                    roi_results=[make_result(60.0)]),
        ImageRecord(path=Path("b_400uC.tif"), scale=SCALE, dose=400.0,
                    roi_results=[make_result(30.0)]),
    ]


def test_file_panel_lists_one_row_per_record(qapp):
    panel = FilePanel()
    panel.set_records(make_records())
    assert panel.row_count() == 2
    assert panel.current_index() == 0


def test_file_panel_emits_selection_changes(qapp):
    panel = FilePanel()
    panel.set_records(make_records())
    seen = []
    panel.selection_changed.connect(seen.append)
    panel.select(1)
    assert seen[-1] == 1
    assert panel.current_index() == 1


def test_file_panel_shows_the_parsed_dose(qapp):
    panel = FilePanel()
    panel.set_records(make_records())
    assert panel.dose_text(0) == "300"
    assert panel.dose_text(1) == "400"


def test_file_panel_shows_an_empty_dose_cell_when_unknown(qapp):
    panel = FilePanel()
    panel.set_records([ImageRecord(path=Path("x.tif"), scale=SCALE, dose=None)])
    assert panel.dose_text(0) == ""


def test_setting_a_dose_updates_the_record_and_the_cell(qapp):
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)
    panel.set_dose(0, 275.0)
    assert records[0].dose == pytest.approx(275.0)
    assert panel.dose_text(0) == "275"


def test_file_panel_marks_measured_and_unmeasured_rows(qapp):
    records = make_records()
    records[1].roi_results = []
    panel = FilePanel()
    panel.set_records(records)
    assert "측정" in panel.status_text(0)
    assert panel.status_text(1) == "미측정"


def test_file_panel_marks_load_errors(qapp):
    record = ImageRecord(path=Path("bad.tif"), error="메타데이터 없음")
    panel = FilePanel()
    panel.set_records([record])
    assert "오류" in panel.status_text(0)


def test_result_panel_reports_the_mean_counts_and_scale_source(qapp):
    panel = ResultPanel()
    panel.show_result(make_result(42.0))
    text = panel.text()
    assert "42.00" in text
    assert "280" in text  # 유효 라인
    assert "12" in text   # short 라인
    assert "fei_metadata" in text
    assert "2.4" in text  # 각도


def test_result_panel_lists_warnings(qapp):
    panel = ResultPanel()
    panel.show_result(make_result(warnings=("short 발생 구간 있음, 확인 필요",)))
    assert "short 발생" in panel.text()


def test_result_panel_says_so_when_there_is_no_measurement(qapp):
    panel = ResultPanel()
    panel.show_result(RoiResult(mean_nm=None, std_nm=None, n_valid=0,
                                n_short=300, n_uncertain=0, n_low_confidence=0,
                                angle_deg=0.0, lines=(), warnings=(),
                                scale=SCALE))
    assert "측정 불가" in panel.text()


def test_result_panel_clears(qapp):
    panel = ResultPanel()
    panel.show_result(make_result())
    panel.clear()
    assert panel.text().strip() == ""


def test_result_table_has_one_row_per_measured_roi(qapp):
    session = Session()
    for record in make_records():
        session.add(record)
    table = ResultTable()
    table.set_session(session)
    assert table.row_count() == 2
    assert table.cell(0, "file") == "a_300uC.tif"
    assert table.cell(0, "mean_nm") == "60.00"


def test_result_table_is_emptied_by_a_fresh_session(qapp):
    table = ResultTable()
    session = Session()
    session.add(make_records()[0])
    table.set_session(session)
    table.set_session(Session())
    assert table.row_count() == 0
