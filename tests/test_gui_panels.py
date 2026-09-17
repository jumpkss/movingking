from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from ebl_gap.dataset import Session  # noqa: E402
from ebl_gap.types import ImageRecord, RoiResult, ScaleInfo  # noqa: E402
from ebl_gap_gui.panels import FilePanel, ResultPanel, ResultTable  # noqa: E402
from tests.synth import synth_gap_image  # noqa: E402

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


def test_editing_the_dose_cell_updates_the_record_and_emits(qapp):
    """실제 사용자가 타는 경로: 셀 텍스트를 직접 고친다.

    set_dose()는 _loading으로 막힌 경로를 지나므로 _on_item_changed를 거치지
    않는다. 그것만 테스트하면 편집 처리가 통째로 깨져도 통과한다.
    """
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)
    seen = []
    panel.dose_edited.connect(lambda i, v: seen.append((i, v)))
    panel._table.item(0, 1).setText("275")
    assert records[0].dose == pytest.approx(275.0)
    assert seen == [(0, 275.0)]


def test_clearing_the_dose_cell_means_unknown(qapp):
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)
    seen = []
    panel.dose_edited.connect(lambda i, v: seen.append((i, v)))
    panel._table.item(0, 1).setText("")
    assert records[0].dose is None
    assert seen == [(0, None)]


def test_unparseable_dose_reverts_and_does_not_emit(qapp):
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)
    seen = []
    panel.dose_edited.connect(lambda i, v: seen.append((i, v)))
    panel._table.item(0, 1).setText("abc")
    assert records[0].dose == pytest.approx(300.0)
    assert panel.dose_text(0) == "300"
    assert seen == []


def test_negative_dose_reverts(qapp):
    """음수 dose는 물리적으로 불가능하다. 조용히 받으면 곡선 x축이 틀어진다."""
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)
    seen = []
    panel.dose_edited.connect(lambda i, v: seen.append((i, v)))
    panel._table.item(0, 1).setText("-50")
    assert records[0].dose == pytest.approx(300.0)
    assert seen == []


def test_reloading_onto_the_same_row_still_announces_the_selection(qapp):
    """0행이 선택된 채 다른 폴더를 열면 Qt는 신호를 보내지 않는다.

    그대로 두면 결과 패널이 이전 폴더의 결과를 계속 보여준다.
    """
    panel = FilePanel()
    panel.set_records(make_records())
    seen = []
    panel.selection_changed.connect(seen.append)
    panel.set_records(make_records())
    assert seen == [0]


def test_loading_flag_is_cleared_even_when_filling_a_row_raises(qapp):
    """예외로 _loading이 True로 남으면 이후 dose 편집이 전부 조용히 무시된다.

    이 상태는 눈에 보이지 않는다 — 사용자는 dose를 고쳤는데 아무 일도 일어나지
    않는 것만 본다. 그래서 플래그 불변식을 직접 확인한다.

    예외 뒤에 성공하는 set_records를 한 번 끼워 넣고 셀 편집으로 확인하려 하면
    안 된다. 그 호출이 try/finally 없이도 자기 끝에서 플래그를 내려버려서,
    버그가 있든 없든 통과하는 테스트가 된다.
    """
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)

    broken = ImageRecord(path=Path("x.tif"), scale=SCALE, dose=1.0)
    broken.path = None  # _fill_row에서 .name 접근이 터진다
    with pytest.raises(AttributeError):
        panel.set_records([broken])

    assert panel._loading is False


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


def test_thumbnail_starts_absent(qapp):
    panel = FilePanel()
    panel.set_records(make_records())
    assert panel.has_thumbnail(0) is False


def test_thumbnail_reaches_the_table_item(qapp):
    """썸네일이 실제 표 항목에 붙는다.

    has_thumbnail()은 딕셔너리만 본다. _fill_row의 setIcon 세 줄을 통째로
    지워도 그 검사는 전부 통과했다 — 기능이 죽어도 초록이었다. 그래서 여기서는
    사용자가 실제로 보는 것, 즉 표 항목의 아이콘을 직접 읽는다.
    """
    panel = FilePanel()
    panel.set_records(make_records())
    panel.set_thumbnail(0, synth_gap_image(gap_nm=50.0))

    icon = panel._table.item(0, 0).icon()
    assert icon.isNull() is False
    assert icon.availableSizes()          # 실제 픽스맵이 들어 있다


def test_setting_a_thumbnail_attaches_an_icon(qapp):
    import numpy as np

    panel = FilePanel()
    panel.set_records(make_records())
    panel.set_thumbnail(0, np.arange(64 * 64, dtype=float).reshape(64, 64))
    assert panel.has_thumbnail(0) is True
    assert panel.has_thumbnail(1) is False
    # 딕셔너리만이 아니라 화면에 붙은 것까지 본다.
    assert panel._table.item(0, 0).icon().isNull() is False
    assert panel._table.item(1, 0).icon().isNull() is True


def test_empty_pixels_do_not_attach_a_thumbnail(qapp):
    import numpy as np

    panel = FilePanel()
    panel.set_records(make_records())
    panel.set_thumbnail(0, np.empty((0, 0)))
    assert panel.has_thumbnail(0) is False
    assert panel._table.item(0, 0).icon().isNull() is True


def test_thumbnail_survives_a_row_refresh(qapp):
    import numpy as np

    panel = FilePanel()
    panel.set_records(make_records())
    panel.set_thumbnail(0, np.full((32, 32), 120.0))
    panel.refresh_row(0)
    assert panel.has_thumbnail(0) is True
    # refresh_row가 항목을 새로 만들므로 아이콘을 다시 붙이지 않으면 사라진다.
    assert panel._table.item(0, 0).icon().availableSizes()


# --- 방향 판별 표시 (Task 28) ----------------------------------------------


def _angled_result(angle_deg):
    return RoiResult(mean_nm=70.0, std_nm=1.5, n_valid=280, n_short=0,
                     n_uncertain=0, n_low_confidence=0, angle_deg=angle_deg,
                     lines=(), warnings=(), scale=SCALE)


def test_the_result_panel_names_the_gap_orientation(qapp):
    """91.2도는 오타가 아니라 가로 갭이다. 그 말을 옆에 적어 준다.

    이 한 줄이 없으면 사용자는 90도 근처의 각도를 추정 붕괴로 읽는다 —
    실제로는 가로 갭에서 정상이다.
    """
    panel = ResultPanel()
    panel.show_result(_angled_result(91.2))
    assert "갭 각도 91.20도 (가로 갭)" in panel.text()

    panel.show_result(_angled_result(2.4))
    assert "갭 각도 2.40도 (세로 갭)" in panel.text()
