import csv
from dataclasses import replace

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from ebl_gap_gui.app import MainWindow  # noqa: E402
from tests.synth import synth_gap_image  # noqa: E402
from tests.test_metadata import FEI_INI  # noqa: E402


def write_sample(path, gap_nm, dose):
    """3 nm/px 메타데이터가 붙은 512x512 합성 SEM TIFF를 만든다."""
    # 합성 이미지가 3.0 nm/px이므로 메타데이터도 정확히 3.0 nm/px로 맞춘다.
    # 어긋나면 측정값에 계통 오차가 생겨 테스트가 무엇을 재는지 흐려진다.
    ini = (FEI_INI
           .replace("ResolutionX=1024", "ResolutionX=512")
           .replace("ResolutionY=884", "ResolutionY=512")
           .replace("PixelWidth=3.0517578125e-009", "PixelWidth=3.0e-009")
           .replace("PixelHeight=3.0517578125e-009", "PixelHeight=3.0e-009")
           .replace("HorFieldsize=3.125e-006", "HorFieldsize=1.536e-006"))
    img = synth_gap_image(width=512, height=512, gap_nm=gap_nm, nm_per_px=3.0,
                          angle_deg=2.0, edge_sigma_px=1.2, noise_sigma=3.0,
                          seed=int(dose))
    data = np.clip(img, 0, 255).astype(np.uint8)
    out = path / f"pattern_{dose:g}uC.tif"
    tifffile.imwrite(out, data, extratags=[(34682, 's', 0, ini, True)])
    return out


@pytest.fixture()
def folder(tmp_path):
    write_sample(tmp_path, gap_nm=90.0, dose=300)
    write_sample(tmp_path, gap_nm=60.0, dose=400)
    return tmp_path


def test_open_folder_loads_every_image(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    assert len(window.session.records) == 2
    assert all(r.scale is not None for r in window.session.records)


def test_open_folder_reads_dose_from_filenames(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    assert sorted(r.dose for r in window.session.records) == [300.0, 400.0]


def test_measure_current_produces_a_result_near_the_true_gap(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    record = window.session.records[0]
    assert len(record.roi_results) == 1
    assert record.roi_results[0].mean_nm == pytest.approx(90.0, abs=3.0)


def test_measuring_twice_replaces_rather_than_appends(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    window.measure_current()
    assert len(window.session.records[0].roi_results) == 1


def test_measurement_updates_the_result_panel_and_table(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    assert "nm" in window.result_panel.text()
    assert window.result_table.row_count() == 1


def test_measuring_both_images_fills_the_dose_curve(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    for index in (0, 1):
        window.select_image(index)
        window.measure_current()
    assert window.dose_plot.point_count() == 2


def test_dragging_the_roi_clears_the_stale_overlay(qapp, folder):
    """측정 후 ROI를 옮기면 이전 위치의 에지 오버레이가 남아 있으면 안 된다."""
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    assert window.image_view.has_overlay() is True
    window.image_view._roi.setPos([120, 130])
    assert window.image_view.has_overlay() is False


def test_export_summary_csv_has_a_row_per_measurement(qapp, folder, tmp_path):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    out = tmp_path / "summary.csv"
    window.export_summary_csv(out)
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert len(rows) == 2  # 측정 1건 + 미측정 1건
    assert any(row["mean_nm"] for row in rows)


def test_export_overlay_writes_a_png(qapp, folder, tmp_path):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    out = tmp_path / "overlay.png"
    window.export_overlay(out)
    assert out.exists() and out.stat().st_size > 0


def test_export_report_mentions_the_scale_source(qapp, folder, tmp_path):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    out = tmp_path / "report.txt"
    window.export_report(out)
    assert "fei_metadata" in out.read_text(encoding="utf-8")


def test_measuring_without_a_scale_is_refused_with_a_message(qapp, tmp_path):
    plain = tmp_path / "plain_300uC.tif"
    tifffile.imwrite(plain, np.zeros((256, 256), dtype=np.uint8))
    window = MainWindow()
    window.open_folder(tmp_path)
    window.select_image(0)
    window.measure_current()
    assert window.session.records[0].roi_results == []
    assert "스케일" in window.status_text()


def test_empty_folder_is_handled_without_crashing(qapp, tmp_path):
    window = MainWindow()
    window.open_folder(tmp_path)
    assert window.session.records == []
    window.measure_current()  # 예외 없이 지나가야 한다


def test_open_folder_attaches_thumbnails(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    assert window.file_panel.has_thumbnail(0) is True
    assert window.file_panel.has_thumbnail(1) is True


def test_measuring_shows_a_representative_profile(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    assert window.profile_plot.has_curve() is True
    assert "valid" in window.profile_plot.title_text()


def test_show_line_switches_to_the_requested_row(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    window.show_line(5)
    assert "행 5" in window.profile_plot.title_text()


def test_show_line_is_ignored_before_measuring(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.show_line(5)
    assert window.profile_plot.has_curve() is False


def test_selecting_another_image_clears_the_profile_plot(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    window.select_image(1)
    assert window.profile_plot.has_curve() is False
    # 플래그가 아니라 그려진 항목을 본다. clear()가 _plot.clear()를 빼먹어도
    # has_curve()는 False라고 답하므로 플래그만으로는 아무것도 못 지킨다.
    assert window.profile_plot.threshold_lines() == []


def test_switching_to_an_unreadable_image_drops_the_previous_profiles(qapp,
                                                                      tmp_path):
    """읽지 못한 이미지로 넘어가면 이전 이미지의 프로파일이 남으면 안 된다.

    빈 배열은 ImageView.set_image에서 일찍 빠져 roi_changed가 나지 않는다.
    그래서 ROI 경로가 대신 비워 주지 못하는 유일한 경로다. 여기서 배열이
    남으면 show_line이 앞 이미지의 프로파일을 현재 이미지의 것으로 그린다.
    """
    write_sample(tmp_path, gap_nm=90.0, dose=300)
    # 이름이 z로 시작해야 sorted()에서 뒤로 간다 — 정상 파일이 0번이어야 측정이 된다.
    (tmp_path / "zbroken.tif").write_bytes(b"not a tiff at all")
    window = MainWindow()
    window.open_folder(tmp_path)
    window.measure_current()
    assert window._profiles is not None

    window.select_image(1)

    assert window._profiles is None
    assert window.profile_plot.has_curve() is False


def test_open_folder_selects_first_image_exactly_once(qapp, folder, monkeypatch):
    """폴더 열기가 첫 장을 정확히 한 번만 선택한다.

    set_records가 selection_changed(0)을 동기 발신하므로 명시 호출을 더하면
    렌더와 ROI 리셋이 두 배로 돈다. 상태가 깨지지는 않지만 낭비이고, 무엇보다
    '정확히 한 번'이라는 set_records의 불변식을 무너뜨린다.
    """
    window = MainWindow()

    calls: list[int] = []
    original = window.select_image
    monkeypatch.setattr(window, "select_image",
                        lambda index: (calls.append(index), original(index))[1])

    window.open_folder(folder)

    assert calls == [0]
    assert window._current == 0


def test_moving_the_roi_clears_the_profile_plot(qapp, folder):
    """ROI를 옮기면 이전 위치의 프로파일이 남으면 안 된다.

    미니 플롯은 오버레이와 마찬가지로 위치에 묶여 있다. 사용자가 ROI를 옮겨
    놓고 플롯을 보면 다른 자리의 프로파일을 현재 자리의 것으로 읽는다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    assert window.profile_plot.has_curve()

    window.image_view._roi.setPos([120, 130])

    assert window.profile_plot.has_curve() is False
    assert window._profiles is None
    # 플래그가 아니라 그려진 항목을 본다. clear()가 _plot.clear()를 빼먹어도
    # has_curve()는 False라고 답하므로 플래그만으로는 아무것도 못 지킨다.
    assert window.profile_plot.threshold_lines() == []


def test_representative_line_is_the_median_width_valid_line(qapp, folder):
    """대표 라인은 폭이 중앙값에 가장 가까운 valid 라인이다.

    첫 줄로 바꿔도 통과하던 자리다. 평균이 어떤 프로파일에서 나왔는지
    보여주는 것이 이 플롯의 목적이므로 규칙 자체를 고정한다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()

    result = window.session.records[0].roi_results[0]
    valid = [ln for ln in result.lines
             if ln.status == "valid" and ln.width_nm is not None]
    widths = sorted(ln.width_nm for ln in valid)
    expected = min(valid,
                   key=lambda ln: abs(ln.width_nm - widths[len(widths) // 2])).row

    assert window.profile_plot.row() == expected
    assert expected != result.lines[0].row  # 첫 줄로 퇴화하면 무의미한 검사다


@pytest.fixture()
def folder_with_short(tmp_path):
    """갭이 닫힌 이미지 한 장. 모든 라인이 valid가 아닌 상태로 판정된다."""
    write_sample(tmp_path, gap_nm=0.0, dose=100)
    return tmp_path


def test_line_spinbox_shows_that_line(qapp, folder):
    """스핀박스에 직접 타이핑한 행이 플롯에 뜬다."""
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()

    window.line_selector.setValue(12)      # 실제 위젯 값 변경 -> 신호

    assert window.profile_plot.row() == 12


def test_line_spinbox_is_disabled_until_measured(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    assert window.line_selector.isEnabled() is False
    window.measure_current()
    assert window.line_selector.isEnabled() is True
    assert window.line_selector.maximum() == len(
        window.session.records[0].roi_results[0].lines) - 1


def test_next_anomaly_button_jumps_to_a_non_valid_line(qapp, folder_with_short):
    """'다음 이상' 버튼이 valid가 아닌 다음 라인으로 간다.

    이 버튼이 특이사항 분리를 실제로 쓸 수 있게 만드는 부분이다.
    """
    window = MainWindow()
    window.open_folder(folder_with_short)
    window.measure_current()
    result = window.session.records[0].roi_results[0]
    anomalies = [ln.row for ln in result.lines if ln.status != "valid"]
    assert anomalies, "픽스처가 이상 라인을 만들어야 이 테스트가 뜻이 있다"

    window.line_selector.setValue(0)
    window.next_anomaly_button.click()     # 실제 클릭

    assert window.profile_plot.row() in anomalies
    assert window.profile_plot.row() == min(r for r in anomalies if r > 0)


def test_anomaly_button_is_disabled_when_every_line_is_valid(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    result = window.session.records[0].roi_results[0]
    assert all(ln.status == "valid" for ln in result.lines), \
        "픽스처가 전부 valid여야 이 테스트가 뜻이 있다"
    assert window.next_anomaly_button.isEnabled() is False


@pytest.mark.parametrize("size", [(1400, 900), (1000, 700)])
def test_the_file_name_fits_beside_the_thumbnail_at_both_sizes(qapp, folder,
                                                               size):
    """좁은 창에서도 파일 이름이 살아 있다.

    픽셀 값을 박아두지 않고 관계를 본다: 이름을 그리는 데 필요한 폭이 썸네일을
    뺀 나머지 칸 폭 안에 들어가야 한다. 넓은 창 하나만 검사하면 여유가 커서
    어느 변경을 되돌려도 통과한다. dose가 안 잡히는 파일에서는 이름이 행을
    구분하는 유일한 수단이다.
    """
    window = MainWindow()
    window.resize(*size)
    window.show()
    qapp.processEvents()
    window.open_folder(folder)
    qapp.processEvents()

    table = window.file_panel._table
    name = table.item(0, 0).text()
    needed = table.fontMetrics().horizontalAdvance(name)
    available = table.columnWidth(0) - table.iconSize().width()
    assert available >= needed, (
        f"{size}에서 이름 '{name}'에 {needed}px 필요한데 {available}px 남는다")


def test_rows_are_tall_enough_for_the_thumbnail(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    table = window.file_panel._table
    assert table.rowHeight(0) >= table.iconSize().height()


def test_profile_plot_has_usable_height_at_the_default_geometry(qapp):
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    qapp.processEvents()
    assert window.profile_plot.height() >= 180


def _mark_anomalies(window, rows):
    """현재 이미지의 결과에서 주어진 행을 이상 라인으로 바꾼 뒤 돌려준다.

    RoiResult는 frozen이므로 제자리 대입이 아니라 replace로 갈아 끼운다.
    상태를 직접 심어 이상 라인 위치를 안다 — 합성 이미지가 어떤 상태를 낼지에
    기대지 않는 편이 픽스처가 바뀌어도 테스트가 뜻을 잃지 않는다.
    """
    record = window.session.records[window._current]
    result = record.roi_results[0]
    marked = set(rows)
    record.roi_results[0] = replace(
        result,
        lines=tuple(replace(ln, status="no_edge") if ln.row in marked else ln
                    for ln in result.lines),
    )
    return record.roi_results[0]


def test_anomaly_buttons_step_both_ways_and_wrap(qapp, folder):
    """이상 라인 사이를 앞뒤로 오가고, 끝에서 반대쪽으로 감는다.

    valid 라인에는 절대 서지 않는다 — 이상 라인만 보려고 누르는 버튼이다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    result = window.session.records[0].roi_results[0]
    marked = {3, 40, 100, 150, result.lines[-1].row}
    result = _mark_anomalies(window, marked)
    window._arm_line_selector(result, representative_row=0)

    window.line_selector.setValue(0)
    seen = []
    for _ in range(6):
        window.next_anomaly_button.click()
        seen.append(window.profile_plot.row())
    assert seen == [3, 40, 100, 150, max(marked), 3]   # 끝에서 감긴다

    back = []
    for _ in range(3):
        window.prev_anomaly_button.click()
        back.append(window.profile_plot.row())
    assert back == [max(marked), 150, 100]
    assert all(row in marked for row in seen + back)


def test_measure_draws_the_representative_line_exactly_once(qapp, folder,
                                                            monkeypatch):
    """무장 중 새는 valueChanged가 같은 라인을 두 번 그리게 두지 않는다.

    지금은 _profiles가 먼저 채워져 있어 두 번 그려도 결과가 같지만, 호출
    순서가 바뀌는 순간 무장 도중의 신호가 빈 배열을 그리게 된다.
    """
    window = MainWindow()
    window.open_folder(folder)
    drawn: list[int] = []
    original = window.show_line
    monkeypatch.setattr(window, "show_line",
                        lambda row: (drawn.append(row), original(row))[1])

    window.measure_current()

    assert drawn == [window.line_selector.value()], drawn


def test_the_bottom_dock_does_not_swallow_the_window(qapp):
    """바닥 도크가 중앙 영역을 잡아먹지 않는다.

    resizeDocks가 없으면 도크가 창의 60%를 가져가 이미지 뷰가 눌린다.
    갭을 보려고 여는 프로그램에서 이미지가 가장 작으면 안 된다.
    """
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    qapp.processEvents()
    assert window.centralWidget().height() >= window.height() // 2


def test_moving_the_roi_disables_the_line_controls(qapp, folder_with_short):
    """플롯이 비었으면 선택기도 죽어야 한다. 살아 있으면 눌러도 아무 일이
    없는 죽은 버튼이 된다.

    이상 라인이 있는 픽스처를 쓴다. 전부 valid인 폴더에서는 두 버튼이 애초에
    꺼져 있어서, 세 줄 중 스핀박스 한 줄만 고정된다.
    """
    window = MainWindow()
    window.open_folder(folder_with_short)
    window.measure_current()
    assert window.line_selector.isEnabled()
    assert window.next_anomaly_button.isEnabled()
    assert window.prev_anomaly_button.isEnabled()

    window.image_view._roi.setPos([120, 130])

    assert window.line_selector.isEnabled() is False
    assert window.next_anomaly_button.isEnabled() is False
    assert window.prev_anomaly_button.isEnabled() is False


def test_single_anomaly_button_says_so(qapp, folder):
    """이상 라인이 하나뿐이면 왜 안 움직이는지 말해 준다.

    같은 행을 다시 그리기만 하면 화면이 그대로라 고장난 버튼과 구별되지 않는다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    result = _mark_anomalies(window, {50})
    window._arm_line_selector(result, representative_row=50)

    window.next_anomaly_button.click()

    assert window.profile_plot.row() == 50
    assert "하나" in window.status_text()
