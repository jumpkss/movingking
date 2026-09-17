import csv
from dataclasses import replace

import numpy as np
import pytest
import tifffile
from PIL import Image

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QToolBar  # noqa: E402

from ebl_gap.types import Roi  # noqa: E402
from ebl_gap_gui.calibration import CalibrationDialog  # noqa: E402
from ebl_gap_gui.app import MainWindow  # noqa: E402
from tests.synth import synth_gap_image  # noqa: E402
from tests.test_metadata import FEI_INI  # noqa: E402


def _fei_ini():
    """512x512, 정확히 3.0 nm/px인 FEI 메타데이터 블록.

    합성 이미지가 3.0 nm/px이므로 메타데이터도 정확히 맞춘다. 어긋나면 측정값에
    계통 오차가 생겨 테스트가 무엇을 재는지 흐려진다.
    """
    return (FEI_INI
            .replace("ResolutionX=1024", "ResolutionX=512")
            .replace("ResolutionY=884", "ResolutionY=512")
            .replace("PixelWidth=3.0517578125e-009", "PixelWidth=3.0e-009")
            .replace("PixelHeight=3.0517578125e-009", "PixelHeight=3.0e-009")
            .replace("HorFieldsize=3.125e-006", "HorFieldsize=1.536e-006"))


def _write_tif(path, dose, data):
    out = path / f"pattern_{dose:g}uC.tif"
    tifffile.imwrite(out, data, extratags=[(34682, 's', 0, _fei_ini(), True)])
    return out


def write_sample(path, gap_nm, dose):
    """3 nm/px 메타데이터가 붙은 512x512 합성 SEM TIFF를 만든다."""
    img = synth_gap_image(width=512, height=512, gap_nm=gap_nm, nm_per_px=3.0,
                          angle_deg=2.0, edge_sigma_px=1.2, noise_sigma=3.0,
                          seed=int(dose))
    return _write_tif(path, dose, np.clip(img, 0, 255).astype(np.uint8))


def write_closed_sample(path, dose, noise_sigma=3.0):
    """갭이 완전히 닫힌 이미지 — 평탄한 금속과 잡음뿐이다.

    `gap_nm=0`으로 만든 합성 이미지에는 아직 좁은 골이 남아 `sub_resolution`으로
    판정된다. 실제로 닫힌 패턴은 대비 자체가 없어서 `short`가 되므로, 그 상태를
    만들려면 골 없는 평탄한 금속이어야 한다.
    """
    rng = np.random.default_rng(int(dose))
    data = np.clip(200.0 + rng.normal(0.0, noise_sigma, (512, 512)), 0, 255)
    return _write_tif(path, dose, data.astype(np.uint8))


def write_horizontal_sample(path, gap_nm, dose):
    """갭이 가로로 뻗은 합성 SEM TIFF. 사용자의 S/D 패턴이 이 방향이다."""
    img = synth_gap_image(width=512, height=512, gap_nm=gap_nm, nm_per_px=3.0,
                          angle_deg=88.0, edge_sigma_px=1.2, noise_sigma=3.0,
                          seed=int(dose))
    return _write_tif(path, dose, np.clip(img, 0, 255).astype(np.uint8))


def write_sample_with_databar(path, gap_nm, dose, databar_rows=60):
    """스캔 영역 아래에 어두운 데이터바를 붙이고 ResolutionY로 경계를 알린다."""
    scan_rows = 512 - databar_rows
    img = synth_gap_image(width=512, height=scan_rows, gap_nm=gap_nm,
                          nm_per_px=3.0, angle_deg=2.0, edge_sigma_px=1.2,
                          noise_sigma=3.0, seed=int(dose))
    data = np.full((512, 512), 10, dtype=np.uint8)
    data[:scan_rows] = np.clip(img, 0, 255).astype(np.uint8)
    out = path / f"bar_{dose:g}uC.tif"
    ini = _fei_ini().replace("ResolutionY=512", f"ResolutionY={scan_rows}")
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
    현재 것으로 남으면 show_line이 앞 이미지의 프로파일을 현재 이미지의 것으로
    그린다. 0번의 보관분 자체는 남아 있어야 한다 — 돌아오면 되살릴 것이다.
    """
    write_sample(tmp_path, gap_nm=90.0, dose=300)
    # 이름이 z로 시작해야 sorted()에서 뒤로 간다 — 정상 파일이 0번이어야 측정이 된다.
    (tmp_path / "zbroken.tif").write_bytes(b"not a tiff at all")
    window = MainWindow()
    window.open_folder(tmp_path)
    window.measure_current()
    assert window._profiles_by_index.get(0) is not None

    window.select_image(1)

    assert window._profiles_by_index.get(1) is None
    assert window.profile_plot.has_curve() is False
    window.show_line(5)
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
    assert window._profiles_by_index == {}
    assert window._measured_rois == {}
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


@pytest.mark.parametrize("size", [(1400, 900), (1000, 700), (900, 650)])
def test_the_file_name_fits_beside_the_thumbnail_at_every_size(qapp, folder,
                                                               size):
    """지원하는 어떤 창 크기에서도 파일 이름이 화면에 보인다.

    dose가 안 잡히는 파일에서는 이름이 행을 구분하는 유일한 수단이다. 넓은 창
    하나만 검사하면 여유가 커서 어느 변경을 되돌려도 통과하므로 여러 크기를 본다.
    가장 좁은 것이 지원 하한 900x650이다. 그보다 좁은 기하는 setMinimumSize가
    막으므로 도달할 수 없고, 도달 불가능한 상태를 검사하는 것은 검사가 아니다.
    """
    window = MainWindow()
    window.resize(*size)
    window.show()
    qapp.processEvents()
    window.open_folder(folder)
    qapp.processEvents()

    table = window.file_panel._table
    # 이름을 못 읽게 되는 길은 두 가지고, 둘 다 막아야 뜻이 있다.
    #
    # 하나: 열이 내용보다 좁아 이름이 "..."으로 줄어든다. 예전 단언이 쓰던
    # columnWidth - iconSize는 ResizeToContents 아래서 아이콘이 약분돼 상수가
    # 되므로 뜻이 없다. 대신 열이 내용의 크기 힌트를 담는지를 직접 본다 —
    # 아이콘·여백·글자를 Qt가 직접 합산한 값이라 환경이 달라도 성립한다.
    assert table.sizeHintForColumn(0) <= table.columnWidth(0), (
        f"{size}에서 이름 열이 {table.columnWidth(0)}px인데 내용에는 "
        f"{table.sizeHintForColumn(0)}px 필요하다 — 이름이 잘린다")
    # 둘: 열이 통째로 뷰포트 밖으로 밀려난다. 이때 이름은 줄어들지 않고 그냥
    # 화면에서 사라지므로 위 단언은 통과한다. 640x480이 그 상태였다.
    assert table.columnWidth(0) <= table.viewport().width(), (
        f"{size}에서 이름 열 {table.columnWidth(0)}px이 "
        f"뷰포트 {table.viewport().width()}px를 넘는다")


def _toolbar(window, title):
    bars = [b for b in window.findChildren(QToolBar) if b.windowTitle() == title]
    assert len(bars) == 1, f"{title} 도구 모음이 {len(bars)}개다"
    return bars[0]


@pytest.mark.parametrize("title", ["주요 동작", "측정 설정"])
@pytest.mark.parametrize("size", [(1400, 900), (900, 650)])
def test_no_toolbar_action_hides_behind_the_overflow_chevron(qapp, size, title):
    """지원 하한에서도 도구 모음의 동작이 하나도 가려지지 않아야 한다.

    측정 설정 도구 모음이 주 도구 모음과 한 줄을 나눠 쓰던 때는 900x650에서
    주 도구 모음이 578px 힌트 대비 404px로 눌려 `라인 CSV`, `오버레이 PNG`,
    `요약 리포트`가 오버플로 뒤로 숨었다. 눈으로 보지 않고 동작의 가시성과
    크기 힌트로 확인한다.
    """
    window = MainWindow()
    window.resize(*size)
    window.show()
    qapp.processEvents()

    bar = _toolbar(window, title)
    hidden = [action.text() for action in bar.actions()
              if not bar.widgetForAction(action).isVisible()]
    assert hidden == [], f"{size}에서 {hidden}이 오버플로 뒤에 숨는다"
    assert bar.sizeHint().width() <= bar.width(), (
        f"{size}에서 {title}이 {bar.width()}px인데 힌트는 "
        f"{bar.sizeHint().width()}px다")


def test_the_main_toolbar_takes_the_top_row(qapp):
    """주 도구 모음이 윗줄이어야 한다. 순서를 뒤집어도 아무도 안 잡았다.

    설정 도구 모음을 먼저 얹으면 그쪽이 윗줄을 잡고 주 도구 모음이 아랫줄로
    내려간다. 폴더 열기·측정·내보내기가 이 툴의 전부인데 각도·문턱 조절값
    아래에 놓이고, 좁은 창에서 먼저 잘려 나가는 쪽도 아랫줄이다.
    """
    window = MainWindow()
    window.resize(900, 650)
    window.show()
    qapp.processEvents()

    main_bar = _toolbar(window, "주요 동작")
    settings_bar = _toolbar(window, "측정 설정")

    assert main_bar.y() < settings_bar.y(), (
        f"주요 동작이 y={main_bar.y()}, 측정 설정이 y={settings_bar.y()}다 — "
        "주 도구 모음이 아랫줄로 내려갔다")


def test_the_window_refuses_to_shrink_below_its_usable_size(qapp):
    """더 좁아지면 파일 이름 열이 패널 밖으로 나간다. 지원 하한을 Qt가 지킨다.

    두 변을 따로 본다. 튜플 비교는 사전식이라 (1000, 100)도 (900, 650) 이상으로
    쳐 주는데, 높이가 눌린 창은 여기서 막으려는 바로 그 상태다.
    """
    window = MainWindow()
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    assert window.width() >= 900, f"폭 {window.width()}px이 하한 900px보다 좁다"
    assert window.height() >= 650, \
        f"높이 {window.height()}px이 하한 650px보다 낮다"


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

    지금은 보관분이 먼저 채워져 있어 두 번 그려도 결과가 같지만, 호출
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


def test_returning_to_a_measured_image_restores_its_diagnostics(qapp, folder):
    """측정한 이미지로 돌아오면 라인 진단이 그대로 있다.

    dose 시리즈를 오가며 보는 것이 이 프로그램의 사용 방식이다. 돌아올 때마다
    ROI를 다시 끌고 다시 측정해야 하면 이상 라인 확인을 포기하게 된다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    first_roi = window.image_view.current_roi()
    window.line_selector.setValue(40)

    window.select_image(1)
    window.select_image(0)

    assert window.image_view.current_roi() == first_roi
    assert window.line_selector.isEnabled()
    window.line_selector.setValue(40)
    assert window.profile_plot.row() == 40


def test_returning_to_a_measured_image_restores_the_edge_overlay(qapp, folder):
    """되살린 ROI 위에 에지 오버레이도 다시 그린다.

    ROI와 요약과 프로파일은 돌아왔는데 그림만 없으면, 이미지 뷰 혼자
    '이 장은 아직 안 쟀다'고 말하는 꼴이 된다. 오버레이는 되살린 그 ROI와
    그 결과에서 그대로 다시 나오므로 어긋날 여지가 없다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    assert window.image_view.has_overlay() is True

    window.select_image(1)
    window.select_image(0)

    assert window.image_view.has_overlay() is True


def test_a_moved_roi_is_not_restored_when_you_come_back(qapp, folder):
    """ROI를 옮겨 버린 뒤 돌아오면 되살릴 것이 없어야 한다.

    Task 20이 고친 결함이 보관 기능을 타고 되살아나는 경로다. 옮긴 ROI에
    이전 위치의 프로파일을 붙이면 x축이 다른 뜻인 채로 진단을 읽게 된다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    window.image_view._roi.setPos([120, 130])

    window.select_image(1)
    window.select_image(0)

    assert window.profile_plot.has_curve() is False
    assert window.line_selector.isEnabled() is False


def test_opening_another_folder_does_not_resurrect_old_profiles(qapp, folder,
                                                                tmp_path):
    """새 폴더의 0번이 이전 폴더 0번의 보관분을 물려받으면 안 된다.

    인덱스를 열쇠로 쓰므로 비우지 않으면 다른 시료의 진단이 그대로 붙는다.
    화면 단언만으로는 부족하다 — 새 레코드에는 roi_results가 없어서 복원
    경로가 어차피 비켜 가기 때문이다. 그래서 두 보관함 자체가 비었는지도 본다.
    """
    window = MainWindow()
    window.open_folder(folder)
    # 기본 위치 그대로 측정하면 새 폴더 0번의 기본 ROI와 값이 같아져서,
    # ROI가 되살아나도 테스트가 알아채지 못한다.
    moved = Roi(40, 50, 240, 250)
    window.image_view.set_roi(moved)
    window.measure_current()

    other = tmp_path / "other"
    other.mkdir()
    write_sample(other, gap_nm=30.0, dose=500)
    window.open_folder(other)

    assert window.line_selector.isEnabled() is False
    assert window.profile_plot.has_curve() is False
    assert window.image_view.current_roi() != moved
    assert window._profiles_by_index == {}
    assert window._measured_rois == {}


def test_exporting_after_moving_the_roi_does_not_draw_stale_edges(qapp, folder,
                                                                  tmp_path):
    """ROI를 옮긴 뒤 내보낸 PNG에 옛 에지가 그려지면 안 된다.

    화면은 이미 비워진다. 그런데 파일로 나가는 그림이 현재 ROI에 과거 결과를
    겹쳐 그리면, 실험 노트에 들어가는 것은 아무 에지도 없는 자리에 에지가
    찍힌 사진이다. 화면보다 이쪽이 더 오래 남는다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    window.image_view._roi.setPos([60, 60])

    out = tmp_path / "overlay.png"
    window.export_overlay(str(out))

    assert out.exists() is False
    assert "측정" in window.status_text()


def test_exporting_right_after_measuring_still_works(qapp, folder, tmp_path):
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    out = tmp_path / "overlay.png"
    window.export_overlay(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_recalibrating_after_measuring_does_not_rewrite_the_csv_scale(
        qapp, folder, tmp_path, monkeypatch):
    """측정 -> 재캘리브레이션 -> 요약 CSV. 행에는 측정에 쓴 스케일이 남는다.

    `calibrate_current`는 `record.scale`만 갈아 끼우고 `result.scale`은 그대로
    둔다 — mean_nm을 만든 스케일이 그쪽이기 때문이다. CSV가 record 쪽을 읽으면
    한 행 안의 mean_nm과 nm_per_px가 서로 다른 스케일을 가리켜, 그 행으로는
    아무것도 재현할 수 없다. 사용자가 실제로 밟는 순서로 확인한다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    assert window.session.records[0].roi_results[0].scale.source == "fei_metadata"

    class AcceptingDialog(CalibrationDialog):
        """실제 다이얼로그. 모달로 멈추는 exec만 대신한다 — 값은 위젯에 넣는다."""

        def exec(self):
            self.set_length(1.0, "µm")
            self.set_manual_pixels(200.0)   # 1000 nm / 200 px = 5.0 nm/px
            return CalibrationDialog.Accepted

    monkeypatch.setattr("ebl_gap_gui.app.CalibrationDialog", AcceptingDialog)
    window.calibrate_current()

    record = window.session.records[0]
    assert record.scale.nm_per_px == pytest.approx(5.0)
    assert record.scale.source == "manual"
    assert record.roi_results[0].scale.nm_per_px == pytest.approx(3.0)

    out = tmp_path / "summary.csv"
    window.export_summary_csv(out)
    measured = [row for row in csv.DictReader(out.open(encoding="utf-8-sig"))
                if row["mean_nm"]]
    assert len(measured) == 1
    assert float(measured[0]["nm_per_px"]) == pytest.approx(3.0)
    assert measured[0]["scale_source"] == "fei_metadata"


def test_a_dose_whose_gap_closed_stays_on_the_curve_and_in_the_report(qapp,
                                                                      tmp_path):
    """갭이 닫힌 dose가 곡선과 리포트에 남는다. 사용자가 밟는 경로 전체로 확인한다.

    "이 dose에서 갭이 닫힌다"가 dose test의 답이다. 그 dose는 측정값이 없어
    곡선에서 통째로 빠져 있었다 — 표와 CSV에는 있지만 사용자가 dose를 고르는
    곳은 곡선이다.
    """
    write_sample(tmp_path, gap_nm=90.0, dose=300)
    write_closed_sample(tmp_path, dose=400)

    window = MainWindow()
    window.open_folder(tmp_path)
    for index in (0, 1):
        window.select_image(index)
        window.measure_current()

    closed_result = window.session.records[1].roi_results[0]
    assert closed_result.mean_nm is None, "픽스처가 실제로 닫혀야 뜻이 있다"
    assert closed_result.n_short > closed_result.n_uncertain

    assert window.dose_plot.point_count() == 1          # 300 uC만 측정됨
    assert window.dose_plot.closed_dose_marks() == [(400.0, 0.0)]

    out = tmp_path / "report.txt"
    window.export_report(out)
    block = out.read_text(encoding="utf-8").split("dose - 갭 관계")[1]
    assert "400.0 uC" in block and "전 구간 short" in block


def test_a_session_where_every_dose_closes_suspects_the_setup(qapp, tmp_path):
    """모든 dose가 전 구간 short면 상태 표시줄이 설정을 의심하라고 말한다.

    한 장만 보면 "갭 측정 불가"는 이 dose에서 갭이 닫혔다는 뜻으로 읽힌다.
    그런데 시리즈 전체가 그렇다면 dose test로서 말이 안 되고, ROI가 패턴을
    벗어났거나 스케일·문턱이 틀렸을 쪽이 훨씬 그럴듯하다. 사용자가 측정 직후에
    보는 줄은 이것 하나뿐이다.
    """
    write_closed_sample(tmp_path, dose=300)
    write_closed_sample(tmp_path, dose=400)

    window = MainWindow()
    window.open_folder(tmp_path)
    for index in (0, 1):
        window.select_image(index)
        window.measure_current()

    assert "ROI" in window.status_text()
    assert "확인" in window.status_text()


def test_one_measured_dose_keeps_the_setup_warning_off_the_status_bar(qapp,
                                                                      tmp_path):
    """진짜로 닫힌 dose 하나에까지 설정 경고가 붙으면 그 문장이 무시된다."""
    write_sample(tmp_path, gap_nm=90.0, dose=300)
    write_closed_sample(tmp_path, dose=400)

    window = MainWindow()
    window.open_folder(tmp_path)
    for index in (0, 1):
        window.select_image(index)
        window.measure_current()

    assert "갭 측정 불가" in window.status_text()
    assert "스케일·문턱" not in window.status_text()


def test_opening_a_png_folder_says_what_to_do_instead_of_a_tiff_error(qapp,
                                                                      tmp_path):
    """PNG 크롭 폴더에서 상태 표시줄과 파일 목록이 다음 행동을 말해 준다.

    예전에는 `오류: 메타데이터를 읽지 못했다: not a TIFF file: header=b'\\x89PNG'`가
    떴다. 파일 목록 칸은 20자로 잘리므로 거기 남는 것이 `not `쯤이었다. 파일은
    멀쩡하고 올바른 다음 행동은 스케일 캘리브레이션인데, 화면 어디에도 그 말이
    없었다.
    """
    Image.fromarray(np.full((256, 256), 200, dtype=np.uint8)).save(
        tmp_path / "crop_300uC.png")

    window = MainWindow()
    window.open_folder(tmp_path)
    window.select_image(0)

    assert "스케일" in window.status_text()
    assert "TIFF" not in window.file_panel.status_text(0), \
        "잘린 목록 칸에 영문 라이브러리 메시지가 남았다"
    assert "스케일" in window.file_panel.status_text(0)


def test_the_full_png_crop_workflow_leaves_no_error_in_the_report(
        qapp, tmp_path, monkeypatch):
    """README가 안내하는 PNG 크롭 경로 전체를 밟는다.

    폴더 열기 -> 캘리브레이션 -> 측정 -> 리포트. 오류 메시지가 하라고 시킨
    행동(스케일 캘리브레이션)을 그대로 했는데 리포트가 그 이미지를 여전히
    `오류:`로 찍으면, 실험 노트가 자기가 낸 60 nm를 의심하게 만든다.
    """
    img = synth_gap_image(width=512, height=512, gap_nm=60.0, nm_per_px=3.0,
                          angle_deg=2.0, edge_sigma_px=1.2, noise_sigma=3.0,
                          seed=300)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
        tmp_path / "crop_300uC.png")

    window = MainWindow()
    window.open_folder(tmp_path)
    before = tmp_path / "before.txt"
    window.export_report(before)
    assert "오류:" in before.read_text(encoding="utf-8"), \
        "캘리브레이션 전까지는 오류가 맞다 — 그래야 뒤의 단언에 뜻이 있다"

    class AcceptingDialog(CalibrationDialog):
        """실제 다이얼로그. 모달로 멈추는 exec만 대신한다."""

        def exec(self):
            self.set_length(0.6, "µm")
            self.set_manual_pixels(200.0)   # 600 nm / 200 px = 3.0 nm/px
            return CalibrationDialog.Accepted

    monkeypatch.setattr("ebl_gap_gui.app.CalibrationDialog", AcceptingDialog)
    window.calibrate_current()
    window.measure_current()

    assert window.session.records[0].roi_results[0].mean_nm == pytest.approx(
        60.0, abs=1.0)

    out = tmp_path / "report.txt"
    window.export_report(out)
    text = out.read_text(encoding="utf-8")

    assert "오류:" not in text, text
    assert window.file_panel.status_text(0).startswith("측정")


def test_exporting_an_overlay_with_nothing_open_says_so(qapp, tmp_path):
    """선택된 이미지가 없을 때 조용히 아무 일도 안 하면 안 된다.

    툴바의 "오버레이 PNG"는 파일 대화상자까지 띄우고 저장을 누르게 해 놓고
    아무것도 쓰지 않았다. 사용자는 저장된 줄 안다.
    """
    window = MainWindow()
    window.export_overlay(tmp_path / "overlay.png")
    assert (tmp_path / "overlay.png").exists() is False
    assert window.status_text() != ""


def test_locking_the_angle_sends_it_to_the_engine(qapp, folder):
    """체크박스와 스핀박스의 값이 measure_roi까지 실제로 닿는지 확인한다.

    위젯 값만 확인하는 테스트는 이 프로젝트에서 죽은 경로를 아홉 번 초록으로
    덮었다. 여기서는 엔진이 돌려준 angle_deg로 확인한다 — 그 값은 measure_roi를
    통과하지 않고는 나올 수 없다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    auto_deg = window.session.records[0].roi_results[0].angle_deg
    assert auto_deg == pytest.approx(2.0, abs=0.5)  # 합성 이미지의 실제 기울기

    window.angle_lock_check.setChecked(True)
    window.angle_deg_spin.setValue(-12.0)
    window.measure_current()

    assert window.session.records[0].roi_results[0].angle_deg == pytest.approx(-12.0)


def test_unchecking_the_lock_returns_to_automatic_estimation(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.angle_lock_check.setChecked(True)
    window.angle_deg_spin.setValue(-12.0)
    window.measure_current()
    assert window.session.records[0].roi_results[0].angle_deg == pytest.approx(-12.0)

    window.angle_lock_check.setChecked(False)
    window.measure_current()
    assert window.session.records[0].roi_results[0].angle_deg == pytest.approx(
        2.0, abs=0.5)


def write_dark_bottom_sample(path, gap_nm, dose):
    """아래쪽이 어두운 멀쩡한 시료. 데이터바는 없고 측정도 정상으로 된다."""
    img = synth_gap_image(width=512, height=512, gap_nm=gap_nm, nm_per_px=3.0,
                          angle_deg=2.0, edge_sigma_px=1.2, noise_sigma=3.0,
                          seed=int(dose))
    data = np.clip(img, 0, 255).astype(np.uint8)
    data[460:, :] = 10
    return _write_tif(path, dose, data)


def test_selecting_an_image_labels_its_note_as_guidance(qapp, tmp_path):
    """잴 수 있는 이미지의 안내가 상태 표시줄에서 오류로 읽히면 안 된다.

    화면과 파일이 같은 문구를 써야 한다 — 상태 표시줄에서만 채널이 뭉개지면
    사용자는 리포트를 열었을 때 다른 이야기를 읽는다.
    """
    write_sample(tmp_path, gap_nm=90.0, dose=300)
    write_dark_bottom_sample(tmp_path, gap_nm=60.0, dose=400)
    window = MainWindow()
    window.open_folder(tmp_path)

    window.file_panel.select(1)

    assert "참고:" in window.status_text()
    assert "오류:" not in window.status_text()
    assert "데이터바" in window.status_text()


def test_opening_a_folder_keeps_the_first_images_notice_on_screen(qapp,
                                                                  tmp_path):
    """0번에 붙은 안내가 "N장 불러옴"에 덮이면 그 장에서만 안내가 사라진다.

    첫 장은 사용자가 아무것도 누르지 않아도 선택되는 유일한 장이다. 다른 장은
    클릭하면 안내가 뜨는데 0번만 안 뜨면, 폴더의 첫 이미지에 붙은 데이터바
    경고나 스케일 안내를 아무도 못 본다.
    """
    write_dark_bottom_sample(tmp_path, gap_nm=60.0, dose=300)
    write_sample(tmp_path, gap_nm=90.0, dose=400)

    window = MainWindow()
    window.open_folder(tmp_path)

    assert window._current == 0
    assert "2장 불러옴" in window.status_text()
    assert "참고:" in window.status_text()
    assert "데이터바" in window.status_text()


def test_switching_images_releases_the_angle_lock(qapp, folder):
    """각도는 그 이미지의 성질이다. 고정이 전역으로 남으면 다음 이미지를 그
    각도로 재고, 틀린 갭 폭이 조용히 나온다. 리뷰어 실측: 이미지 0에서 -12도로
    고정한 뒤 이미지 1을 재면 참값 60.0 nm가 61.856 nm(+3.09%)로 나왔다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    window.angle_lock_check.setChecked(True)
    window.angle_deg_spin.setValue(-12.0)

    window.file_panel.select(1)

    assert window.angle_lock_check.isChecked() is False
    window.measure_current()
    assert window.session.records[1].roi_results[0].angle_deg == pytest.approx(
        2.0, abs=0.5)


def test_switching_images_puts_that_images_own_angle_in_the_spin_box(qapp,
                                                                     folder):
    """고정을 풀기만 하고 스핀박스에 남긴 값은 다음 고정의 출발점이 된다.

    떠나는 이미지의 -12도가 남아 있으면 사용자가 새 이미지에서 고정을 켜는
    순간 그 값이 그대로 쓰인다. 되돌릴 값은 이 이미지의 각도다 — 잰 적이
    있으면 그때 쓴 각도, 아직 없으면 0도.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    measured_deg = window.session.records[0].roi_results[0].angle_deg
    window.angle_lock_check.setChecked(True)
    window.angle_deg_spin.setValue(-12.0)

    window.file_panel.select(1)  # 아직 측정하지 않은 이미지
    assert window.angle_deg_spin.value() == pytest.approx(0.0)

    window.file_panel.select(0)  # 이미 잰 이미지
    assert window.angle_deg_spin.value() == pytest.approx(measured_deg,
                                                          abs=0.01)


def test_releasing_the_angle_lock_is_announced_once(qapp, folder):
    """말없이 풀면 사용자는 여전히 고정된 줄 알고 결과를 읽는다."""
    window = MainWindow()
    window.open_folder(folder)
    window.angle_lock_check.setChecked(True)
    window.angle_deg_spin.setValue(-12.0)

    window.file_panel.select(1)
    assert "고정" in window.status_text()
    assert "해제" in window.status_text()
    # 어느 이미지를 보고 있는지도 같은 줄에 남아야 한다.
    assert window.session.records[1].path.name in window.status_text()


def test_switching_images_without_a_lock_says_nothing_about_it(qapp, folder):
    """고정한 적이 없는데 해제 안내가 뜨면 안내가 잡음이 된다."""
    window = MainWindow()
    window.open_folder(folder)
    window.file_panel.select(1)
    assert "해제" not in window.status_text()


def test_measuring_puts_the_estimated_angle_into_the_spin_box(qapp, folder):
    """사용자가 추정된 각도를 보고 고정할지 정할 수 있어야 한다."""
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    result = window.session.records[0].roi_results[0]
    assert window.angle_deg_spin.value() == pytest.approx(result.angle_deg,
                                                          abs=0.01)


def test_arming_the_angle_spin_box_leaves_its_signal_working(qapp, folder):
    """무장하면서 막은 신호를 되돌리지 않으면 이후 조작이 조용히 무시된다."""
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    seen = []
    window.angle_deg_spin.valueChanged.connect(seen.append)
    window.angle_deg_spin.setValue(7.5)
    assert seen == [pytest.approx(7.5)]


def test_arming_the_angle_spin_box_does_not_overwrite_the_result_message(qapp,
                                                                         folder):
    """측정 결과 한 줄이 각도 안내로 덮이면 사용자는 측정값을 못 본다."""
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    assert "nm" in window.status_text()
    assert "고정" not in window.status_text()


def test_locking_the_angle_says_that_a_new_measurement_is_needed(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    window.angle_lock_check.setChecked(True)
    assert "고정" in window.status_text()
    assert "다시 측정" in window.status_text()


def test_the_threshold_control_reaches_the_engine(qapp, folder):
    """문턱 비율 조작이 measure_roi까지 닿는지 측정값의 방향으로 확인한다.

    어두운 갭이므로 문턱을 낮추면 갭이 좁게 측정된다. 이 방향이 뒤집히면
    README가 적어 놓은 설명이 거짓말이 된다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.threshold_spin.setValue(0.30)
    window.measure_current()
    narrow_nm = window.session.records[0].roi_results[0].mean_nm

    window.threshold_spin.setValue(0.70)
    window.measure_current()
    wide_nm = window.session.records[0].roi_results[0].mean_nm

    assert narrow_nm < 90.0 < wide_nm  # 합성 이미지의 참값은 90 nm


def test_the_along_average_control_reaches_the_engine(qapp, folder):
    """갭 축 이동평균을 올리면 행 사이 산포가 실제로 줄어야 한다."""
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    raw_std_nm = window.session.records[0].roi_results[0].std_nm

    window.along_average_spin.setValue(5)
    window.measure_current()
    smoothed_std_nm = window.session.records[0].roi_results[0].std_nm

    assert smoothed_std_nm < 0.9 * raw_std_nm


def test_changing_the_threshold_blanks_the_profile_plot_instead_of_mixing_it(
        qapp, folder):
    """문턱을 바꾸면 미니 플롯은 따라 움직이는 것이 아니라 비워진다.

    따라 움직이게 하면 0.30 문턱선을 0.50으로 잡은 에지 위에 겹쳐 그리게 된다 —
    `ProfilePlot.show_line`의 주석이 금지한 바로 그 그림이고, 사용자는 그것을
    "새 문턱으로 잰 결과"로 읽는다. 섞인 그림보다 빈 그림이 낫다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    assert len(window.profile_plot.threshold_lines()) == 2

    window.threshold_spin.setValue(0.30)

    assert window.profile_plot.threshold_lines() == []
    assert window.profile_plot.has_curve() is False


def test_changing_a_measure_setting_is_treated_like_moving_the_roi(qapp, folder):
    """설정을 바꾸면 그 설정으로 재지 않은 그림은 전부 사라져야 한다.

    Task 23이 (ROI, 결과) 쌍 불변식을 세웠는데 설정은 (params, 결과)라는 쌍을
    새로 만들고 아무것도 유지하지 않았다. 문턱을 0.50에서 0.30으로 돌린 뒤
    재측정하지 않아도 오버레이와 미니 플롯이 0.50으로 잰 에지를 그대로 들고
    있었다. ROI를 옮겼을 때와 똑같이 버린다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    assert window.image_view.has_overlay()

    window.along_average_spin.setValue(5)

    assert window.image_view.has_overlay() is False
    assert window.profile_plot.has_curve() is False
    assert window.line_selector.isEnabled() is False
    assert window._profiles_by_index == {}
    assert window._measured_rois == {}


def test_changing_a_measure_setting_says_why_the_screen_went_blank(qapp, folder):
    """화면을 비웠으면 왜 비웠는지 말해야 한다.

    이 줄을 지워도 358개가 통과했다. 지우면 화면은 텅 빈 채로 상태 표시줄만
    방금 성공한 측정값을 계속 말하게 되고, 사용자는 진단이 사라진 이유를
    어디에서도 읽을 수 없다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    assert "nm" in window.status_text()

    window.threshold_spin.setValue(0.30)

    assert "설정" in window.status_text()
    assert "다시 측정" in window.status_text()


def test_exporting_an_overlay_after_changing_a_setting_is_refused(qapp, folder,
                                                                  tmp_path):
    """실험 노트에 "문턱 0.30"이라 적으면서 0.50으로 잰 그림을 붙이면 안 된다.

    화면은 비워지지만 PNG는 파일로 남아 화면보다 오래 간다. ROI를 옮겼을 때와
    같은 이유로 같은 거부를 받는다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.measure_current()
    window.threshold_spin.setValue(0.30)

    out = tmp_path / "overlay.png"
    window.export_overlay(str(out))

    assert out.exists() is False
    assert "다시 측정" in window.status_text()


def test_measuring_an_roi_that_reaches_into_the_databar_is_refused(qapp,
                                                                   tmp_path):
    """거부는 엔진이 한다. GUI는 그 사유를 그대로 사용자에게 보여준다.

    ROI가 데이터바를 걸치면 균일한 띠가 라인마다 short로 판정돼 날조된 이상
    비율이 나온다. 측정 결과가 남지 않는 것까지 확인한다 — 결과가 남으면
    dose-gap 곡선과 CSV에 그 가짜 숫자가 그대로 실린다.
    """
    write_sample_with_databar(tmp_path, gap_nm=90.0, dose=300)
    window = MainWindow()
    window.open_folder(tmp_path)
    window.image_view.set_roi(Roi(100, 300, 400, 500))  # 452행부터가 데이터바다

    window.measure_current()

    assert window.session.records[0].roi_results == []
    assert "데이터바" in window.status_text()
    assert "452" in window.status_text()


def test_measuring_above_the_databar_still_works(qapp, tmp_path):
    write_sample_with_databar(tmp_path, gap_nm=90.0, dose=300)
    window = MainWindow()
    window.open_folder(tmp_path)
    window.image_view.set_roi(Roi(100, 100, 400, 400))

    window.measure_current()

    result = window.session.records[0].roi_results[0]
    assert result.mean_nm == pytest.approx(90.0, abs=3.0)


def test_the_angle_spin_box_spans_every_angle_the_engine_can_return(qapp):
    """estimate_angle_deg는 arctan 결과라 값의 범위가 (-90, 90)이다.

    스핀박스 범위가 그보다 좁으면 무장할 때 값이 조용히 잘리고, 화면에 보이는
    각도와 측정에 실제로 쓴 각도가 갈라진다. 각도 붕괴를 사용자에게 보여주는
    것이 이 조작의 존재 이유인데, 하필 그 붕괴한 값이 잘린다.
    """
    window = MainWindow()
    for angle_deg in (-70.0, 70.0):
        window.angle_deg_spin.setValue(angle_deg)
        assert window.angle_deg_spin.value() == pytest.approx(angle_deg)


def test_the_angle_spin_box_reaches_the_horizontal_base(qapp):
    """가로 갭의 기준은 90도이고 거기에 기울기가 붙으면 90을 넘는다.

    범위가 (-90, 90)이면 91.5도가 조용히 90도로 잘려, 화면의 각도와 측정에
    실제로 쓴 각도가 갈라진다.
    """
    window = MainWindow()
    for angle_deg in (91.5, -91.5, 180.0, -180.0):
        window.angle_deg_spin.setValue(angle_deg)
        assert window.angle_deg_spin.value() == pytest.approx(angle_deg)


def test_an_unmeasured_image_inherits_the_last_used_roi(qapp, folder):
    """폴더의 이미지들은 배율도 패턴 위치도 같다. 한 번 맞춘 ROI를 장마다
    다시 끌게 하면 10장짜리 dose 시리즈에서 같은 동작을 열 번 한다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.image_view.set_roi(Roi(100, 400, 800, 480))
    shaped = window.image_view.current_roi()

    window.select_image(1)                      # 아직 측정 안 한 이미지

    assert window.image_view.current_roi() == shaped


def test_a_measured_images_own_roi_still_beats_the_inherited_one(qapp, folder):
    """우선순위: 측정한 이미지면 그 ROI, 아니면 마지막 ROI, 그것도 없으면 기본값.

    Task 22가 세운 복원이 물려받기에 밀리면, 측정 결과와 화면의 상자가 갈라진
    채로 오버레이가 그려진다.
    """
    window = MainWindow()
    window.open_folder(folder)
    window.image_view.set_roi(Roi(60, 70, 260, 170))
    window.measure_current()
    measured = window.image_view.current_roi()

    window.select_image(1)
    window.image_view.set_roi(Roi(120, 130, 320, 230))
    later = window.image_view.current_roi()
    window.select_image(0)

    assert window.image_view.current_roi() == measured
    assert window.image_view.current_roi() != later


def test_an_inherited_roi_keeps_its_shape_on_a_smaller_image(qapp, tmp_path):
    """작은 이미지에 안 들어가면 줄이되 가로:세로 비율은 지킨다.

    비율을 버리고 잘라 넣으면 측정 방향 폭이 조용히 넓어지거나 좁아져서,
    사용자가 맞춰 놓은 "갭 두께의 네댓 배"라는 모양이 사라진다. 줄였으면
    상태 표시줄로 알린다 — 말없이 바뀐 상자는 사용자가 알아채지 못한다.
    """
    write_sample(tmp_path, gap_nm=90.0, dose=300)
    Image.fromarray(np.full((256, 256), 200, dtype=np.uint8)).save(
        tmp_path / "z_small_400uC.png")

    window = MainWindow()
    window.open_folder(tmp_path)
    window.image_view.set_roi(Roi(50, 100, 450, 180))
    inherited = window.image_view.current_roi()

    window.select_image(1)

    fitted = window.image_view.current_roi()
    assert fitted.x1 < 256 and fitted.y1 < 256
    assert fitted.width / fitted.height == pytest.approx(
        inherited.width / inherited.height, rel=0.05)
    assert "ROI" in window.status_text()


def test_the_default_roi_is_long_along_a_vertical_gap(qapp, folder):
    """첫 이미지에는 물려받을 ROI가 없다. 그때라도 갭 모양에 맞춘 상자여야 한다.

    정사각형은 갭 축 방향으로는 짧아 라인 수가 모자라고, 측정 방향으로는
    쓸데없이 넓어 옆 패턴까지 끌어들인다. 사용자가 매 장 손으로 늘리던 것이
    바로 이 모양이다.
    """
    window = MainWindow()
    window.open_folder(folder)

    roi = window.image_view.current_roi()
    assert roi.height > roi.width * 2, f"세로 갭인데 상자가 {roi}"


def test_a_horizontal_gap_gets_a_wide_default_roi(qapp, tmp_path):
    """갭이 가로면 가로로 긴 상자다. 방향을 잘못 고르면 값까지 틀린다."""
    write_horizontal_sample(tmp_path, gap_nm=90.0, dose=300)
    window = MainWindow()
    window.open_folder(tmp_path)

    roi = window.image_view.current_roi()
    assert roi.width > roi.height * 2, f"가로 갭인데 상자가 {roi}"

    window.measure_current()
    assert window.session.records[0].roi_results[0].mean_nm == pytest.approx(
        90.0, abs=3.0)


def test_the_default_roi_stays_clear_of_the_databar(qapp, tmp_path):
    """기본 ROI가 데이터바를 침범하면 폴더를 연 직후 측정이 거부된다.

    첫 화면에서 "다시 잡으세요"만 나오는 프로그램은 쓸 수 없다. 긴 쪽을 잡을
    때 데이터바 위쪽만 이미지로 친다.

    상자가 데이터바를 피했는지만 보면 부족하다. 데이터바를 포함한 채 방향을
    판별하면 균일한 어두운 띠가 세로 방향 대비를 키워 "가로 갭"으로 뒤집히고,
    그때 나오는 납작한 상자는 우연히 데이터바 위에 앉는다 — 거부는 피하지만
    세로 갭을 가로로 재는 상자다. 그래서 모양까지 함께 못박는다.
    """
    write_sample_with_databar(tmp_path, gap_nm=90.0, dose=300,
                              databar_rows=220)
    window = MainWindow()
    window.open_folder(tmp_path)

    roi = window.image_view.current_roi()
    assert roi.y1 < 292, f"데이터바는 292행부터다: {roi}"
    assert roi.height > roi.width * 2, f"세로 갭인데 상자가 {roi}"

    window.measure_current()

    assert "데이터바" not in window.status_text(), window.status_text()
    assert window.session.records[0].roi_results


# ------------------------------------------------ 파일명에서 dose를 읽어 채운다

def write_named_sample(path, name, gap_nm, seed):
    """이름을 직접 정하는 합성 SEM TIFF. uC도 FEI dose도 이름에 없다."""
    img = synth_gap_image(width=512, height=512, gap_nm=gap_nm, nm_per_px=3.0,
                          angle_deg=2.0, edge_sigma_px=1.2, noise_sigma=3.0,
                          seed=seed)
    out = path / name
    tifffile.imwrite(out, np.clip(img, 0, 255).astype(np.uint8),
                     extratags=[(34682, 's', 0, _fei_ini(), True)])
    return out


@pytest.fixture()
def named_folder(tmp_path):
    """사용자의 실제 이름 규칙: ARP_70_C_<dose>_<반복>.tif."""
    for dose, gap_nm in ((140, 90.0), (160, 75.0)):
        for rep in (1, 2):
            write_named_sample(tmp_path, f"ARP_70_C_{dose}_{rep:03d}.tif",
                               gap_nm, seed=dose * 10 + rep)
    return tmp_path


def test_open_folder_reads_the_dose_from_the_name_fields(qapp, named_folder):
    """사용자 요청: "도즈를 하나하나 입력하지 않도록". 자동으로 채운다."""
    window = MainWindow()
    window.open_folder(named_folder)
    assert sorted(r.dose for r in window.session.records) == [140.0, 140.0,
                                                              160.0, 160.0]
    assert window.file_panel.dose_text(0) == "140"


def test_the_banner_shows_which_field_was_read(qapp, named_folder):
    """자동으로 넣되 무엇을 넣었는지 보인다. 틀린 dose는 곡선을 통째로 뒤집는다."""
    window = MainWindow()
    window.open_folder(named_folder)
    assert window.naming_banner.is_shown()
    assert "4번째 칸" in window.naming_banner.text()
    assert "140" in window.naming_banner.text()


def test_the_uc_rule_wins_over_the_name_field_guess(qapp, tmp_path):
    """기존 dose가 있으면 파일명 칸 추정은 손대지 않는다.

    이 이름들에서 변하는 숫자 칸은 반복 번호(001, 002)뿐이라, 추정이 이기면
    dose가 300/400에서 1/2로 덮인다 — x축이 통째로 거짓이 된다.
    """
    for dose in (300, 400):
        for rep in (1, 2):
            write_named_sample(tmp_path, f"ARP_70_C_{dose}uC_{rep:03d}.tif",
                               90.0, seed=dose + rep)
    window = MainWindow()
    window.open_folder(tmp_path)
    assert sorted(r.dose for r in window.session.records) == [300.0, 300.0,
                                                              400.0, 400.0]
    # 아무것도 채우지 않았으면 배너도 없다. 채우지 않은 것을 채웠다고 말하면 안 된다.
    assert not window.naming_banner.is_shown()


def test_choosing_another_field_refills_every_dose(qapp, named_folder):
    window = MainWindow()
    window.open_folder(named_folder)
    window.naming_banner.choose_field(1)        # 상수 칸 '70'
    assert {r.dose for r in window.session.records} == {70.0}
    assert window.file_panel.dose_text(0) == "70"


def test_turning_the_inference_off_empties_the_doses_it_filled(qapp,
                                                               named_folder):
    window = MainWindow()
    window.open_folder(named_folder)
    window.naming_banner.disable_button.click()
    assert all(r.dose is None for r in window.session.records)
    assert window.file_panel.dose_text(0) == ""
    assert not window.naming_banner.is_shown()


def test_a_hand_edited_dose_survives_turning_the_inference_off(qapp,
                                                               named_folder):
    """사용자가 직접 고친 값은 추정의 것이 아니다. 끄기가 지우면 안 된다."""
    window = MainWindow()
    window.open_folder(named_folder)
    window.file_panel._table.item(0, 1).setText("999")
    window.naming_banner.disable_button.click()
    assert window.session.records[0].dose == pytest.approx(999.0)
    assert all(r.dose is None for r in window.session.records[1:])


def test_names_without_a_numeric_field_show_no_banner(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    assert not window.naming_banner.is_shown()


def test_the_result_panel_notes_the_target_gap_from_the_name(qapp,
                                                             named_folder):
    window = MainWindow()
    window.open_folder(named_folder)
    assert "목표 갭 70" in window.result_panel.text()


def test_the_target_gap_from_the_name_never_touches_the_measurement(
        qapp, named_folder):
    """파일명은 의도이지 계측값이 아니다.

    이름은 70 nm를 말하고 실제 갭은 90 nm다. 측정값도, 경고도, 판정도 이름을
    쳐다보면 안 된다. 참고 줄로만 남는다.
    """
    window = MainWindow()
    window.open_folder(named_folder)
    window.select_image(0)
    window.measure_current()
    result = window.session.records[0].roi_results[0]
    assert result.mean_nm == pytest.approx(90.0, abs=3.0)
    assert all("목표" not in warning for warning in result.warnings)
    assert "목표 갭 70" in window.result_panel.text()


def test_opening_another_folder_drops_the_previous_name_reading(qapp,
                                                                named_folder,
                                                                folder):
    """폴더가 바뀌면 다른 시료다. 앞 폴더의 추정이 남으면 남의 목표 갭을 읽는다."""
    window = MainWindow()
    window.open_folder(named_folder)
    window.open_folder(folder)
    assert not window.naming_banner.is_shown()
    assert "목표 갭" not in window.result_panel.text()


def test_repeat_shots_of_one_dose_land_on_one_curve_point(qapp, named_folder):
    """폴더를 열고 네 장을 다 재면 곡선에는 dose 두 개뿐이다.

    이름에서 dose를 읽는 것(Step 1~2)과 같은 dose를 한 점으로 묶는 것(Step 3)이
    한 경로에서 만나는 자리다. 둘 중 하나만 되면 점이 네 개거나 없다.
    """
    window = MainWindow()
    window.open_folder(named_folder)
    for index in range(4):
        window.select_image(index)
        window.measure_current()
    assert window.dose_plot.point_count() == 2
    assert "2장 평균" in window.dose_plot.hover_tooltip(0)
