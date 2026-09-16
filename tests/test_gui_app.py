import csv

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
