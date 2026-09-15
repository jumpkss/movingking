import csv
from pathlib import Path

import numpy as np
import pytest

from ebl_gap.dataset import Session
from ebl_gap.export import (
    format_report,
    render_overlay,
    write_lines_csv,
    write_overlay_png,
    write_summary_csv,
)
from ebl_gap.measure import measure_roi
from ebl_gap.types import ImageRecord, Roi, ScaleInfo
from tests.synth import synth_gap_image

ROI = Roi(106, 106, 405, 405)
SCALE = ScaleInfo(nm_per_px=3.0, source="fei_metadata")


def measured_session(tmp_path):
    img = synth_gap_image(gap_nm=120.0, nm_per_px=3.0, angle_deg=4.0)
    result = measure_roi(img, ROI, SCALE)
    session = Session()
    session.add(ImageRecord(path=tmp_path / "pattern_320uC.tif", scale=SCALE,
                            dose=320.0, roi_results=[result]))
    return session, img, result


def test_summary_csv_has_one_row_per_roi_with_scale_provenance(tmp_path):
    session, _, result = measured_session(tmp_path)
    out = tmp_path / "summary.csv"
    write_summary_csv(out, session)
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert len(rows) == 1
    row = rows[0]
    assert row["file"] == "pattern_320uC.tif"
    assert float(row["dose_uC"]) == pytest.approx(320.0)
    assert float(row["nm_per_px"]) == pytest.approx(3.0)
    assert row["scale_source"] == "fei_metadata"
    # CSV는 mean_nm을 소수 3자리로 쓴다. 허용오차는 그 반올림 단위(5e-4)여야
    # 한다. 더 조이면 구조적으로 실패하고, CSV 자릿수를 늘려 맞추는 것은
    # 측정이 갖지 않은 정밀도를 보고하는 셈이 된다 — 1~3 nm/px 해상도에서
    # 0.001 nm는 이미 물리적 의미보다 두 자릿수 이상 세밀하다.
    assert float(row["mean_nm"]) == pytest.approx(result.mean_nm, abs=5e-4)
    assert int(row["n_valid"]) == result.n_valid


def test_summary_csv_writes_empty_cells_for_unmeasured_images(tmp_path):
    session = Session()
    session.add(ImageRecord(path=Path("blank.tif"), scale=None, dose=None,
                            roi_results=[], error="FEI 메타데이터 없음"))
    out = tmp_path / "summary.csv"
    write_summary_csv(out, session)
    row = list(csv.DictReader(out.open(encoding="utf-8-sig")))[0]
    assert row["mean_nm"] == ""
    assert row["dose_uC"] == ""
    assert "메타데이터" in row["warnings"]


def test_lines_csv_has_one_row_per_scanline(tmp_path):
    session, _, result = measured_session(tmp_path)
    out = tmp_path / "lines.csv"
    write_lines_csv(out, session.records[0], roi_index=0)
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert len(rows) == ROI.height
    assert rows[0]["status"] in {"valid", "short", "no_edge", "multi_edge",
                                 "sub_resolution", "outlier"}
    assert float(rows[0]["width_nm"]) > 0


def test_overlay_is_rgb_and_marks_the_roi(tmp_path):
    _, img, result = measured_session(tmp_path)
    overlay = render_overlay(img, ROI, result)
    assert overlay.shape == (img.shape[0], img.shape[1], 3)
    assert overlay.dtype == np.uint8
    # ROI 테두리는 노란색으로 그린다.
    assert tuple(overlay[ROI.y0, ROI.x0]) == (255, 255, 0)


def test_overlay_draws_detected_edges_in_green(tmp_path):
    _, img, result = measured_session(tmp_path)
    overlay = render_overlay(img, ROI, result)
    green = np.all(overlay == np.array([0, 255, 0], dtype=np.uint8), axis=-1)
    # 유효 라인마다 좌우 에지 두 점씩 찍힌다.
    assert green.sum() >= result.n_valid


def test_overlay_marks_short_lines_in_red():
    img = np.full((512, 512), 200.0)
    img[:, 250:262] = 40.0
    img[300:400, 250:262] = 200.0  # 100행의 갭을 메워 short를 만든다
    result = measure_roi(img, ROI, SCALE)
    overlay = render_overlay(img, ROI, result)
    red = np.all(overlay == np.array([255, 0, 0], dtype=np.uint8), axis=-1)
    assert result.n_short > 0
    assert red.sum() > 0


def test_overlay_png_is_written_and_readable(tmp_path):
    from PIL import Image

    _, img, result = measured_session(tmp_path)
    out = tmp_path / "overlay.png"
    write_overlay_png(out, img, ROI, result)
    assert out.exists()
    with Image.open(out) as handle:
        assert handle.mode == "RGB"
        assert handle.size == (img.shape[1], img.shape[0])


def test_report_states_the_scale_source_and_the_counts(tmp_path):
    session, _, result = measured_session(tmp_path)
    text = format_report(session)
    assert "fei_metadata" in text
    assert "pattern_320uC.tif" in text
    assert f"{result.n_valid}" in text
    assert "320" in text


def test_report_surfaces_warnings(tmp_path):
    img = np.full((512, 512), 200.0)
    result = measure_roi(img, ROI, SCALE)
    session = Session()
    session.add(ImageRecord(path=Path("closed.tif"), scale=SCALE, dose=500.0,
                            roi_results=[result]))
    text = format_report(session)
    assert "short" in text
