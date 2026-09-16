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
from ebl_gap.types import ImageRecord, Roi, RoiResult, ScaleInfo
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


def test_summary_csv_reports_the_scale_the_measurement_actually_used(tmp_path):
    """측정 뒤 캘리브레이션을 다시 해도 CSV는 그 숫자를 만든 스케일을 쓴다.

    `record.scale`은 "이 이미지의 현재 스케일"이고 `result.scale`은 "이 mean_nm을
    계산할 때 쓴 스케일"이다. 측정 -> 재캘리브레이션 순서로 가면 둘이 실제로
    갈라진다. 그때 record 쪽을 쓰면 CSV 한 행 안에서 mean_nm과 nm_per_px가 서로
    다른 스케일을 가리켜, 나중에 그 행으로 계산을 재현할 수 없게 된다. 출처를
    값과 함께 들고 다니는 것이 `ScaleInfo`의 존재 이유다.
    """
    session, _, result = measured_session(tmp_path)
    recalibrated = ScaleInfo(nm_per_px=5.0, source="manual")
    session.records[0].scale = recalibrated   # 측정 뒤 사용자가 다시 잡았다
    assert result.scale != recalibrated       # 둘이 갈라져야 검사에 뜻이 있다

    out = tmp_path / "summary.csv"
    write_summary_csv(out, session)

    row = list(csv.DictReader(out.open(encoding="utf-8-sig")))[0]
    assert float(row["nm_per_px"]) == pytest.approx(3.0)
    assert row["scale_source"] == "fei_metadata"


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


def test_overlay_survives_an_roi_that_extends_past_the_image():
    """측정이 되는 ROI는 오버레이도 그려져야 한다.

    extract_profiles는 경계를 벗어난 ROI를 mode="nearest"로 허용한다. 같은 ROI로
    measure_roi가 성공했는데 오버레이만 IndexError로 터지면, CSV에는 값이 남고
    그림만 안 나오는 상태가 된다.
    """
    img = synth_gap_image(width=100, height=100, gap_nm=20.0, nm_per_px=1.0)
    roi = Roi(50, 50, 120, 120)
    result = measure_roi(img, roi, SCALE)
    overlay = render_overlay(img, roi, result)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8


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
    # "short"는 모든 ROI 줄에 라벨로 항상 찍히므로 그것만 보면 구현이 틀려도
    # 통과한다. 실제로 short 라인이 잡혔는지와, stats.py가 만든 경고 문구가
    # 리포트에 올라왔는지를 본다.
    assert result.n_short > 0
    assert "short 발생 구간 있음" in text


def closed_record(path, dose, n_short=300, n_uncertain=0):
    """갭이 전 구간에서 닫힌 이미지 레코드."""
    return ImageRecord(path=path, scale=SCALE, dose=dose, roi_results=[
        RoiResult(mean_nm=None, std_nm=None, n_valid=0, n_short=n_short,
                  n_uncertain=n_uncertain, n_low_confidence=0, angle_deg=0.0,
                  lines=(), warnings=(), scale=SCALE)])


def test_report_lists_a_closed_dose_in_the_dose_curve_block(tmp_path):
    """갭이 닫힌 dose가 dose-갭 블록에 dose 순서대로 들어간다.

    그 블록이 "어느 dose에서 갭이 닫히는가"를 읽는 자리다. 닫힌 dose가 빠지면
    리포트는 측정된 폭만 나열하고 답은 말하지 않는다.
    """
    session, _, _ = measured_session(tmp_path)          # 320 uC, 측정됨
    session.add(closed_record(tmp_path / "pattern_400uC.tif", 400.0))

    block = format_report(session).split("dose - 갭 관계")[1]

    lines = [ln for ln in block.splitlines() if "uC" in ln]
    assert len(lines) == 2
    assert "320.0 uC" in lines[0] and "nm" in lines[0]
    assert "400.0 uC" in lines[1]
    assert "전 구간 short" in lines[1]
    assert "300" in lines[1]


def test_report_shows_the_dose_block_when_every_dose_is_closed(tmp_path):
    """전 구간이 닫힌 시리즈에서도 블록이 나와야 한다.

    측정된 점이 하나도 없으면 블록을 통째로 빼던 자리다. 그러면 리포트가
    "dose 관계 없음"처럼 읽히는데, 실제로는 모든 dose에서 갭이 닫힌 것이다.
    """
    session = Session()
    session.add(closed_record(tmp_path / "pattern_400uC.tif", 400.0))
    session.add(closed_record(tmp_path / "pattern_500uC.tif", 500.0))

    text = format_report(session)

    assert "dose - 갭 관계" in text
    block = text.split("dose - 갭 관계")[1]
    assert block.count("전 구간 short") == 2


def test_report_does_not_call_an_unmeasurable_roi_a_closed_dose(tmp_path):
    """판정보류뿐인 이미지는 "닫혔다"가 아니라 "못 쟀다"이다."""
    session = Session()
    session.add(closed_record(tmp_path / "pattern_400uC.tif", 400.0,
                              n_short=0, n_uncertain=300))
    assert "전 구간 short" not in format_report(session)
