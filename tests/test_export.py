import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

from ebl_gap.dataset import Session
from ebl_gap.export import (
    format_report,
    record_notices,
    render_overlay,
    write_lines_csv,
    write_overlay_png,
    write_summary_csv,
)
from ebl_gap.loader import load_image
from ebl_gap.measure import measure_roi
from ebl_gap.types import ImageRecord, Roi, RoiResult, ScaleInfo
from tests.synth import synth_gap_image
from tests.test_metadata import FEI_INI

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


def test_summary_csv_marks_a_note_as_guidance_not_as_an_error(tmp_path):
    """잴 수 있는 이미지의 안내가 오류로 기록되면 실험 노트가 거짓말을 한다."""
    session = Session()
    session.add(ImageRecord(
        path=Path("dark_bottom.tif"), scale=SCALE, dose=None, roi_results=[],
        notes=["아래쪽 250행부터 데이터바로 보이는 띠가 있습니다"]))
    out = tmp_path / "summary.csv"
    write_summary_csv(out, session)
    row = list(csv.DictReader(out.open(encoding="utf-8-sig")))[0]
    assert "참고:" in row["warnings"]
    assert "오류:" not in row["warnings"]
    assert "데이터바" in row["warnings"]


def test_a_calibrated_image_is_no_longer_reported_as_an_error():
    """오류 메시지가 시킨 행동을 한 뒤에는 오류가 남으면 안 된다.

    스케일이 없다는 것은 캘리브레이션 전까지만 참이다. 측정이 끝난 이미지 옆에
    `오류:`가 남으면 실험 노트가 그 숫자를 의심하게 만든다. `record.error`를
    지우지 않고 표시 시점에 판단한다 — 왜 수동 스케일인지가 기록으로 남는다.
    """
    record = ImageRecord(
        path=Path("crop_300uC.png"), scale=None,
        error="스케일 메타데이터가 없습니다 — 스케일 캘리브레이션으로 직접 지정하세요")
    assert any(n.startswith("오류:") for n in record_notices(record))

    record.scale = ScaleInfo(3.0, "manual")

    assert not any(n.startswith("오류:") for n in record_notices(record))
    assert record.error is not None, "사유 자체는 기록으로 남아 있어야 한다"


def test_report_separates_the_two_channels(tmp_path):
    """`오류:`는 잴 수 없다는 뜻이고 `참고:`는 확인하라는 뜻이다."""
    session = Session()
    session.add(ImageRecord(path=Path("dark_bottom.tif"), scale=SCALE,
                            roi_results=[],
                            notes=["아래쪽 250행부터 띠가 있습니다"]))
    session.add(ImageRecord(path=Path("broken.tif"), scale=None,
                            roi_results=[], error="이미지를 읽지 못했다"))

    text = format_report(session)
    note_block = text.split("- dark_bottom.tif")[1].split("- broken.tif")[0]
    error_block = text.split("- broken.tif")[1]

    assert "참고: 아래쪽 250행부터" in note_block
    assert "오류:" not in note_block
    assert "오류: 이미지를 읽지 못했다" in error_block


def test_summary_csv_carries_the_hfw_mismatch_onto_the_measured_row(tmp_path):
    """픽셀 크기가 틀렸을 수 있다는 유일한 신호가 CSV에서 사라지면 안 된다.

    HFW 불일치는 보고되는 모든 nm를 조용히 편향시킨다. 측정된 행이
    `result.warnings`만 싣던 때는 그 이미지가 CSV에서 가장 믿음직한 행으로
    남았다 — 출처가 `fei_metadata`이고 경고 칸이 비어 있었다.
    """
    ini = (FEI_INI
           .replace("ResolutionX=1024", "ResolutionX=512")
           .replace("ResolutionY=884", "ResolutionY=512")
           .replace("PixelWidth=3.0517578125e-009", "PixelWidth=3.0e-009")
           .replace("PixelHeight=3.0517578125e-009", "PixelHeight=3.0e-009")
           .replace("HorFieldsize=3.125e-006", "HorFieldsize=9.0e-006"))
    img = synth_gap_image(width=512, height=512, gap_nm=120.0, nm_per_px=3.0,
                          angle_deg=4.0)
    path = tmp_path / "pattern_320uC.tif"
    tifffile.imwrite(path, np.clip(img, 0, 255).astype(np.uint8),
                     extratags=[(34682, 's', 0, ini, True)])

    loaded = load_image(path)
    assert any("HFW" in note for note in loaded.record.notes), loaded.record.notes
    loaded.record.roi_results = [measure_roi(loaded.pixels, ROI,
                                             loaded.record.scale)]
    session = Session()
    session.add(loaded.record)

    out = tmp_path / "summary.csv"
    write_summary_csv(out, session)

    row = list(csv.DictReader(out.open(encoding="utf-8-sig")))[0]
    assert row["mean_nm"] != "", "측정된 행이어야 검사에 뜻이 있다"
    assert row["scale_source"] == "fei_metadata"
    assert "참고:" in row["warnings"]
    assert "HFW" in row["warnings"]


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


def test_a_closed_dose_line_says_the_roi_could_also_be_off_the_pattern(tmp_path):
    """평탄 금속 위의 ROI도 전 구간 short를 낸다. 엔진은 둘을 구별할 수 없으므로
    화면이 단정하면 안 된다.

    실측(갭이 x=256인 합성 이미지, ROI를 왼쪽 평탄 금속 위에): 301줄 전부
    short가 나고 no_edge는 한 줄도 나지 않는다. 대비가 없다는 점에서 닫힌 갭과
    빗나간 ROI는 픽셀만으로 같은 그림이다. 사용자가 dose를 고르는 곳이 이
    블록이므로, 여기서 "갭이 닫혔다"로 단정하면 재지 않은 결론을 대신 말하는
    것이 된다.
    """
    session = Session()
    session.add(closed_record(tmp_path / "pattern_400uC.tif", 400.0))

    text = format_report(session)
    block = text.split("dose - 갭 관계")[1]

    assert "전 구간 short (300/300 라인, 유효 0)" in block
    assert "ROI가 패턴을 벗어났습니다" in block
    assert "오버레이" in block


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


def measured_record(path, dose, mean_nm):
    """측정값이 나온 이미지 레코드."""
    return ImageRecord(path=path, scale=SCALE, dose=dose, roi_results=[
        RoiResult(mean_nm=mean_nm, std_nm=1.0, n_valid=300, n_short=0,
                  n_uncertain=0, n_low_confidence=0, angle_deg=0.0,
                  lines=(), warnings=(), scale=SCALE)])


def test_a_closed_dose_between_two_measured_ones_keeps_its_place(tmp_path):
    """dose 블록은 측정된 dose와 닫힌 dose를 dose 순서로 섞어서 낸다.

    닫힌 dose가 측정된 dose "사이에" 끼는 경우가 어느 테스트에도 없었다.
    그래서 정렬을 지우고 측정된 점을 전부 낸 뒤 닫힌 점을 뒤에 붙여도 통과했다.
    이 블록은 dose 곡선을 글로 옮긴 것이고, 사용자는 여기서 "어느 dose부터
    갭이 닫히기 시작하는가"를 읽는다 — 순서가 곧 내용이다.
    """
    session = Session()
    session.add(measured_record(tmp_path / "pattern_300uC.tif", 300.0, 90.0))
    session.add(closed_record(tmp_path / "pattern_400uC.tif", 400.0))
    session.add(measured_record(tmp_path / "pattern_500uC.tif", 500.0, 40.0))

    block = format_report(session).split("dose - 갭 관계")[1]
    doses = [ln.split("uC")[0].strip() for ln in block.splitlines() if "uC :" in ln]

    assert doses == ["300.0", "400.0", "500.0"]


def test_report_does_not_call_an_unmeasurable_roi_a_closed_dose(tmp_path):
    """판정보류뿐인 이미지는 "닫혔다"가 아니라 "못 쟀다"이다."""
    session = Session()
    session.add(closed_record(tmp_path / "pattern_400uC.tif", 400.0,
                              n_short=0, n_uncertain=300))
    assert "전 구간 short" not in format_report(session)


def test_the_report_head_carries_the_session_warning(tmp_path):
    """모든 dose가 전 구간 short인 세션은 리포트 머리에서 그 사실을 말한다.

    dose 블록의 줄마다 붙는 확인 요청과 달리 이것은 세션 전체의 진단이다.
    리포트를 실험 노트에 붙이는 사람이 가장 먼저 읽는 자리에 있어야 한다.
    """
    session = Session()
    session.add(closed_record(tmp_path / "pattern_400uC.tif", 400.0))
    session.add(closed_record(tmp_path / "pattern_500uC.tif", 500.0))

    head = format_report(session).split("- pattern_400uC.tif")[0]

    assert "[세션 경고]" in head
    assert "ROI" in head
