import numpy as np
import pytest

pytest.importorskip("PySide6")

from ebl_gap_gui.calibration import UNIT_FACTORS, CalibrationDialog  # noqa: E402


def databar_image(bar_length=100):
    img = np.full((943, 1024), 120.0)
    img[884:, :] = 10.0
    img[918:923, 60:60 + bar_length] = 250.0
    return img


def test_unit_factors_convert_to_nanometres():
    assert UNIT_FACTORS["nm"] == pytest.approx(1.0)
    assert UNIT_FACTORS["µm"] == pytest.approx(1000.0)


def test_dialog_detects_the_bar_length_on_open(qapp):
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    assert dialog.detected_length_px() == pytest.approx(100, abs=2)


def test_dialog_reports_no_detection_on_a_plain_image(qapp):
    dialog = CalibrationDialog(np.full((200, 200), 100.0))
    assert dialog.detected_length_px() is None


def test_scale_from_detected_bar_and_entered_length(qapp):
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    dialog.set_length(1.0, "µm")
    dialog.confirm_detection(True)
    scale = dialog.scale_info()
    assert scale.nm_per_px == pytest.approx(10.0, rel=0.05)
    assert scale.source == "scalebar_auto"


def test_nanometre_unit_is_used_as_entered(qapp):
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    dialog.set_length(500.0, "nm")
    dialog.confirm_detection(True)
    assert dialog.scale_info().nm_per_px == pytest.approx(5.0, rel=0.05)


def test_detected_bar_is_not_used_until_the_user_confirms_it(qapp):
    """자동 검출은 데이터바의 밝은 텍스트를 막대로 오인할 수 있다.

    스펙이 OCR을 거부한 이유와 같은 실패 방식이므로, 확인 없이는 쓰지 않는다.
    """
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    dialog.set_length(1.0, "µm")
    assert dialog.scale_info() is None
    dialog.confirm_detection(True)
    assert dialog.scale_info().source == "scalebar_auto"


def test_summary_tells_the_user_where_the_bar_was_found(qapp):
    """길이만 보여주면 잘못 잡았는지 알 수 없다. 위치를 보여줘야 확인이 가능하다."""
    text = CalibrationDialog(databar_image(), databar_top=884).summary_text()
    assert "행" in text
    assert "60" in text  # 막대 시작 x 좌표


def test_manual_pixel_distance_needs_no_confirmation(qapp):
    """사람이 직접 잰 거리는 이미 눈으로 확인한 값이다."""
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    dialog.set_length(1.0, "µm")
    dialog.set_manual_pixels(200.0)
    scale = dialog.scale_info()
    assert scale.nm_per_px == pytest.approx(5.0)
    assert scale.source == "manual"


def test_scale_is_none_before_a_length_is_entered(qapp):
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    assert dialog.scale_info() is None


def test_scale_is_none_when_nothing_was_detected_and_nothing_entered(qapp):
    dialog = CalibrationDialog(np.full((200, 200), 100.0))
    dialog.set_length(1.0, "µm")
    assert dialog.scale_info() is None
