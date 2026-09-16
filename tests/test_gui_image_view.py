import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from ebl_gap.types import Roi  # noqa: E402
from ebl_gap_gui.image_view import ImageView  # noqa: E402
from tests.synth import synth_gap_image  # noqa: E402


def test_view_starts_with_no_roi(qapp):
    view = ImageView()
    assert view.current_roi() is None


def test_setting_an_image_creates_a_default_centred_roi(qapp):
    view = ImageView()
    view.set_image(synth_gap_image(width=512, height=512))
    roi = view.current_roi()
    assert roi is not None
    assert 0 <= roi.x0 < roi.x1 < 512
    assert 0 <= roi.y0 < roi.y1 < 512


def test_set_roi_round_trips(qapp):
    view = ImageView()
    view.set_image(synth_gap_image(width=512, height=512))
    view.set_roi(Roi(100, 120, 300, 340))
    roi = view.current_roi()
    assert (roi.x0, roi.y0, roi.x1, roi.y1) == (100, 120, 300, 340)


def test_roi_is_clamped_to_the_image_bounds(qapp):
    view = ImageView()
    view.set_image(np.zeros((100, 100)))
    view.set_roi(Roi(-50, -50, 400, 400))
    roi = view.current_roi()
    assert roi.x0 >= 0 and roi.y0 >= 0
    assert roi.x1 <= 99 and roi.y1 <= 99


def test_setting_the_roi_programmatically_emits_roi_changed(qapp):
    view = ImageView()
    view.set_image(synth_gap_image(width=512, height=512))
    seen = []
    view.roi_changed.connect(lambda: seen.append(1))
    view.set_roi(Roi(10, 10, 200, 200))
    assert seen


def test_dragging_the_roi_emits_roi_changed(qapp):
    """마우스 드래그가 타는 경로를 직접 확인한다.

    set_roi()는 자기 본문에서 roi_changed를 명시적으로 발신하므로, 그것만
    테스트하면 pyqtgraph 배선이 완전히 깨져 있어도 통과한다. 실제 드래그는
    RectROI 내부의 setPos/setSize를 거쳐 sigRegionChanged로 나오므로 그 경로를
    직접 두드려야 한다.
    """
    view = ImageView()
    view.set_image(synth_gap_image(width=512, height=512))
    seen = []
    view.roi_changed.connect(lambda: seen.append(1))
    view._roi.setPos([120, 130])
    assert seen


def test_overlay_can_be_shown_and_cleared(qapp):
    view = ImageView()
    view.set_image(synth_gap_image(width=64, height=64))
    overlay = np.zeros((64, 64, 3), dtype=np.uint8)
    overlay[..., 1] = 255
    view.show_overlay(overlay)
    assert view.has_overlay() is True
    view.clear_overlay()
    assert view.has_overlay() is False


def test_changing_the_image_resets_the_overlay(qapp):
    view = ImageView()
    view.set_image(synth_gap_image(width=64, height=64))
    view.show_overlay(np.zeros((64, 64, 3), dtype=np.uint8))
    view.set_image(synth_gap_image(width=64, height=64))
    assert view.has_overlay() is False


def test_empty_image_is_rejected_without_crashing(qapp):
    view = ImageView()
    view.set_image(np.empty((0, 0)))
    assert view.current_roi() is None
