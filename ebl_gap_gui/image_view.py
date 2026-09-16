"""SEM 이미지를 띄우고 드래그로 ROI를 지정하는 위젯."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ebl_gap.types import Roi

DEFAULT_ROI_FRACTION = 0.4


class ImageView(QWidget):
    """이미지 + 드래그/리사이즈 ROI + 오버레이."""

    roi_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._shape: tuple[int, int] | None = None

        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.invertY(True)  # 이미지 좌표계: y가 아래로 증가
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self._plot.addItem(self._image_item)

        self._overlay_item = pg.ImageItem(axisOrder="row-major")
        self._overlay_item.setZValue(5)
        self._overlay_item.setVisible(False)
        self._plot.addItem(self._overlay_item)

        self._roi = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("y", width=2))
        self._roi.addScaleHandle([1, 1], [0, 0])
        self._roi.addScaleHandle([0, 0], [1, 1])
        self._roi.setZValue(10)
        self._roi.setVisible(False)
        # sigRegionChanged는 ROI 객체를 인자로 넘기며 발신한다. 0-인자 Signal의
        # emit에 직접 연결하면 PySide6가 매 변경마다 TypeError를 던지고 리스너는
        # 호출되지 않는다 — 마우스 드래그는 전부 이 경로를 타므로, 직접 연결하면
        # 프로그램이 set_roi()로 바꿀 때만 신호가 살아 있는 상태가 된다.
        self._roi.sigRegionChanged.connect(lambda *_: self.roi_changed.emit())
        self._plot.addItem(self._roi)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

    def set_image(self, pixels) -> None:
        """새 이미지를 띄운다. 오버레이는 지우고 ROI는 기본 위치로 돌린다."""
        array = np.asarray(pixels, dtype=np.float64)
        self.clear_overlay()
        if array.ndim != 2 or array.size == 0:
            self._shape = None
            self._roi.setVisible(False)
            self._image_item.clear()
            return

        self._shape = (array.shape[0], array.shape[1])
        self._image_item.setImage(array, autoLevels=True)
        self._plot.autoRange()

        height, width = self._shape
        w = max(9, int(width * DEFAULT_ROI_FRACTION))
        h = max(5, int(height * DEFAULT_ROI_FRACTION))
        self.set_roi(Roi((width - w) // 2, (height - h) // 2,
                         (width - w) // 2 + w - 1, (height - h) // 2 + h - 1))

    def set_roi(self, roi: Roi) -> None:
        """ROI를 지정한다. 이미지 밖으로 나가면 경계 안쪽으로 잘라 넣는다."""
        if self._shape is None:
            return
        height, width = self._shape
        x0 = max(0, min(roi.x0, width - 9))
        y0 = max(0, min(roi.y0, height - 5))
        x1 = max(x0 + 8, min(roi.x1, width - 1))
        y1 = max(y0 + 4, min(roi.y1, height - 1))
        self._roi.setVisible(True)
        self._roi.setPos([x0, y0], finish=False)
        self._roi.setSize([x1 - x0 + 1, y1 - y0 + 1], finish=False)
        self.roi_changed.emit()

    def current_roi(self) -> Roi | None:
        if self._shape is None or not self._roi.isVisible():
            return None
        pos = self._roi.pos()
        size = self._roi.size()
        height, width = self._shape
        x0 = int(round(pos.x()))
        y0 = int(round(pos.y()))
        x1 = x0 + int(round(size.x())) - 1
        y1 = y0 + int(round(size.y())) - 1
        x0 = max(0, min(x0, width - 9))
        y0 = max(0, min(y0, height - 5))
        x1 = max(x0 + 8, min(x1, width - 1))
        y1 = max(y0 + 4, min(y1, height - 1))
        try:
            return Roi(x0, y0, x1, y1)
        except ValueError:
            return None

    def show_overlay(self, rgb) -> None:
        array = np.asarray(rgb, dtype=np.uint8)
        self._overlay_item.setImage(array, autoLevels=False)
        self._overlay_item.setVisible(True)

    def clear_overlay(self) -> None:
        self._overlay_item.setVisible(False)

    def has_overlay(self) -> bool:
        return bool(self._overlay_item.isVisible())
