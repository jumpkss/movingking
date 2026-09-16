"""스캔라인 한 줄의 밝기 프로파일과 문턱 위치를 보여주는 진단 플롯.

측정값이 이상할 때 문턱이 어디에 섰는지, 에지가 어디로 잡혔는지 눈으로 확인할 수
있는 유일한 수단이다.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ebl_gap.edges import ProfileAnalysis
from ebl_gap.types import LineResult


class ProfilePlot(QWidget):
    """프로파일 곡선 + 좌우 문턱 + 검출된 에지 위치."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._has_curve = False

        self._title = QLabel("")
        self._title.setWordWrap(True)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "측정 방향 위치", units="px")
        self._plot.setLabel("left", "밝기")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setMaximumHeight(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._plot)

    def show_line(self, profile, line: LineResult,
                  analysis: ProfileAnalysis) -> None:
        values = np.asarray(profile, dtype=np.float64)
        self._plot.clear()
        self._plot.plot(np.arange(values.size, dtype=float), values,
                        pen=pg.mkPen("#1f77b4", width=1))

        # 좌우 문턱을 따로 그린다. 조명이 기울면 두 선의 높이가 달라진다.
        for level, color in (
            (analysis.i_lo + 0.5 * (analysis.i_hi_left - analysis.i_lo), "#888888"),
            (analysis.i_lo + 0.5 * (analysis.i_hi_right - analysis.i_lo), "#bbbbbb"),
        ):
            self._plot.addItem(pg.InfiniteLine(
                pos=level, angle=0,
                pen=pg.mkPen(color, style=Qt.PenStyle.DashLine)))

        for edge in (line.left_px, line.right_px):
            if edge is not None:
                self._plot.addItem(pg.InfiniteLine(
                    pos=edge, angle=90, pen=pg.mkPen("#2ca02c", width=2)))

        if line.width_nm is None:
            detail = "폭 측정 없음"
        else:
            detail = f"폭 {line.width_px:.2f} px = {line.width_nm:.2f} nm"
        flags = f" [{' '.join(sorted(line.flags))}]" if line.flags else ""
        reason = f" — {line.reason}" if line.reason else ""
        self._title.setText(
            f"행 {line.row} · {line.status}{flags} · {detail}{reason}"
        )
        self._has_curve = True

    def clear(self) -> None:
        self._plot.clear()
        self._title.setText("")
        self._has_curve = False

    def has_curve(self) -> bool:
        return self._has_curve

    def title_text(self) -> str:
        return self._title.text()
