"""dose에 따른 갭 폭 변화를 보여주는 그래프."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ebl_gap.dataset import Session

#: short 라인 비율이 이 값을 넘으면 그 dose 점을 빨갛게 표시한다.
SHORT_RATIO_MARK = 0.05


class DosePlot(QWidget):
    """dose-gap 곡선. short가 섞인 점은 눈에 띄게 표시한다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_points = 0
        self._n_shorted = 0

        self._warning = QLabel("")
        self._warning.setWordWrap(True)
        self._warning.setStyleSheet("color: #b35c00;")

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "dose", units="uC/cm2")
        self._plot.setLabel("left", "갭 폭", units="nm")
        self._plot.showGrid(x=True, y=True, alpha=0.3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._warning)
        layout.addWidget(self._plot)

    def set_session(self, session: Session) -> None:
        self._plot.clear()
        points = session.dose_curve()
        self._n_points = len(points)
        self._n_shorted = 0

        warnings = session.scale_warnings()
        self._warning.setText(" | ".join(warnings))

        if not points:
            return

        doses = np.array([p.dose for p in points], dtype=float)
        means = np.array([p.mean_nm for p in points], dtype=float)
        spreads = np.array(
            [0.0 if p.std_nm is None else p.std_nm for p in points], dtype=float
        )

        self._plot.plot(doses, means, pen=pg.mkPen("#1f77b4", width=2),
                        symbol="o", symbolSize=8, symbolBrush="#1f77b4")
        self._plot.addItem(pg.ErrorBarItem(x=doses, y=means, height=2 * spreads,
                                           pen=pg.mkPen("#1f77b4")))

        shorted = [p for p in points
                   if p.n_valid + p.n_short > 0
                   and p.n_short / (p.n_valid + p.n_short) > SHORT_RATIO_MARK]
        self._n_shorted = len(shorted)
        if shorted:
            self._plot.plot(
                np.array([p.dose for p in shorted], dtype=float),
                np.array([p.mean_nm for p in shorted], dtype=float),
                pen=None, symbol="x", symbolSize=16,
                symbolPen=pg.mkPen("#d62728", width=3),
            )

    def point_count(self) -> int:
        return self._n_points

    def shorted_point_count(self) -> int:
        return self._n_shorted

    def warning_text(self) -> str:
        return self._warning.text()
