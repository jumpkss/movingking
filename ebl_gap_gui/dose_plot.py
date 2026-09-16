"""dose에 따른 갭 폭 변화를 보여주는 그래프."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ebl_gap.dataset import ClosedDose, Session

#: short 라인 비율이 이 값에 닿으면 그 dose 점을 빨갛게 표시한다.
#: `stats.SHORT_RATIO_WARN`과 같은 값이고 비교도 `>=`로 같다 — 딱 5%인 dose에서
#: 결과 패널은 경고하는데 곡선에는 표시가 없으면 두 화면이 다른 답을 준다.
SHORT_RATIO_MARK = 0.05


class DosePlot(QWidget):
    """dose-gap 곡선. short가 섞인 점은 눈에 띄게 표시한다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_points = 0
        self._n_shorted = 0
        self._closed_item: pg.PlotDataItem | None = None

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
        self._closed_item = None
        points = session.dose_curve()
        self._n_points = len(points)
        self._n_shorted = 0

        warnings = session.scale_warnings()
        self._warning.setText(" | ".join(warnings))

        # 갭이 닫힌 dose를 먼저 찍는다. 측정된 점이 하나도 없어도 그려야 한다 —
        # 전 구간이 닫힌 시리즈에서 곡선이 통째로 비면 사용자는 "아직 아무것도
        # 안 쟀다"로 읽는다. "이 dose에서 갭이 닫힌다"가 dose test의 답이므로
        # 이 표시가 결론이다.
        self._draw_closed_doses(session.closed_doses())

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
                   and p.n_short / (p.n_valid + p.n_short) >= SHORT_RATIO_MARK]
        self._n_shorted = len(shorted)
        if shorted:
            self._plot.plot(
                np.array([p.dose for p in shorted], dtype=float),
                np.array([p.mean_nm for p in shorted], dtype=float),
                pen=None, symbol="x", symbolSize=16,
                symbolPen=pg.mkPen("#d62728", width=3),
            )

    def _draw_closed_doses(self, closed: list[ClosedDose]) -> None:
        """갭이 닫힌 dose를 갭 0 자리에 빨간 X로 찍는다."""
        if not closed:
            return
        self._closed_item = self._plot.plot(
            np.array([c.dose for c in closed], dtype=float),
            np.zeros(len(closed), dtype=float),
            pen=None, symbol="x", symbolSize=16,
            symbolPen=pg.mkPen("#d62728", width=3),
        )

    def closed_dose_marks(self) -> list[tuple[float, float]]:
        """실제로 그려진 "갭 닫힘" 표시의 좌표.

        플래그가 아니라 그림을 읽는다. 항목이 씬에서 빠지면(예: set_session이
        새 세션으로 지웠으면) 빈 목록이 된다 — 카운터만 올리고 그리지 않는
        구현은 여기를 통과하지 못한다.
        """
        item = self._closed_item
        if item is None or item.scene() is not self._plot.scene():
            return []
        xs, ys = item.getData()
        if xs is None:
            return []
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def point_count(self) -> int:
        """측정된 dose 점의 개수. 갭이 닫힌 dose는 여기 들어가지 않는다.

        그 점들에는 잰 갭 폭이 없다. 세려면 `closed_dose_marks()`를 본다.
        """
        return self._n_points

    def shorted_point_count(self) -> int:
        return self._n_shorted

    def warning_text(self) -> str:
        return self._warning.text()
