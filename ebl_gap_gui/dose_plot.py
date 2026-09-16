"""dose에 따른 갭 폭 변화를 보여주는 그래프."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ebl_gap.dataset import ClosedDose, Session

#: short 라인 비율이 이 값에 닿으면 그 dose 점을 빨갛게 표시한다.
#: `stats.SHORT_RATIO_WARN`과 같은 값이고 비교도 `>=`로 같다 — 딱 5%인 dose에서
#: 결과 패널은 경고하는데 곡선에는 표시가 없으면 두 화면이 다른 답을 준다.
#: 분모도 같아야 한다: `DosePoint.n_total`이 `stats.py`와 같은 정의를 쓴다.
SHORT_RATIO_MARK = 0.05

#: 범례에 붙는 이름들. 셋 다 필요하다 — 닫힘 표시와 short 섞인 점은 똑같은
#: 빨간 X라서, 하나만 이름을 달면 나머지가 그 이름으로 읽힌다.
MEASURED_LABEL = "측정된 갭"
SHORT_MIXED_LABEL = "short 섞임"
#: "전 구간 short"에서 끝내면 안 된다. 엔진은 픽셀만으로 닫힌 갭과 평탄한 금속을
#: 구별할 수 없고(둘 다 대비가 없다), 빗나간 ROI도 실측하면 전 구간 short를 낸다.
#: 사용자는 이 곡선을 보고 dose를 고르므로 확인을 요구해야 한다.
CLOSED_LABEL = "전 구간 short (확인 필요)"


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
        self._legend = self._plot.addLegend()
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

        # 전 구간 short인 dose를 먼저 찍는다. 측정된 점이 하나도 없어도 그려야
        # 한다 — 곡선이 통째로 비면 사용자는 "아직 아무것도 안 쟀다"로 읽는다.
        # 다만 이 표시는 결론이 아니라 확인 요청이다: 엔진은 닫힌 갭과 패턴을
        # 벗어난 ROI를 구별할 수 없으므로 범례가 그렇게 말한다.
        self._draw_closed_doses(session.closed_doses())

        if not points:
            return

        doses = np.array([p.dose for p in points], dtype=float)
        means = np.array([p.mean_nm for p in points], dtype=float)
        spreads = np.array(
            [0.0 if p.std_nm is None else p.std_nm for p in points], dtype=float
        )

        self._plot.plot(doses, means, pen=pg.mkPen("#1f77b4", width=2),
                        symbol="o", symbolSize=8, symbolBrush="#1f77b4",
                        name=MEASURED_LABEL)
        self._plot.addItem(pg.ErrorBarItem(x=doses, y=means, height=2 * spreads,
                                           pen=pg.mkPen("#1f77b4")))

        shorted = [p for p in points
                   if p.n_total > 0
                   and p.n_short / p.n_total >= SHORT_RATIO_MARK]
        self._n_shorted = len(shorted)
        if shorted:
            self._plot.plot(
                np.array([p.dose for p in shorted], dtype=float),
                np.array([p.mean_nm for p in shorted], dtype=float),
                pen=None, symbol="x", symbolSize=16,
                symbolPen=pg.mkPen("#d62728", width=3),
                name=SHORT_MIXED_LABEL,
            )

    def _draw_closed_doses(self, closed: list[ClosedDose]) -> None:
        """전 구간 short인 dose를 갭 0 자리에 빨간 X로 찍는다.

        0은 잰 값이 아니라 "여기에는 잴 갭이 없었다"는 자리 표시다. 이름이
        붙지 않으면 그 0이 측정값으로 읽힌다.
        """
        if not closed:
            return
        self._closed_item = self._plot.plot(
            np.array([c.dose for c in closed], dtype=float),
            np.zeros(len(closed), dtype=float),
            pen=None, symbol="x", symbolSize=16,
            symbolPen=pg.mkPen("#d62728", width=3),
            name=CLOSED_LABEL,
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

    def legend_labels(self) -> list[str]:
        """범례에 실제로 그려진 글자들.

        상수가 아니라 범례 위젯이 들고 있는 항목을 읽는다 — 이름을 넘기지 않는
        구현은 여기를 통과하지 못한다.
        """
        return [label.text for _, label in self._legend.items]

    def point_count(self) -> int:
        """측정된 dose 점의 개수. 갭이 닫힌 dose는 여기 들어가지 않는다.

        그 점들에는 잰 갭 폭이 없다. 세려면 `closed_dose_marks()`를 본다.
        """
        return self._n_points

    def shorted_point_count(self) -> int:
        return self._n_shorted

    def warning_text(self) -> str:
        return self._warning.text()
