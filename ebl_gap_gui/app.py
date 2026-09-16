"""메인 윈도우. 엔진 호출과 위젯 갱신을 배선하기만 한다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from ebl_gap.dataset import Session
from ebl_gap.edges import analyze_profile
from ebl_gap.export import (
    format_report,
    render_overlay,
    write_lines_csv,
    write_overlay_png,
    write_summary_csv,
)
from ebl_gap.loader import load_image
from ebl_gap.measure import MeasureParams, measure_roi
from ebl_gap.profile import extract_profiles
from ebl_gap.stats import representative_line
from ebl_gap_gui.calibration import CalibrationDialog
from ebl_gap_gui.dose_plot import DosePlot
from ebl_gap_gui.image_view import ImageView
from ebl_gap_gui.panels import FilePanel, ResultPanel, ResultTable
from ebl_gap_gui.profile_plot import ProfilePlot

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class MainWindow(QMainWindow):
    """폴더 열기 -> ROI 드래그 -> 측정 -> 내보내기."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("EBL dose test 갭 분석")
        self.session = Session()
        self.params = MeasureParams()

        self._pixels: dict[int, np.ndarray] = {}
        self._databar_tops: dict[int, int | None] = {}
        self._current: int | None = None
        self._status = ""

        self.file_panel = FilePanel()
        self.image_view = ImageView()
        self.result_panel = ResultPanel()
        self.profile_plot = ProfilePlot()
        self._profiles: np.ndarray | None = None
        self.line_selector = QSpinBox()
        self.line_selector.setPrefix("라인 ")
        self.line_selector.setEnabled(False)
        self.prev_anomaly_button = QPushButton("◀ 이상")
        self.next_anomaly_button = QPushButton("이상 ▶")
        for button in (self.prev_anomaly_button, self.next_anomaly_button):
            button.setEnabled(False)
        self.result_table = ResultTable()
        self.dose_plot = DosePlot()

        self.file_panel.selection_changed.connect(self.select_image)
        self.file_panel.dose_edited.connect(lambda *_: self._refresh_session_views())
        # ROI를 옮기면 낡은 오버레이를 지운다. 측정 결과는 그 ROI에 묶여 있으므로,
        # 새 위치에 이전 위치의 에지가 그려진 채로 남으면 사용자가 틀린 그림을
        # 보고 판단하게 된다. 다시 측정할 때 새 오버레이가 그려진다.
        self.image_view.roi_changed.connect(self.image_view.clear_overlay)
        # 미니 플롯도 ROI에 묶여 있다. x축이 그 ROI의 정렬 좌표계이므로
        # ROI가 움직이면 축 자체가 다른 뜻이 된다.
        self.image_view.roi_changed.connect(self._clear_profile)
        self.line_selector.valueChanged.connect(self.show_line)
        self.prev_anomaly_button.clicked.connect(
            lambda: self._jump_to_anomaly(-1))
        self.next_anomaly_button.clicked.connect(
            lambda: self._jump_to_anomaly(+1))

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.file_panel)
        splitter.addWidget(self.image_view)
        # 미니 플롯과 라인 조작 줄은 한 덩어리다 — 스플리터가 둘을 갈라 놓으면
        # 사용자가 플롯만 남기고 조작을 접어버릴 수 있다.
        profile_box = QWidget()
        profile_layout = QVBoxLayout(profile_box)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.addWidget(self.profile_plot)
        line_bar = QHBoxLayout()
        line_bar.setContentsMargins(0, 0, 0, 0)
        line_bar.addWidget(self.prev_anomaly_button)
        line_bar.addWidget(self.line_selector)
        line_bar.addWidget(self.next_anomaly_button)
        line_bar.addStretch(1)
        profile_layout.addLayout(line_bar)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.result_panel)
        right.addWidget(profile_box)
        right.setSizes([500, 260])
        splitter.addWidget(right)
        splitter.setSizes([360, 700, 300])
        self.setCentralWidget(splitter)

        tabs = QTabWidget()
        tabs.addTab(self.result_table, "결과 테이블")
        tabs.addTab(self.dose_plot, "dose-gap 곡선")
        dock = QDockWidget("세션", self)
        dock.setWidget(tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        # 바닥 도크가 중앙 영역을 눌러 미니 플롯을 80픽셀로 만들었다. 초기
        # 배치만 줄인다 — setMaximumHeight로 막으면 결과 테이블을 넓게 보려는
        # 사용자가 영영 늘릴 수 없게 된다.
        self.resizeDocks([dock], [260], Qt.Vertical)

        self._build_toolbar()
        self.statusBar().showMessage("폴더를 열어 시작하세요")

    # ------------------------------------------------------------------ 배선

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("주요 동작")
        bar.addAction("폴더 열기", self._choose_folder)
        bar.addAction("측정", self.measure_current)
        bar.addAction("스케일 캘리브레이션", self.calibrate_current)
        bar.addSeparator()
        bar.addAction("요약 CSV", lambda: self._save_as(self.export_summary_csv,
                                                        "summary.csv"))
        bar.addAction("라인 CSV", lambda: self._save_as(self.export_lines_csv,
                                                        "lines.csv"))
        bar.addAction("오버레이 PNG", lambda: self._save_as(self.export_overlay,
                                                            "overlay.png"))
        bar.addAction("요약 리포트", lambda: self._save_as(self.export_report,
                                                           "report.txt"))

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "SEM 이미지 폴더 선택")
        if folder:
            self.open_folder(folder)

    def _save_as(self, handler, default_name: str) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "저장", default_name)
        if path:
            handler(path)

    def _set_status(self, message: str) -> None:
        self._status = message
        self.statusBar().showMessage(message)

    def status_text(self) -> str:
        return self._status

    # ------------------------------------------------------------------ 동작

    def open_folder(self, folder: str | Path) -> None:
        """폴더 안의 이미지를 전부 읽어 세션을 새로 만든다."""
        folder = Path(folder)
        paths = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in IMAGE_SUFFIXES)

        self.session = Session()
        self._pixels.clear()
        self._databar_tops.clear()
        self._current = None

        for index, path in enumerate(paths):
            loaded = load_image(path)
            self.session.add(loaded.record)
            self._pixels[index] = loaded.pixels
            self._databar_tops[index] = loaded.databar_top

        self.file_panel.set_records(self.session.records)
        for index, pixels in self._pixels.items():
            self.file_panel.set_thumbnail(index, pixels)
        self._refresh_session_views()
        if self.session.records:
            # 첫 장 선택은 set_records가 발신하는 selection_changed(0)이 한다.
            # 여기서 또 부르면 렌더와 ROI 리셋이 두 번 돈다.
            self._set_status(f"{len(self.session.records)}장 불러옴")
        else:
            self.image_view.set_image(np.empty((0, 0)))
            self.result_panel.clear()
            self._set_status("폴더에 이미지가 없습니다")

    def select_image(self, index: int) -> None:
        if not (0 <= index < len(self.session.records)):
            return
        self._current = index
        self.image_view.set_image(self._pixels[index])
        self._clear_profile()

        record = self.session.records[index]
        if record.roi_results:
            self.result_panel.show_result(record.roi_results[0])
        else:
            self.result_panel.clear()
        if record.error:
            self._set_status(f"{record.path.name}: {record.error}")
        else:
            self._set_status(record.path.name)

    def measure_current(self) -> None:
        """현재 이미지의 ROI를 측정하고 결과를 화면 전체에 반영한다."""
        if self._current is None:
            self._set_status("측정할 이미지가 없습니다")
            return
        record = self.session.records[self._current]
        if record.scale is None:
            self._set_status(
                "스케일이 확정되지 않았습니다. 스케일 캘리브레이션을 먼저 하세요"
            )
            return
        roi = self.image_view.current_roi()
        if roi is None:
            self._set_status("ROI를 드래그해서 지정하세요")
            return

        pixels = self._pixels[self._current]
        databar_top = self._databar_tops.get(self._current)
        if databar_top is not None and roi.y1 >= databar_top:
            self._set_status(
                f"ROI가 데이터바 영역({databar_top}행 이하)을 침범했습니다"
            )
            return

        result = measure_roi(pixels, roi, record.scale, params=self.params)
        record.roi_results = [result]  # ROI 하나만 유지한다

        self._profiles = extract_profiles(pixels, roi, result.angle_deg,
                                          along_average=self.params.along_average)
        self._show_representative_line(result)

        self.result_panel.show_result(result)
        self.image_view.show_overlay(render_overlay(pixels, roi, result))
        self.file_panel.refresh_row(self._current)
        self._refresh_session_views()

        if result.mean_nm is None:
            self._set_status("갭 측정 불가 — 결과 패널의 경고를 확인하세요")
        else:
            self._set_status(f"갭 {result.mean_nm:.2f} nm "
                             f"(유효 {result.n_valid} 라인)")

    def _show_representative_line(self, result) -> None:
        """대표 라인 하나를 미니 플롯에 띄우고 라인 조작을 무장한다.

        어느 라인이 대표인지는 계측 판단이므로 엔진(`representative_line`)이
        정한다. 여기는 그 결과를 화면에 올리기만 한다.
        """
        line = representative_line(result.lines)
        self._arm_line_selector(result, 0 if line is None else line.row)
        if line is not None:
            # 무장하면서 신호를 막았으므로 대표 라인은 여기서 한 번 직접 그린다.
            self.show_line(line.row)

    def _arm_line_selector(self, result, representative_row: int) -> None:
        """측정 결과에 맞춰 라인 조작의 범위와 활성 상태를 맞춘다."""
        anomalies = [ln.row for ln in result.lines if ln.status != "valid"]
        # 채우는 도중의 valueChanged가 show_line을 헛돌게 한다. try/finally로
        # 반드시 되돌린다 — True로 남으면 이후 사용자의 스핀박스 조작이 전부
        # 조용히 무시되고, 그것이 라인을 고르는 유일한 경로다.
        self.line_selector.blockSignals(True)
        try:
            self.line_selector.setRange(0, max(0, len(result.lines) - 1))
            self.line_selector.setValue(representative_row)
        finally:
            self.line_selector.blockSignals(False)
        self.line_selector.setEnabled(bool(result.lines))
        for button in (self.prev_anomaly_button, self.next_anomaly_button):
            button.setEnabled(bool(anomalies))

    def _jump_to_anomaly(self, step: int) -> None:
        """현재 행에서 step 방향으로 가장 가까운 valid 아닌 행으로 간다."""
        if self._current is None:
            return
        record = self.session.records[self._current]
        if not record.roi_results:
            return
        rows = [ln.row for ln in record.roi_results[0].lines
                if ln.status != "valid"]
        if not rows:
            return
        current = self.line_selector.value()
        candidates = [r for r in rows
                      if (r > current if step > 0 else r < current)]
        # 끝에 닿으면 반대쪽 끝으로 감는다. 이상 라인이 한 개뿐일 때도 닿을 수 있다.
        target = (min(candidates) if step > 0 else max(candidates)) \
            if candidates else (min(rows) if step > 0 else max(rows))
        if target == current:
            # 이상 라인이 하나뿐이면 setValue가 no-op이라 valueChanged가 안 난다.
            # 그대로 두면 눌러도 아무 일이 없는 죽은 버튼이 된다.
            self.show_line(target)
        else:
            self.line_selector.setValue(target)

    def _clear_profile(self) -> None:
        """ROI나 이미지가 바뀌면 미니 플롯과 그 원본 프로파일 배열을 함께 버린다.

        둘 중 하나만 지우면 show_line이 다른 자리의 프로파일을 현재 자리의
        것으로 그린다.

        조작도 같이 끈다 — 플롯이 비었는데 스핀박스만 살아 있으면 눌렀을 때
        아무 일도 안 일어나는 죽은 버튼이 된다.
        """
        self._profiles = None
        self.profile_plot.clear()
        self.line_selector.setEnabled(False)
        for button in (self.prev_anomaly_button, self.next_anomaly_button):
            button.setEnabled(False)

    def show_line(self, row: int) -> None:
        """특정 스캔라인의 프로파일을 미니 플롯에 띄운다."""
        if self._profiles is None or self._current is None:
            return
        record = self.session.records[self._current]
        if not record.roi_results:
            return
        result = record.roi_results[0]
        if not (0 <= row < len(result.lines)) or row >= self._profiles.shape[0]:
            return
        profile = self._profiles[row]
        self.profile_plot.show_line(profile, result.lines[row],
                                    analyze_profile(profile,
                                                    **self.params.edge_kwargs))

    def calibrate_current(self) -> None:
        """메타데이터가 없는 이미지의 스케일을 사용자가 정한다."""
        if self._current is None:
            return
        pixels = self._pixels[self._current]
        dialog = CalibrationDialog(pixels,
                                   databar_top=self._databar_tops.get(self._current),
                                   parent=self)
        if dialog.exec() != CalibrationDialog.Accepted:
            return
        scale = dialog.scale_info()
        if scale is None:
            QMessageBox.warning(self, "스케일 확정 실패",
                                "실제 길이와 픽셀 거리를 모두 입력해야 합니다.")
            return
        self.session.records[self._current].scale = scale
        self._set_status(f"스케일 {scale.nm_per_px:.4f} nm/px [{scale.source}]")

    def _refresh_session_views(self) -> None:
        self.result_table.set_session(self.session)
        self.dose_plot.set_session(self.session)

    # -------------------------------------------------------------- 내보내기

    def export_summary_csv(self, path: str | Path) -> None:
        write_summary_csv(path, self.session)
        self._set_status(f"요약 CSV 저장: {path}")

    def export_lines_csv(self, path: str | Path) -> None:
        if self._current is None or not self.session.records[self._current].roi_results:
            self._set_status("먼저 측정하세요")
            return
        write_lines_csv(path, self.session.records[self._current], roi_index=0)
        self._set_status(f"라인 CSV 저장: {path}")

    def export_overlay(self, path: str | Path) -> None:
        if self._current is None:
            return
        record = self.session.records[self._current]
        roi = self.image_view.current_roi()
        if not record.roi_results or roi is None:
            self._set_status("먼저 측정하세요")
            return
        write_overlay_png(path, self._pixels[self._current], roi,
                          record.roi_results[0])
        self._set_status(f"오버레이 저장: {path}")

    def export_report(self, path: str | Path) -> None:
        Path(path).write_text(format_report(self.session), encoding="utf-8")
        self._set_status(f"리포트 저장: {path}")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
