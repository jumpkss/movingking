"""메인 윈도우. 엔진 호출과 위젯 갱신을 배선하기만 한다."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
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
    record_notices,
    render_overlay,
    write_lines_csv,
    write_overlay_png,
    write_summary_csv,
)
from ebl_gap.loader import load_image
from ebl_gap.measure import (
    DatabarOverlapError,
    MeasureParams,
    measure_roi,
)
from ebl_gap.naming import dose_values_for, infer_fields
from ebl_gap.orientation import detect_base_angle_deg
from ebl_gap.profile import extract_profiles
from ebl_gap.stats import representative_line
from ebl_gap.types import Roi
from ebl_gap_gui.calibration import CalibrationDialog
from ebl_gap_gui.dose_plot import DosePlot
from ebl_gap_gui.image_view import ImageView
from ebl_gap_gui.panels import FilePanel, NamingBanner, ResultPanel, ResultTable
from ebl_gap_gui.profile_plot import ProfilePlot

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

#: `Roi`가 허용하는 최소 크기. `ImageView.set_roi`의 하한과 같은 값이다.
MIN_ROI_WIDTH_PX = 9
MIN_ROI_HEIGHT_PX = 5

#: 기본 ROI가 차지하는 비율. 긴 쪽은 갭 축을 따라, 짧은 쪽은 측정 방향으로.
#: 짧은 쪽(측정 방향)은 갭보다 넉넉해야 문턱을 전극 평탄부에서 잡는다 — 갭
#: 두께의 4~5배가 목표다. 갭 두께를 아직 모르므로 이미지 크기의 비율로 잡는다.
DEFAULT_LONG_FRACTION = 0.60
DEFAULT_SHORT_FRACTION = 0.12

#: 갭 방향을 판별할 때 훑어보는 상자의 비율. 예전 기본 ROI와 같은 크기다.
_PROBE_FRACTION = 0.40


def target_gap_note(guess) -> str:
    """파일명의 상수 숫자 칸을 목표 갭 후보로 적는 참고 문구.

    **참고일 뿐이다.** 측정값과 비교하지도, 판정에 쓰지도 않는다 — 파일명은
    의도이지 계측값이 아니다. 이름에 적힌 70 nm와 실측 90 nm가 다를 때 옳은
    것은 실측이고, 둘을 자동으로 견주면 툴이 이름을 근거로 계측을 의심하게 된다.
    """
    values = sorted(guess.constant_numeric.values())
    if not values:
        return ""
    if len(values) == 1:
        head = f"파일명 기준 목표 갭 {values[0]:g} nm"
    else:
        head = ("파일명 기준 목표 갭 후보 "
                + ", ".join(f"{value:g}" for value in values) + " nm")
    return f"{head} (참고용 — 측정값과 비교하지 않습니다)"


def _centred_roi(width: int, height: int, box_width: int,
                 box_height: int) -> Roi:
    """주어진 영역 한가운데에 놓인 상자. 크기는 `Roi`의 하한까지 보장한다."""
    box_width = max(MIN_ROI_WIDTH_PX, min(width, box_width))
    box_height = max(MIN_ROI_HEIGHT_PX, min(height, box_height))
    x0 = max(0, (width - box_width) // 2)
    y0 = max(0, (height - box_height) // 2)
    return Roi(x0, y0, x0 + box_width - 1, y0 + box_height - 1)


def default_roi(image, databar_top: int | None = None) -> Roi:
    """물려받을 ROI가 없을 때 띄우는 상자. 긴 쪽을 갭 축에 맞춘다.

    갭이 가로면 가로로 긴 상자, 세로면 세로로 긴 상자다. 방향은 Task 28의
    `detect_base_angle_deg`가 판별한다 — 여기서 따로 세지 않는다.

    데이터바는 빼고 본다. 데이터바는 균일한 띠라 판별을 흐리고, 무엇보다 긴
    쪽이 거기까지 뻗으면 `measure_roi`가 첫 측정을 `DatabarOverlapError`로
    거부한다. 폴더를 연 직후 "ROI를 다시 잡으세요"만 나오는 프로그램이 된다.

    판별이 실패하면(이미지가 판별할 수 없을 만큼 작은 경우) 가로로 긴 상자를
    쓴다 — 사용자의 S/D 패턴이 가로이고, 세로 갭이면 한 번만 돌려 주면 그
    뒤로는 마지막 ROI가 유지한다.
    """
    array = np.asarray(image)
    height, width = int(array.shape[0]), int(array.shape[1])
    usable_height = height if databar_top is None \
        else max(MIN_ROI_HEIGHT_PX, min(height, int(databar_top)))

    try:
        probe = _centred_roi(width, usable_height,
                             int(width * _PROBE_FRACTION),
                             int(usable_height * _PROBE_FRACTION))
        base_angle_deg = detect_base_angle_deg(array, probe)
    except Exception:
        base_angle_deg = 90.0

    if abs(base_angle_deg) >= 45.0:     # 가로 갭 — 갭 축이 이미지 가로다
        box_width = int(width * DEFAULT_LONG_FRACTION)
        box_height = int(usable_height * DEFAULT_SHORT_FRACTION)
    else:                               # 세로 갭 — 갭 축이 이미지 세로다
        box_width = int(width * DEFAULT_SHORT_FRACTION)
        box_height = int(usable_height * DEFAULT_LONG_FRACTION)
    return _centred_roi(width, usable_height, box_width, box_height)


def fit_roi_to_image(roi: Roi, shape: tuple[int, int]) -> tuple[Roi, bool]:
    """물려받은 ROI를 이 이미지 안에 넣는다. 가로:세로 비율은 지킨다.

    비율을 버리고 한 변씩 잘라 넣으면 측정 방향(짧은 쪽) 폭이 조용히 달라진다.
    사용자가 맞춰 놓은 것은 위치만이 아니라 "갭 두께의 네댓 배"라는 모양이고,
    그 모양이 문턱을 전극 평탄부에서 잡게 해 준다. 그래서 두 변에 같은 배율을
    먹여 줄인 뒤 경계 안으로 민다.

    Returns:
        (맞춘 ROI, 실제로 줄이거나 옮겼는가)
    """
    height, width = shape
    scale = min(1.0, width / roi.width, height / roi.height)
    new_width = max(MIN_ROI_WIDTH_PX, min(width, int(roi.width * scale)))
    new_height = max(MIN_ROI_HEIGHT_PX, min(height, int(roi.height * scale)))
    x0 = max(0, min(roi.x0, width - new_width))
    y0 = max(0, min(roi.y0, height - new_height))
    fitted = Roi(x0, y0, x0 + new_width - 1, y0 + new_height - 1)
    return fitted, fitted != roi


class MainWindow(QMainWindow):
    """폴더 열기 -> ROI 드래그 -> 측정 -> 내보내기."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("EBL dose test 갭 분석")
        # 지원 하한. 이보다 좁으면 파일 이름 열이 패널 밖으로 밀려나고 이미지
        # 뷰가 쓸 수 없을 만큼 눌린다. 상수를 고르는 대신 실측으로 정했다:
        # 800x600에서 이름 열 159px < 뷰포트 185px, 여기에 여유를 둔 값이다.
        self.setMinimumSize(900, 650)
        self.session = Session()
        self.params = MeasureParams()

        self._pixels: dict[int, np.ndarray] = {}
        self._databar_tops: dict[int, int | None] = {}
        self._current: int | None = None
        self._status = ""

        self.file_panel = FilePanel()
        self.naming_banner = NamingBanner()
        # 파일명 추정이 채워 넣은 행. [사용 안 함]이 되돌릴 대상이 정확히 이것이다 —
        # 사용자가 손으로 고친 값이나 uC 규칙이 읽은 값까지 지우면 안 된다.
        self._inferred_doses: set[int] = set()
        self._naming_names: list[str] = []
        self.image_view = ImageView()
        self.result_panel = ResultPanel()
        self.profile_plot = ProfilePlot()
        # 이미지마다 따로 보관한다. dose 시리즈를 오가며 보는 것이 이 프로그램의
        # 사용 방식인데, 현재 한 장만 들고 있으면 돌아올 때마다 다시 측정해야 한다.
        # 열쇠는 세션 인덱스이므로 open_folder에서 반드시 함께 비운다.
        self._profiles_by_index: dict[int, np.ndarray] = {}
        self._measured_rois: dict[int, Roi] = {}
        # 아직 안 잰 이미지가 물려받을 ROI. 같은 폴더의 dose 시리즈는 배율도
        # 패턴 위치도 같아서 ROI가 거의 그대로여야 한다. 폴더가 바뀌면 다른
        # 시료이므로 반드시 비운다 — 남겨두면 새 시료에 남의 상자가 얹힌다.
        self._last_roi: Roi | None = None
        self.line_selector = QSpinBox()
        self.line_selector.setPrefix("라인 ")
        self.line_selector.setEnabled(False)
        self.prev_anomaly_button = QPushButton("◀ 이상")
        self.next_anomaly_button = QPushButton("이상 ▶")
        for button in (self.prev_anomaly_button, self.next_anomaly_button):
            button.setEnabled(False)
        self.result_table = ResultTable()
        self.dose_plot = DosePlot()

        # 각도 조작. 스펙 4.2가 "화면에 표시하고 사용자가 수동으로 고정할 수
        # 있다"를 요구한다. 자동 추정이 무너지는 ROI(갭이 가장자리에 붙는 배치)
        # 에서 사용자가 손으로 잡을 수 있는 유일한 수단이다.
        self.angle_deg_spin = QDoubleSpinBox()
        # estimate_angle_deg는 기준(세로 0도 / 가로 90도)에 arctan 기울기를 얹어
        # 돌려주므로 90을 넘을 수 있다(가로 갭에서 91.5도가 정상값이다). 범위를
        # 그보다 좁히면 무장할 때 값이 조용히 잘려, 화면의 각도와 측정에 쓴
        # 각도가 갈라진다. 하필 잘리는 것이 가로 갭의 정상값이거나 사용자에게
        # 보여줘야 할 붕괴한 각도다.
        self.angle_deg_spin.setRange(-180.0, 180.0)
        self.angle_deg_spin.setDecimals(2)
        self.angle_deg_spin.setSingleStep(0.25)
        self.angle_deg_spin.setSuffix("도")
        self.angle_lock_check = QCheckBox("각도 고정")

        # 스펙 4.3/4.4가 UI 노출을 요구하는 두 값. MeasureParams의 docstring이
        # "GUI가 그대로 노출한다"고 적고도 지금까지 노출하지 않았다.
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.10, 0.90)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(self.params.threshold_fraction)
        self.along_average_spin = QSpinBox()
        self.along_average_spin.setRange(1, 15)
        self.along_average_spin.setSuffix("행")
        self.along_average_spin.setValue(self.params.along_average)

        self.file_panel.selection_changed.connect(self.select_image)
        self.file_panel.dose_edited.connect(lambda *_: self._refresh_session_views())
        # 손으로 고친 값은 더 이상 추정의 것이 아니다. 잊지 않으면 [사용 안 함]이
        # 사용자가 직접 입력한 dose까지 비운다.
        self.file_panel.dose_edited.connect(
            lambda row, _value: self._inferred_doses.discard(row))
        self.naming_banner.field_chosen.connect(self._use_dose_field)
        self.naming_banner.inference_disabled.connect(self._drop_inferred_doses)
        # ROI를 옮기면 그 ROI로 그린 것을 전부 버린다. 측정 설정 변경도 같은
        # 슬롯을 탄다 — 처리가 갈라지면 한쪽에만 고친 것이 다른 쪽에 빠진다.
        self.image_view.roi_changed.connect(self._discard_stale_diagnostics)
        # Task 20/22가 세운 위 한 줄(오버레이·프로파일·보관함 삭제)을 건드리지
        # 않고 기억만 따로 붙인다. `_discard_stale_diagnostics`는 측정 설정
        # 변경도 함께 타는 자리라, ROI를 기억하는 일을 거기 섞으면 "ROI가
        # 움직였다"와 "설정이 바뀌었다"가 한 몸이 된다.
        self.image_view.roi_changed.connect(self._remember_roi)
        self.line_selector.valueChanged.connect(self.show_line)
        self.angle_deg_spin.valueChanged.connect(self._angle_controls_changed)
        self.angle_lock_check.toggled.connect(self._angle_controls_changed)
        self.threshold_spin.valueChanged.connect(self._measure_settings_changed)
        self.along_average_spin.valueChanged.connect(self._measure_settings_changed)
        self.prev_anomaly_button.clicked.connect(
            lambda: self._jump_to_anomaly(-1))
        self.next_anomaly_button.clicked.connect(
            lambda: self._jump_to_anomaly(+1))

        # 배너는 파일 목록 바로 위다. 자동으로 채운 dose가 보이는 칸이 그 아래
        # 목록이므로, 무엇을 읽었는지는 같은 눈길 안에 있어야 한다.
        file_box = QWidget()
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.addWidget(self.naming_banner)
        file_layout.addWidget(self.file_panel)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(file_box)
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
        """주 도구 모음이 윗줄을 통째로 쓰고, 측정 설정은 그 아래 줄로 내린다.

        한 줄을 나눠 쓰던 때는 지원 하한 900x650에서 주 도구 모음이 578px 힌트
        대비 404px로 눌려 `라인 CSV`, `오버레이 PNG`, `요약 리포트`가 오버플로
        뒤로 숨었다. 저장이 이 툴의 결과물을 남기는 유일한 경로이므로 가장
        좁은 지원 크기에서 먼저 사라지면 안 되는 쪽이다. 순서도 이 이유로
        정해진다 — 주 도구 모음을 먼저 얹어야 그것이 윗줄을 잡는다.
        """
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
        self.addToolBarBreak()
        self._build_settings_toolbar()

    def _build_settings_toolbar(self) -> None:
        """측정 설정 도구 모음. 엔진의 조절값을 사용자에게 그대로 내준다."""
        bar = self.addToolBar("측정 설정")
        bar.addWidget(QLabel("갭 축 각도 "))
        bar.addWidget(self.angle_deg_spin)
        bar.addWidget(self.angle_lock_check)
        bar.addSeparator()
        bar.addWidget(QLabel(" 문턱 비율 "))
        bar.addWidget(self.threshold_spin)
        bar.addWidget(QLabel(" 갭 축 이동평균 "))
        bar.addWidget(self.along_average_spin)

    def _remember_roi(self) -> None:
        """방금 놓인 ROI를 다음 이미지가 물려받을 수 있게 기억한다."""
        roi = self.image_view.current_roi()
        if roi is not None:
            self._last_roi = roi

    def _discard_stale_diagnostics(self) -> None:
        """지금 화면에 있는 것이 더는 현재 조건으로 잰 것이 아니다 — 전부 버린다.

        ROI 이동과 설정 변경이 같은 처리를 받는 자리다. 둘 다 (무엇으로 쟀는가,
        무엇이 그려져 있는가) 쌍을 깨뜨린다. 오버레이, 미니 플롯, 프로파일
        보관함, 라인 조작이 함께 사라져야 반쪽짜리 화면이 남지 않는다.
        보관함이 비면 오버레이 PNG 내보내기도 같은 이유로 거부된다.
        """
        self.image_view.clear_overlay()
        self._clear_profile()

    def _measure_settings_changed(self, *_) -> None:
        """조절값을 엔진 파라미터에 옮기고, 그 값으로 재지 않은 그림을 버린다.

        MeasureParams는 frozen이므로 통째로 갈아 끼운다. 여기서 다시 측정하지는
        않는다 — 스핀박스를 한 칸씩 돌리는 동안 ROI 전체를 매번 재게 된다.

        새 params로 미니 플롯만 다시 그리지도 않는다. 그렇게 하면 0.30 문턱선이
        0.50으로 잡은 에지 위에 겹쳐 그려진다 — `ProfilePlot.show_line`의 주석이
        금지한 그림이고, 사용자는 그것을 "새 문턱으로 잰 결과"로 읽는다.
        섞인 그림보다 빈 그림이 낫다.
        """
        self.params = replace(
            self.params,
            threshold_fraction=self.threshold_spin.value(),
            along_average=self.along_average_spin.value(),
        )
        self._discard_stale_diagnostics()
        self._set_status("측정 설정이 바뀌었습니다 — 다시 측정하세요")

    def _angle_controls_changed(self, *_) -> None:
        """각도 조작은 다음 측정부터 반영된다는 것을 말해 준다.

        여기서 곧바로 다시 측정하지 않는 이유는, 스핀박스를 한 칸씩 돌리는
        동안 매 단계마다 ROI 전체를 다시 재게 되기 때문이다.
        """
        if self.angle_lock_check.isChecked():
            self._set_status(
                f"갭 축 각도 {self.angle_deg_spin.value():.2f}도로 고정 — "
                "다시 측정하세요"
            )
        else:
            self._set_status("갭 축 각도 자동 추정 — 다시 측정하세요")

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
        # 보관함의 열쇠는 세션 인덱스다. 비우지 않으면 새 폴더의 0번이 이전 폴더
        # 0번의 프로파일과 ROI를 물려받아, 다른 시료의 진단을 이 시료의 것으로
        # 읽게 된다. 이 두 줄이 이 기능에서 가장 위험한 자리다.
        self._profiles_by_index.clear()
        self._measured_rois.clear()
        # 물려받기는 한 폴더 안에서만 뜻이 있다. 다른 시료로 넘어가면서까지
        # 끌고 가면 새 폴더의 첫 장이 남의 상자를 쓴 채로 열린다.
        self._last_roi = None
        self._current = None

        for index, path in enumerate(paths):
            loaded = load_image(path)
            self.session.add(loaded.record)
            self._pixels[index] = loaded.pixels
            self._databar_tops[index] = loaded.databar_top

        # 목록을 채우기 전에 읽는다. dose 칸과 결과 패널의 참고 줄이 첫 화면부터
        # 맞게 보여야 한다.
        self._apply_naming_guess([path.name for path in paths])

        self.file_panel.set_records(self.session.records)
        for index, pixels in self._pixels.items():
            self.file_panel.set_thumbnail(index, pixels)
        self._refresh_session_views()
        if self.session.records:
            # 첫 장 선택은 set_records가 발신하는 selection_changed(0)이 한다.
            # 여기서 또 부르면 렌더와 ROI 리셋이 두 번 돈다.
            #
            # 그 선택이 띄운 안내를 덮지 않고 이어 붙인다. 덮어쓰던 때는 폴더의
            # 0번에 붙은 안내만 화면에서 사라졌다 — 사용자가 클릭하지 않아도
            # 선택되는 유일한 장이라 다시 띄울 기회가 없는 쪽이다.
            loaded = f"{len(self.session.records)}장 불러옴"
            notice = self._record_notice_line(self.session.records[0])
            self._set_status(f"{loaded} | {notice}" if notice else loaded)
        else:
            self.image_view.set_image(np.empty((0, 0)))
            self.result_panel.clear()
            self._set_status("폴더에 이미지가 없습니다")

    def _apply_naming_guess(self, names) -> None:
        """파일명에서 읽은 dose를 자동으로 채운다.

        사용자는 "도즈를 하나하나 입력하지 않도록"을 원했으므로 묻지 않고
        적용한다. 대신 무엇을 읽었는지 배너에 보이고 바꿀 수 있게 한다 —
        추정이지 해독이 아니고, 한 칸 잘못 읽으면 dose-gap 곡선 전체가 아무
        표시 없이 틀린다.

        **이미 dose가 있으면 그쪽이 이긴다.** FEI 메타데이터나 `uC` 규칙이
        읽은 값은 이름 칸의 추측보다 근거가 세다. 파일명 추정은 그것들이
        실패했을 때의 대비책이다.
        """
        self._naming_names = list(names)
        self._inferred_doses = set()
        guess = infer_fields(self._naming_names)
        for index, record in enumerate(self.session.records):
            if record.dose is not None:
                continue
            dose = guess.dose_values.get(record.path.name)
            if dose is not None:
                record.dose = dose
                self._inferred_doses.add(index)

        self.result_panel.set_filename_note(target_gap_note(guess))
        if self._inferred_doses:
            self.naming_banner.show_guess(guess, self._naming_names)
        else:
            # 아무것도 채우지 않았으면 배너도 없다. 채우지 않은 것을 채웠다고
            # 말하는 줄은 사용자를 엉뚱한 칸으로 데려간다.
            self.naming_banner.hide_guess()

    def _use_dose_field(self, field_index: int) -> None:
        """사용자가 dose로 쓸 칸을 직접 골랐다. 추정보다 사용자가 이긴다.

        고른 칸에 숫자가 없는 파일은 dose를 비운다. 그 파일만 예전 값을 남기면
        한 폴더의 dose가 두 규칙에서 섞여 나오고, 어느 것이 어느 쪽인지 화면에
        드러나지 않는다.
        """
        values = dose_values_for(self._naming_names, field_index)
        self._inferred_doses = set()
        for index, record in enumerate(self.session.records):
            record.dose = values.get(record.path.name)
            if record.dose is not None:
                self._inferred_doses.add(index)
            self.file_panel.refresh_row(index)
        self._refresh_session_views()
        self._set_status(
            f"파일명 {field_index + 1}번째 칸을 dose로 읽었습니다 "
            f"({len(self._inferred_doses)}/{len(self.session.records)}장)"
        )

    def _drop_inferred_doses(self) -> None:
        """[사용 안 함]. 파일명에서 채운 dose만 비운다.

        손으로 고친 값과 `uC` 규칙이 읽은 값은 건드리지 않는다 — 추정을 끄는
        것과 사용자가 입력한 것을 지우는 것은 다른 일이다.
        """
        for index in sorted(self._inferred_doses):
            self.session.records[index].dose = None
            self.file_panel.refresh_row(index)
        self._inferred_doses.clear()
        self._refresh_session_views()
        self._set_status("파일명 dose 추정을 껐습니다 — dose를 직접 입력하세요")

    def select_image(self, index: int) -> None:
        if not (0 <= index < len(self.session.records)):
            return
        self._current = index
        record = self.session.records[index]
        roi = self._measured_rois.get(index)
        released = self._release_angle_lock(record)

        # ROI 우선순위: 잰 적이 있으면 그때의 ROI(Task 22), 아니면 마지막으로
        # 쓴 ROI, 그것도 없으면 이 이미지의 기본값. 물려받은 것이 이 이미지에
        # 안 들어가면 모양을 지킨 채 줄이고 그 사실을 말해 준다.
        shrunk = False
        if roi is None and self._last_roi is not None:
            roi, shrunk = fit_roi_to_image(self._last_roi,
                                           self._pixels[index].shape[:2])
        elif roi is None:
            roi = default_roi(self._pixels[index],
                              self._databar_tops.get(index))

        # set_image도 set_roi도 roi_changed를 낸다. 그 신호는 _clear_profile로
        # 이어지고, 그것이 지금 되살리려는 바로 그 보관분을 지운다. 되돌리는
        # 동안만 막되 try/finally로 반드시 되돌린다 — 막힌 채로 남으면 사용자가
        # ROI를 끌어도 낡은 오버레이와 낡은 프로파일이 그대로 남는다.
        blocked = self.image_view.blockSignals(True)
        try:
            self.image_view.set_image(self._pixels[index])
            if roi is not None:
                self.image_view.set_roi(roi)
        finally:
            self.image_view.blockSignals(blocked)

        profiles = self._profiles_by_index.get(index)
        if profiles is None or not record.roi_results:
            # 보관분이 없으면 이 이미지는 아직(또는 ROI를 옮긴 뒤로) 볼 것이 없다.
            self._reset_line_view()
        else:
            result = record.roi_results[0]
            # 오버레이는 방금 되살린 그 ROI와 그 결과에서 다시 그린다. 둘이
            # 한 쌍으로 복원되므로 위치가 어긋날 여지가 없다.
            self.image_view.show_overlay(
                render_overlay(self._pixels[index], roi, result))
            self._show_representative_line(result)

        if record.roi_results:
            self.result_panel.show_result(record.roi_results[0])
        else:
            self.result_panel.clear()
        message = self._record_notice_line(record) or record.path.name
        if shrunk:
            message = (f"{message} | 이미지 크기가 달라 ROI를 줄였습니다 — "
                       "모양(가로:세로)은 그대로입니다")
        if released:
            message = (f"{message} | 각도 고정을 해제했습니다 — "
                       "이 이미지의 각도로 다시 추정합니다")
        self._set_status(message)

    @staticmethod
    def _record_notice_line(record) -> str:
        """이미지에 붙은 안내를 파일 이름과 함께 한 줄로. 없으면 빈 문자열.

        상태 표시줄도 리포트/CSV와 같은 문구를 쓴다. 채널이 화면에서만 뭉개지면
        사용자가 파일을 열었을 때 다른 이야기를 읽게 된다. 이미지 선택과 폴더
        열기가 이 한 자리를 함께 쓰므로 두 경로의 문구가 갈라질 수 없다.
        """
        notices = record_notices(record)
        if not notices:
            return ""
        return f"{record.path.name}: {' | '.join(notices)}"

    def _release_angle_lock(self, record) -> bool:
        """이미지를 떠날 때 각도 고정을 푼다. 실제로 풀었으면 True.

        각도는 그 이미지의 성질이다. 고정이 전역으로 남으면 다음 이미지를 남의
        각도로 재게 되고, 고정한 각도에는 범위 안이면 경고도 붙지 않는다
        (리뷰어 실측: 이미지 0에서 -12도로 고정한 뒤 이미지 1을 재면 참값
        60.0 nm가 61.856 nm). 스핀박스는 이 이미지의 각도로 되돌린다 — 잰 적이
        있으면 그때 쓴 각도, 아직 없으면 0도.

        체크박스 신호를 막는 것은 방어일 뿐, 지금 이 호출에서 막히는 것은
        없다. 이 함수는 select_image 맨 앞에서 불리고 그 끝의 _set_status가
        무조건 다시 쓰므로, _angle_controls_changed가 끼어들어도 사용자는 그
        줄을 보지 못한다 — 실제로 True를 False로 바꿔도 테스트가 다 통과한다.
        (아래 _arm_angle_deg_spin의 같은 모양 가드는 다르다. 그쪽은 측정 직후
        마지막에 불려서 막지 않으면 방금 띄운 측정값 한 줄이 실제로 덮인다.)
        틀린 근거를 단 방어 코드는 다음 사람에게 없는 제약을 믿게 하므로,
        코드는 남기되 이유는 사실대로 적는다. try/finally로 반드시 되돌리는
        것은 진짜다 — 막힌 채로 남으면 이후 사용자의 고정 조작이 전부 조용히
        무시된다.
        """
        was_locked = self.angle_lock_check.isChecked()
        blocked = self.angle_lock_check.blockSignals(True)
        try:
            self.angle_lock_check.setChecked(False)
        finally:
            self.angle_lock_check.blockSignals(blocked)
        angle_deg = (record.roi_results[0].angle_deg
                     if record.roi_results else 0.0)
        self._arm_angle_deg_spin(angle_deg)
        return was_locked

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

        angle_deg = (self.angle_deg_spin.value()
                     if self.angle_lock_check.isChecked() else None)
        # 데이터바 거부는 엔진이 한다. 여기서 한 번 더 판정하면 규칙이 두 군데로
        # 갈라지고, 노트북에서 measure_roi를 직접 부르는 경로는 그중 하나만
        # 받는다. 여기서는 사유를 그대로 상태 표시줄에 옮긴다.
        try:
            result = measure_roi(pixels, roi, record.scale, angle_deg=angle_deg,
                                 params=self.params, databar_top=databar_top)
        except DatabarOverlapError as exc:
            self._set_status(str(exc))
            return
        record.roi_results = [result]  # ROI 하나만 유지한다

        # 프로파일과 그것을 뽑은 ROI를 한 쌍으로 보관한다. 따로 두면 복원할 때
        # 다른 위치의 ROI에 이 프로파일을 붙이게 된다.
        self._profiles_by_index[self._current] = extract_profiles(
            pixels, roi, result.angle_deg,
            along_average=self.params.along_average)
        self._measured_rois[self._current] = roi
        self._show_representative_line(result)

        self.result_panel.show_result(result)
        self.image_view.show_overlay(render_overlay(pixels, roi, result))
        self.file_panel.refresh_row(self._current)
        self._refresh_session_views()

        if result.mean_nm is None:
            message = "갭 측정 불가 — 결과 패널의 경고를 확인하세요"
        else:
            message = (f"갭 {result.mean_nm:.2f} nm "
                       f"(유효 {result.n_valid} 라인)")
        # 세션 전체를 봐야 보이는 진단을 여기에 얹는다. 한 장짜리 메시지만
        # 보여주면 "이 dose에서 갭이 닫혔다"로 읽히는데, 시리즈 전체가 닫혔다면
        # 그쪽이 아니라 설정을 의심해야 한다. 측정 직후 사용자가 읽는 줄은
        # 이것 하나뿐이므로 다른 채널로는 닿지 않는다.
        for warning in self.session.session_warnings():
            message = f"{message} | {warning}"
        self._set_status(message)
        self._arm_angle_deg_spin(result.angle_deg)

    def _arm_angle_deg_spin(self, angle_deg: float) -> None:
        """추정된 각도를 스핀박스에 채운다. 사용자는 이 값을 보고 고정을 정한다.

        채우는 동안의 valueChanged가 _angle_controls_changed를 깨우면, 방금
        띄운 측정 결과 한 줄이 "다시 측정하세요" 안내로 덮인다 — 사용자가 볼
        유일한 측정값이다. try/finally로 반드시 되돌린다: True로 남으면 이후
        사용자가 각도를 돌려도 아무 반응이 없는 죽은 조작이 된다.
        """
        blocked = self.angle_deg_spin.blockSignals(True)
        try:
            self.angle_deg_spin.setValue(float(angle_deg))
        finally:
            self.angle_deg_spin.blockSignals(blocked)

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
            # 다시 그리기만 하면 화면이 그대로라 고장난 버튼과 구별되지 않으므로,
            # 왜 움직이지 않는지 상태 표시줄로 말해 준다.
            self.show_line(target)
            self._set_status("이상 라인이 이것 하나입니다")
        else:
            self.line_selector.setValue(target)

    def _clear_profile(self) -> None:
        """ROI나 측정 설정이 바뀌었다 — 현재 이미지의 보관분까지 버린다.

        보관분을 남겨두면 다음에 이 이미지로 돌아왔을 때 옮기기 전 위치의,
        혹은 바꾸기 전 설정으로 뽑은 프로파일이 되살아난다. 프로파일의 x축은
        그 ROI의 정렬 좌표계이고 값은 그때의 이동평균으로 뽑은 것이므로,
        어긋난 프로파일은 축과 값이 다른 뜻인 채로 읽히는 틀린 진단이다.
        """
        if self._current is not None:
            self._profiles_by_index.pop(self._current, None)
            self._measured_rois.pop(self._current, None)
        self._reset_line_view()

    def _reset_line_view(self) -> None:
        """미니 플롯과 라인 조작만 비운다. 보관분은 건드리지 않는다.

        이미지를 바꿀 때 쓴다 — 떠나는 이미지의 보관분은 그대로 두고 화면만
        비워야 돌아왔을 때 되살릴 것이 남는다.

        조작도 같이 끈다 — 플롯이 비었는데 스핀박스만 살아 있으면 눌렀을 때
        아무 일도 안 일어나는 죽은 버튼이 된다.
        """
        self.profile_plot.clear()
        self.line_selector.setEnabled(False)
        for button in (self.prev_anomaly_button, self.next_anomaly_button):
            button.setEnabled(False)

    def show_line(self, row: int) -> None:
        """특정 스캔라인의 프로파일을 미니 플롯에 띄운다."""
        if self._current is None:
            return
        profiles = self._profiles_by_index.get(self._current)
        if profiles is None:
            return
        record = self.session.records[self._current]
        if not record.roi_results:
            return
        result = record.roi_results[0]
        if not (0 <= row < len(result.lines)) or row >= profiles.shape[0]:
            return
        profile = profiles[row]
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
            self._set_status("내보낼 이미지가 없습니다")
            return
        record = self.session.records[self._current]
        if not record.roi_results:
            self._set_status("먼저 측정하세요")
            return
        # 화면에 있는 ROI가 아니라 *측정에 쓰인* ROI로 그린다. 둘은 사용자가
        # ROI를 옮기거나 측정 설정을 돌리는 순간 갈라지는데, 어느 쪽도
        # roi_results를 지우지 않으므로 지금의 ROI에 그때의 결과를 겹쳐 그리면
        # 평탄한 금속 위에 에지가 찍힌 사진이 파일로 나간다. 화면 오버레이는
        # 지워지지만 이 PNG는 실험 노트에 남아 더 오래 간다. 보관함이 비어
        # 있다는 것이 곧 "이 결과는 지금 조건으로 잰 것이 아니다"라는 뜻이다.
        roi = self._measured_rois.get(self._current)
        if roi is None:
            self._set_status(
                "측정 이후 ROI나 측정 설정이 바뀌었습니다 — 다시 측정하세요")
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
