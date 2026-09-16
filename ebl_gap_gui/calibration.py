"""메타데이터가 없는 이미지의 스케일을 사람이 확정하는 창.

스케일바 라벨을 OCR로 읽지 않고 사용자에게 묻는다. 잘못 읽은 숫자 하나로 모든
측정값이 조용히 틀어지는 것보다 한 번 묻는 편이 낫다.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from ebl_gap.scalebar import (
    detect_scalebar,
    scale_from_scalebar,
    scale_from_two_points,
)
from ebl_gap.types import ScaleInfo

UNIT_FACTORS = {"nm": 1.0, "µm": 1000.0}


class CalibrationDialog(QDialog):
    """검출된 스케일바 길이 또는 사용자가 잰 픽셀 거리로 스케일을 만든다."""

    def __init__(self, pixels, databar_top: int | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("스케일 캘리브레이션")

        array = np.asarray(pixels, dtype=np.float64)
        hit = detect_scalebar(array, databar_top=databar_top) if array.size else None
        self._detected_px: int | None = hit.length_px if hit else None

        if hit is None:
            summary = ("스케일바를 자동으로 찾지 못했습니다. "
                       "이미지에서 두 점을 찍어 픽셀 거리를 입력하세요.")
        else:
            # 어디서 찾았는지 반드시 보여준다. 데이터바의 밝은 텍스트가 막대보다 긴
            # 런을 만들면 검출기가 아무 신호 없이 텍스트 좌표를 돌려주기 때문이다.
            summary = (
                f"스케일바 막대를 {hit.length_px} px로 검출했습니다 "
                f"(행 {hit.row}, x {hit.x0}~{hit.x1}). "
                "데이터바의 밝은 텍스트를 막대로 잘못 잡을 수 있으니 "
                "위치가 맞는지 확인한 뒤 아래를 체크하세요."
            )
        self._summary = QLabel(summary)
        self._summary.setWordWrap(True)

        self._confirm = QCheckBox("검출된 막대가 맞습니다")
        self._confirm.setEnabled(hit is not None)

        self._length = QDoubleSpinBox()
        self._length.setDecimals(4)
        self._length.setRange(0.0, 1e6)
        self._length.setValue(0.0)

        self._unit = QComboBox()
        self._unit.addItems(list(UNIT_FACTORS))
        self._unit.setCurrentText("µm")

        self._manual = QDoubleSpinBox()
        self._manual.setDecimals(2)
        self._manual.setRange(0.0, 1e6)
        self._manual.setValue(0.0)

        form = QFormLayout()
        form.addRow("실제 길이", self._length)
        form.addRow("단위", self._unit)
        form.addRow("직접 잰 픽셀 거리 (선택)", self._manual)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._summary)
        layout.addWidget(self._confirm)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def detected_length_px(self) -> int | None:
        return self._detected_px

    def summary_text(self) -> str:
        return self._summary.text()

    def confirm_detection(self, checked: bool) -> None:
        """검출된 막대가 맞다고 사용자가 확인한다."""
        self._confirm.setChecked(bool(checked))

    def set_length(self, value: float, unit: str) -> None:
        self._length.setValue(float(value))
        self._unit.setCurrentText(unit)

    def set_manual_pixels(self, distance_px: float) -> None:
        self._manual.setValue(float(distance_px))

    def length_nm(self) -> float:
        return self._length.value() * UNIT_FACTORS[self._unit.currentText()]

    def scale_info(self) -> ScaleInfo | None:
        """입력이 충분하면 ScaleInfo를, 아니면 None을 돌려준다.

        자동 검출 경로는 사용자가 확인 체크를 해야만 쓴다. 직접 잰 픽셀 거리는
        사람이 이미 이미지를 보고 잰 값이므로 별도 확인이 필요 없다.
        """
        # 위젯 값만 읽는다. 별도 "입력했음" 플래그를 두면 안 된다 — 헬퍼
        # 메서드에서만 세워지고 실제 스핀박스 입력에는 반응하지 않아서,
        # 사용자가 칸에 직접 타이핑하면 그 값이 통째로 무시된다.
        # 스핀박스 기본값과 하한이 둘 다 0이므로 length_nm <= 0이 "미입력"과
        # 같은 뜻이고, 플래그는 불필요할 뿐 아니라 해롭다.
        length_nm = self.length_nm()
        if length_nm <= 0:
            return None

        manual_px = self._manual.value()
        if manual_px > 0:
            return scale_from_two_points((0.0, 0.0), (manual_px, 0.0), length_nm)

        if self._detected_px and self._confirm.isChecked():
            return scale_from_scalebar(float(self._detected_px), length_nm)
        return None
