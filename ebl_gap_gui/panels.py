"""파일 목록, ROI 결과 요약, 세션 결과 테이블 위젯."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ebl_gap.dataset import Session
from ebl_gap.profile import measurement_runs_down
from ebl_gap.types import ImageRecord, RoiResult

FILE_COLUMNS = ("파일", "dose(uC)", "상태")

# 56에서 줄였다. 좁은 파일 패널에서 56px 아이콘은 이름 칸을 29px만 남겨
# 파일명을 통째로 생략시킨다.
THUMBNAIL_SIZE = 40


def to_thumbnail_icon(pixels, size: int = THUMBNAIL_SIZE) -> QIcon:
    """2차원 밝기 배열을 목록에 넣을 회색조 아이콘으로 만든다."""
    array = np.asarray(pixels, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        return QIcon()

    lo, hi = float(array.min()), float(array.max())
    span = hi - lo if hi > lo else 1.0
    gray = np.clip((array - lo) / span * 255.0, 0, 255).astype(np.uint8)

    step = max(1, max(gray.shape) // size)
    gray = np.ascontiguousarray(gray[::step, ::step])
    height, width = gray.shape
    # .copy()로 numpy 버퍼에서 떼어낸다. 떼지 않으면 배열이 해제될 때 화면이 깨진다.
    image = QImage(gray.data, width, height, width,
                   QImage.Format_Grayscale8).copy()
    return QIcon(QPixmap.fromImage(image))


TABLE_COLUMNS = (
    ("file", "파일"),
    ("dose_uC", "dose(uC)"),
    ("mean_nm", "갭(nm)"),
    ("std_nm", "표준편차(nm)"),
    ("n_valid", "유효"),
    ("n_short", "short"),
    ("n_uncertain", "판정보류"),
    ("angle_deg", "각도(도)"),
    ("scale_source", "스케일 출처"),
)


def _fmt(value, digits=2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


class FilePanel(QWidget):
    """불러온 이미지 목록. dose는 셀을 눌러 직접 고칠 수 있다."""

    selection_changed = Signal(int)
    dose_edited = Signal(int, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[ImageRecord] = []
        self._thumbnails: dict[int, QIcon] = {}
        self._loading = False

        self._table = QTableWidget(0, len(FILE_COLUMNS))
        self._table.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self._table.setHorizontalHeaderLabels(FILE_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self._table.horizontalHeader()
        # 이름 열은 내용에 맞춘다. Stretch로 두면 패널이 좁아질 때 이름부터
        # 잘리는데, 이름은 dose가 안 잡히는 파일에서 행을 구분하는 유일한
        # 수단이다. 남는 폭은 마지막 열이 먹고, 모자라면 표가 가로로 스크롤된다.
        # Stretch였을 때는 여유가 창 크기에 따라 변했다(1400x900에서 +113px,
        # 1000x700에서 정확히 0px, 640x480에서 -119px). 내용 크기로 잡으면
        # 어느 크기에서도 일정한 여유가 남는다 — 글꼴이 큰 환경에서 이름이
        # 잘리던 것이 테스트의 취약함이기 전에 레이아웃의 취약함이었다.
        #
        # dose/상태 칸도 내용만큼만 차지한다.
        #
        # 여기에 setMinimumSectionSize로 바닥을 까는 방법은 쓰지 않는다. 그 설정은
        # 칸별이 아니라 헤더 전체에 걸린다 — 130을 주면 dose/상태 칸까지 130이
        # 되어 366픽셀 패널을 셋이 나눠 갖고, 정작 첫 칸은 바닥값 130에 눌린다.
        # 고치려던 증상(이름이 잘림)이 그대로 남는다.
        for column in (0, 1, 2):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        # 썸네일이 행 높이에 눌려 들어가지 않게 행을 내용에 맞춘다.
        self._table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.currentCellChanged.connect(self._on_current_cell_changed)
        self._table.itemChanged.connect(self._on_item_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)

    def set_records(self, records) -> None:
        self._loading = True
        try:
            self._records = list(records)
            self._thumbnails.clear()
            self._table.setRowCount(len(self._records))
            for index in range(len(self._records)):
                self._fill_row(index)
            if self._records:
                self._table.setCurrentCell(0, 0)
        finally:
            # 예외가 나도 플래그를 반드시 내린다. True로 남으면 이후 사용자의 dose
            # 편집이 전부 조용히 무시되는데, 그것이 잘못 파싱된 dose를 바로잡는
            # 유일한 경로다. 실패가 눈에 보이지도 않는다.
            self._loading = False
        if self._records:
            # Qt의 currentCellChanged는 인덱스가 실제로 바뀔 때만 발신한다. 0행이
            # 선택된 채로 다른 폴더를 열면 setCurrentCell(0, 0)이 no-op이라 신호가
            # 나가지 않고, 결과 패널이 이전 폴더의 결과를 계속 보여준다. 위에서
            # _loading으로 암묵 발신을 막았으므로 여기서 정확히 한 번 발신된다.
            self.selection_changed.emit(0)

    def _fill_row(self, index: int) -> None:
        record = self._records[index]

        name = QTableWidgetItem(record.path.name)
        name.setFlags(name.flags() & ~Qt.ItemIsEditable)
        icon = self._thumbnails.get(index)
        if icon is not None:
            name.setIcon(icon)
        self._table.setItem(index, 0, name)

        dose = QTableWidgetItem("" if record.dose is None else f"{record.dose:g}")
        self._table.setItem(index, 1, dose)

        status = QTableWidgetItem(self._status_for(record))
        status.setFlags(status.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(index, 2, status)

    @staticmethod
    def _status_for(record: ImageRecord) -> str:
        if record.error and record.scale is None:
            return f"오류: {record.error[:20]}"
        if record.roi_results:
            return f"측정 {len(record.roi_results)}건"
        return "미측정"

    def refresh_row(self, index: int) -> None:
        self._loading = True
        try:
            self._fill_row(index)
        finally:
            self._loading = False

    def _on_current_cell_changed(self, row: int, _col, _prow, _pcol) -> None:
        if not self._loading and 0 <= row < len(self._records):
            self.selection_changed.emit(row)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() != 1:
            return
        row = item.row()
        text = item.text().strip()
        try:
            dose = float(text) if text else None
        except ValueError:
            self.refresh_row(row)
            return
        if dose is not None and dose < 0:
            # 음수 dose는 물리적으로 불가능하다. 파싱 불가와 같이 되돌린다 —
            # 조용히 받아들이면 dose-gap 곡선의 x축이 틀어진 채로 해석된다.
            self.refresh_row(row)
            return
        self._records[row].dose = dose
        self.dose_edited.emit(row, dose)

    def set_dose(self, index: int, dose: float | None) -> None:
        self._records[index].dose = dose
        self.refresh_row(index)

    def select(self, index: int) -> None:
        self._table.setCurrentCell(index, 0)

    def current_index(self) -> int | None:
        row = self._table.currentRow()
        return row if 0 <= row < len(self._records) else None

    def row_count(self) -> int:
        return self._table.rowCount()

    def dose_text(self, index: int) -> str:
        item = self._table.item(index, 1)
        return "" if item is None else item.text()

    def status_text(self, index: int) -> str:
        item = self._table.item(index, 2)
        return "" if item is None else item.text()

    def set_thumbnail(self, index: int, pixels) -> None:
        """이미지 미리보기를 목록 행에 붙인다."""
        icon = to_thumbnail_icon(pixels)
        if icon.isNull():
            self._thumbnails.pop(index, None)
        else:
            self._thumbnails[index] = icon
        self.refresh_row(index)

    def has_thumbnail(self, index: int) -> bool:
        return index in self._thumbnails


class ResultPanel(QWidget):
    """선택된 ROI 하나의 결과 요약."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addStretch(1)

    def show_result(self, result: RoiResult) -> None:
        if result.mean_nm is None:
            head = "갭 측정 불가"
        else:
            spread = "" if result.std_nm is None else f" ± {result.std_nm:.2f}"
            head = f"갭 {result.mean_nm:.2f}{spread} nm"

        # 90도 근처의 각도는 오타가 아니라 가로 갭이다. 그 말을 옆에 적어
        # 주지 않으면 사용자가 정상값을 추정 붕괴로 읽는다.
        orientation = ("가로" if measurement_runs_down(result.angle_deg)
                       else "세로")
        parts = [
            head,
            f"유효 {result.n_valid} / short {result.n_short} / "
            f"판정보류 {result.n_uncertain} 라인",
            f"정밀도 주의 {result.n_low_confidence} 라인",
            f"갭 각도 {result.angle_deg:.2f}도 ({orientation} 갭)",
            f"{result.scale.nm_per_px:.4f} nm/px [{result.scale.source}]",
        ]
        if result.warnings:
            parts.append("")
            parts.extend(f"! {w}" for w in result.warnings)
        self._label.setText("\n".join(parts))

    def clear(self) -> None:
        self._label.setText("")

    def text(self) -> str:
        return self._label.text()


class ResultTable(QWidget):
    """세션 전체의 측정 결과 표."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._keys = [key for key, _ in TABLE_COLUMNS]
        self._table = QTableWidget(0, len(TABLE_COLUMNS))
        self._table.setHorizontalHeaderLabels([label for _, label in TABLE_COLUMNS])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)

    def set_session(self, session: Session) -> None:
        rows = [(record, index, result)
                for record in session.records
                for index, result in enumerate(record.roi_results)]
        self._table.setRowCount(len(rows))
        for row, (record, _index, result) in enumerate(rows):
            values = {
                "file": record.path.name,
                "dose_uC": "" if record.dose is None else f"{record.dose:g}",
                "mean_nm": _fmt(result.mean_nm),
                "std_nm": _fmt(result.std_nm),
                "n_valid": str(result.n_valid),
                "n_short": str(result.n_short),
                "n_uncertain": str(result.n_uncertain),
                "angle_deg": f"{result.angle_deg:.2f}",
                "scale_source": result.scale.source,
            }
            for column, key in enumerate(self._keys):
                self._table.setItem(row, column, QTableWidgetItem(values[key]))

    def row_count(self) -> int:
        return self._table.rowCount()

    def cell(self, row: int, key: str) -> str:
        item = self._table.item(row, self._keys.index(key))
        return "" if item is None else item.text()
