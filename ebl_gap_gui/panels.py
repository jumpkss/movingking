"""파일 목록, ROI 결과 요약, 세션 결과 테이블 위젯."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
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
from ebl_gap.types import ImageRecord, RoiResult

FILE_COLUMNS = ("파일", "dose(uC)", "상태")

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
        self._loading = False

        self._table = QTableWidget(0, len(FILE_COLUMNS))
        self._table.setHorizontalHeaderLabels(FILE_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self._table.currentCellChanged.connect(self._on_current_cell_changed)
        self._table.itemChanged.connect(self._on_item_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)

    def set_records(self, records) -> None:
        self._loading = True
        self._records = list(records)
        self._table.setRowCount(len(self._records))
        for index in range(len(self._records)):
            self._fill_row(index)
        self._loading = False
        if self._records:
            self._table.setCurrentCell(0, 0)

    def _fill_row(self, index: int) -> None:
        record = self._records[index]

        name = QTableWidgetItem(record.path.name)
        name.setFlags(name.flags() & ~Qt.ItemIsEditable)
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
        self._fill_row(index)
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

        parts = [
            head,
            f"유효 {result.n_valid} / short {result.n_short} / "
            f"판정보류 {result.n_uncertain} 라인",
            f"정밀도 주의 {result.n_low_confidence} 라인",
            f"갭 각도 {result.angle_deg:.2f}도",
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
