"""엔진 전체가 주고받는 데이터 타입.

이 모듈은 numpy조차 import하지 않는다. 타입 정의가 계산 코드에 끌려가지 않도록
의도적으로 비워 둔 것이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SCALE_SOURCES = ("fei_metadata", "scalebar_auto", "manual")

LINE_STATUSES = (
    "valid",
    "short",
    "no_edge",
    "multi_edge",
    "sub_resolution",
    "outlier",
)

#: 통계에서 제외되고 "판정보류"로 집계되는 상태들.
#: "short"는 판정보류가 아니라 별도 카운트이므로 여기 들어가지 않는다.
UNCERTAIN_STATUSES = ("no_edge", "multi_edge", "sub_resolution", "outlier")

LINE_FLAGS = ("low_confidence",)


@dataclass(frozen=True)
class ScaleInfo:
    """픽셀 크기와 그 출처.

    출처를 값과 함께 들고 다니는 것이 이 타입의 존재 이유다. 스케일 출처를 모르는
    계측값은 나중에 재현할 수 없다.
    """

    nm_per_px: float
    source: str

    def __post_init__(self) -> None:
        if self.source not in SCALE_SOURCES:
            raise ValueError(
                f"알 수 없는 source: {self.source!r}. 허용값: {SCALE_SOURCES}"
            )
        if not self.nm_per_px > 0:
            raise ValueError(f"nm_per_px는 양수여야 한다: {self.nm_per_px!r}")


@dataclass(frozen=True)
class Roi:
    """이미지 좌표계의 축 정렬 사각형. 양 끝 픽셀을 모두 포함한다."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        x0, x1 = sorted((int(self.x0), int(self.x1)))
        y0, y1 = sorted((int(self.y0), int(self.y1)))
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "y1", y1)
        if x1 - x0 < 8 or y1 - y0 < 4:
            raise ValueError(
                f"ROI가 너무 작다: 가로 {x1 - x0 + 1}px, 세로 {y1 - y0 + 1}px "
                "(최소 가로 9px, 세로 5px)"
            )

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class LineResult:
    """스캔라인 한 줄의 측정 결과."""

    row: int
    left_px: float | None
    right_px: float | None
    width_px: float | None
    width_nm: float | None
    status: str
    flags: frozenset[str]
    reason: str

    def __post_init__(self) -> None:
        if self.status not in LINE_STATUSES:
            raise ValueError(
                f"알 수 없는 status: {self.status!r}. 허용값: {LINE_STATUSES}"
            )
        unknown = set(self.flags) - set(LINE_FLAGS)
        if unknown:
            raise ValueError(f"알 수 없는 flag: {sorted(unknown)}")

    @property
    def counts_in_stats(self) -> bool:
        return self.status == "valid"


@dataclass(frozen=True)
class RoiResult:
    """ROI 하나의 측정 결과 전체."""

    mean_nm: float | None
    std_nm: float | None
    n_valid: int
    n_short: int
    n_uncertain: int
    n_low_confidence: int
    angle_deg: float
    lines: tuple[LineResult, ...]
    warnings: tuple[str, ...]
    scale: ScaleInfo

    @property
    def n_total(self) -> int:
        return self.n_valid + self.n_short + self.n_uncertain


@dataclass
class ImageRecord:
    """이미지 한 장과 거기서 나온 측정 결과들.

    안내는 두 채널로 갈린다. `error`는 "이 이미지로는 측정할 수 없다"만 쓴다 —
    파일을 못 읽었거나 스케일이 확정되지 않은 경우다. `notes`는 "측정은 되지만
    확인하라"는 안내다. 한 채널에 섞으면 아래쪽이 어두운 멀쩡한 이미지가
    리포트에 `오류:`로 남고 요약 CSV의 경고 칸에 들어간다.
    """

    path: Path
    scale: ScaleInfo | None = None
    dose: float | None = None
    roi_results: list[RoiResult] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)
