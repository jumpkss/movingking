"""메타데이터가 없는 이미지를 위한 스케일 확정 폴백.

데이터바에서 스케일바 막대를 찾아 픽셀 길이를 재거나, 사용자가 직접 두 점을 찍게
한다. 막대가 몇 나노미터인지는 OCR로 읽지 않고 사용자에게 묻는다 — 잘못 읽은 숫자로
모든 측정값이 조용히 틀어지는 것보다 한 번 묻는 편이 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ebl_gap.types import ScaleInfo

#: 전체 가로폭의 이 비율을 넘는 밝은 구간은 스케일바가 아니라 배경으로 본다.
MAX_BAR_WIDTH_RATIO = 0.9


@dataclass(frozen=True)
class ScalebarHit:
    """검출된 스케일바 막대의 위치."""

    row: int
    x0: int
    x1: int

    @property
    def length_px(self) -> int:
        return self.x1 - self.x0 + 1


def _longest_bright_run(mask: np.ndarray) -> tuple[int, int, int]:
    """불리언 행에서 가장 긴 True 구간의 (길이, 시작, 끝)을 구한다."""
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    if changes.size == 0:
        return 0, 0, 0
    starts, ends = changes[0::2], changes[1::2]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(lengths[best]), int(starts[best]), int(ends[best] - 1)


def detect_scalebar(image, *, databar_top: int | None = None,
                    min_length_px: int = 20) -> ScalebarHit | None:
    """데이터바 영역에서 가장 긴 밝은 수평 막대를 찾는다."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"이미지는 2차원이어야 한다 (받은 차원: {img.ndim})")

    top = databar_top if databar_top is not None else int(img.shape[0] * 0.8)
    top = max(0, min(top, img.shape[0] - 1))
    region = img[top:, :]
    if region.size == 0:
        return None

    span = float(region.max() - region.min())
    if span <= 0:
        return None
    bright = region > (region.min() + 0.5 * span)

    max_width = int(img.shape[1] * MAX_BAR_WIDTH_RATIO)
    best: ScalebarHit | None = None
    best_length = 0
    for offset, row_mask in enumerate(bright):
        length, x0, x1 = _longest_bright_run(row_mask)
        if length < min_length_px or length > max_width:
            continue
        if length > best_length:
            best_length = length
            best = ScalebarHit(row=top + offset, x0=x0, x1=x1)
    return best


def scale_from_scalebar(length_px: float, length_nm: float) -> ScaleInfo:
    """검출된 막대 길이와 사용자가 입력한 실제 길이로 스케일을 만든다."""
    if length_px <= 0 or length_nm <= 0:
        raise ValueError(
            f"길이는 양수여야 한다: {length_px} px, {length_nm} nm"
        )
    return ScaleInfo(nm_per_px=length_nm / length_px, source="scalebar_auto")


def scale_from_two_points(p0, p1, length_nm: float) -> ScaleInfo:
    """사용자가 찍은 두 점 사이 거리와 실제 길이로 스케일을 만든다."""
    distance_px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    if distance_px <= 0 or length_nm <= 0:
        raise ValueError(
            f"길이는 양수여야 한다: {distance_px:.3f} px, {length_nm} nm"
        )
    return ScaleInfo(nm_per_px=length_nm / distance_px, source="manual")
