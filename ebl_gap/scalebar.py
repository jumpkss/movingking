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

#: 아래쪽 띠를 데이터바로 보려면 스캔 영역과 밝기가 이만큼은 달라야 한다.
#: 이미지 전체 밝기 범위에 대한 비율이다. 0.25는 "갭과 금속의 차이만큼 뚜렷한
#: 차이"에 해당한다 — 그보다 옅은 차이는 시료의 명암으로 본다.
DATABAR_CONTRAST_RATIO = 0.25
#: 데이터바로 보기 위한 최소 두께(행). 한두 줄짜리 띠는 스캔 아티팩트다.
DATABAR_MIN_ROWS = 5
#: 이미지 높이의 이 비율을 넘는 띠는 데이터바가 아니라 시료의 일부로 본다.
DATABAR_MAX_HEIGHT_RATIO = 0.4


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


def detect_databar_top(image) -> int | None:
    """메타데이터 없이 아래쪽 데이터바 띠가 시작되는 행을 추정한다.

    `databar_top_row`는 FEI의 ResolutionY가 있어야 답을 낸다. PNG 크롭과 비 FEI
    TIFF — 즉 스케일바 대체 경로로 몰리는 바로 그 집단 — 은 그 값이 없어서
    `measure_roi`가 ROI 침범을 거부할 수 없다. 여기서 찾은 값은 거부 근거로
    쓰지 않고 사용자에게 알리는 데만 쓴다. 휴리스틱으로 측정을 막으면, 아래쪽이
    어두운 멀쩡한 시료를 영영 못 재게 된다.

    FEI 데이터바는 어둡고 Zeiss/Hitachi 계열은 밝다. 어느 쪽이든 스캔 영역과
    밝기가 뚜렷이 다른 균일한 가로 띠이므로 부호를 가리지 않고 찾는다.

    Returns:
        띠가 시작되는 행 번호. 찾지 못하면 None.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"이미지는 2차원이어야 한다 (받은 차원: {img.ndim})")
    height = img.shape[0]
    if img.size == 0 or height < 2 * DATABAR_MIN_ROWS:
        return None

    span = float(img.max() - img.min())
    if span <= 0:
        return None

    row_mean = img.mean(axis=1)
    # 기준은 위쪽 절반의 중앙값이다. 데이터바는 아래쪽에만 붙으므로 위쪽 절반은
    # 스캔 영역이라고 보아도 된다. 평균 대신 중앙값을 쓰는 것은 밝은 패턴 몇
    # 줄에 기준이 끌려가지 않게 하기 위해서다.
    reference = float(np.median(row_mean[: height // 2]))
    limit = DATABAR_CONTRAST_RATIO * span

    top = height
    while top > 0 and abs(row_mean[top - 1] - reference) > limit:
        top -= 1

    band_rows = height - top
    if band_rows < DATABAR_MIN_ROWS:
        return None
    if band_rows > DATABAR_MAX_HEIGHT_RATIO * height:
        return None
    return top
