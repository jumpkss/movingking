"""ROI 하나를 측정하는 단일 진입점.

GUI, CLI, 노트북 어디서든 이 함수 하나만 부르면 된다. 여기 아래로는 Qt가 전혀
등장하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from ebl_gap.classify import classify_line
from ebl_gap.edges import analyze_profile
from ebl_gap.orientation import InsufficientEdgesError, estimate_angle_deg
from ebl_gap.profile import extract_profiles
from ebl_gap.stats import mark_outliers, summarize
from ebl_gap.types import LineResult, Roi, RoiResult, ScaleInfo


@dataclass(frozen=True)
class MeasureParams:
    """측정에 쓰이는 모든 조절값. GUI가 그대로 노출한다."""

    threshold_fraction: float = 0.5
    flat_fraction: float = 0.2
    center_fraction: float = 0.6
    hysteresis: float = 0.1
    along_average: int = 1
    contrast_k: float = 5.0
    min_width_px: float = 3.0
    low_confidence_px: float = 10.0
    mad_k: float = 3.5

    @property
    def edge_kwargs(self) -> dict:
        return dict(
            threshold_fraction=self.threshold_fraction,
            flat_fraction=self.flat_fraction,
            center_fraction=self.center_fraction,
            hysteresis=self.hysteresis,
        )

    @property
    def classify_kwargs(self) -> dict:
        return dict(
            contrast_k=self.contrast_k,
            min_width_px=self.min_width_px,
            low_confidence_px=self.low_confidence_px,
        )


def measure_roi(
    image,
    roi: Roi,
    scale: ScaleInfo,
    *,
    angle_deg: float | None = None,
    params: MeasureParams = MeasureParams(),
) -> RoiResult:
    """ROI 안의 갭 폭을 모든 스캔라인에서 측정한다.

    Args:
        image: 2차원 밝기 배열.
        roi: 측정할 사각 영역.
        scale: 픽셀 크기와 그 출처.
        angle_deg: None이면 자동 추정한다. 값을 주면 그대로 쓴다.
        params: 측정 조절값.
    """
    extra_warnings: list[str] = []

    if angle_deg is None:
        try:
            angle_deg, _ = estimate_angle_deg(image, roi, **params.edge_kwargs)
        except InsufficientEdgesError as exc:
            angle_deg = 0.0
            extra_warnings.append(f"각도 자동 추정 실패, 0도로 측정함 ({exc})")

    profiles = extract_profiles(image, roi, angle_deg,
                                along_average=params.along_average)

    lines: list[LineResult] = []
    for row, profile in enumerate(profiles):
        try:
            analysis = analyze_profile(profile, **params.edge_kwargs)
        except ValueError as exc:
            lines.append(LineResult(row=row, left_px=None, right_px=None,
                                    width_px=None, width_nm=None,
                                    status="no_edge", flags=frozenset(),
                                    reason=str(exc)))
            continue

        status, flags, reason = classify_line(analysis, **params.classify_kwargs)
        width_px = analysis.width_px
        width_nm = None if width_px is None else width_px * scale.nm_per_px
        lines.append(LineResult(
            row=row,
            left_px=analysis.left_px,
            right_px=analysis.right_px,
            width_px=width_px,
            width_nm=width_nm,
            status=status,
            flags=flags,
            reason=reason,
        ))

    lines = mark_outliers(lines, mad_k=params.mad_k)
    return summarize(lines, scale=scale, angle_deg=angle_deg,
                     extra_warnings=tuple(extra_warnings))
