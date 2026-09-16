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

#: 자동 추정된 갭 축 각도가 이 절댓값을 넘으면 적합이 무너진 것으로 본다.
#: 스펙 4.2와 8절이 상정한 시야는 기울기 0~10도이고, 정확도 게이트도 그 범위에서만
#: 참값을 보증한다. 거기에 스펙 상한의 절반인 5도를 여유로 얹어 15도로 둔다.
#: 15도를 넘는 추정치는 둘 중 하나다 — 시료가 스펙이 상정한 촬영 조건 밖이거나,
#: 갭이 ROI 가장자리에 붙어 Theil-Sen 적합이 접혔거나(리뷰어 실측: -18.5도,
#: 참값 70.0 nm를 74.1 nm로, 표준편차 0.4 nm라 정밀해 보인다). 어느 쪽이든
#: 사용자가 ROI를 확인해야 하므로 경고가 맞는 대응이다. 이 값은 특정 테스트를
#: 통과시키려고 맞춘 것이 아니라 스펙이 적은 기울기 범위에서 끌어낸 것이다.
ANGLE_SANITY_DEG = 15.0


class DatabarOverlapError(ValueError):
    """ROI가 데이터바 영역을 침범했을 때.

    ValueError를 상속하는 이유는 노트북에서 `except ValueError`로 감싸 둔 코드가
    그대로 잡을 수 있게 하기 위해서다. GUI는 이 타입만 골라 잡는다.
    """


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
    databar_top: int | None = None,
) -> RoiResult:
    """ROI 안의 갭 폭을 모든 스캔라인에서 측정한다.

    Args:
        image: 2차원 밝기 배열.
        roi: 측정할 사각 영역.
        scale: 픽셀 크기와 그 출처.
        angle_deg: None이면 자동 추정한다. 값을 주면 그대로 쓴다.
        params: 측정 조절값.
        databar_top: 데이터바가 시작되는 행. None이면 위치를 모른다는 뜻이고,
            그때는 거부할 수 없다.

    Raises:
        DatabarOverlapError: ROI가 데이터바 영역에 걸칠 때. 스펙 4.1이 요구하는
            거부이며, 화면이 아니라 여기 있어야 한다 — 노트북에서 직접 부르는
            경로도 같은 보호를 받아야 하기 때문이다. 데이터바의 균일한 띠는
            라인마다 short로 판정돼 날조된 이상 비율을 만든다.
    """
    if databar_top is not None and roi.y1 >= databar_top:
        raise DatabarOverlapError(
            f"ROI가 데이터바 영역({databar_top}행 이하)을 침범했습니다 — "
            f"ROI를 데이터바 위쪽으로 다시 잡으세요"
        )

    extra_warnings: list[str] = []

    if angle_deg is None:
        try:
            angle_deg, _ = estimate_angle_deg(image, roi, **params.edge_kwargs)
        except InsufficientEdgesError as exc:
            angle_deg = 0.0
            extra_warnings.append(f"각도 자동 추정 실패, 0도로 측정함 ({exc})")
        else:
            # 사용자가 직접 고정한 각도에는 이 검사를 하지 않는다. 고정은 이
            # 경고가 권하는 바로 그 행동이므로, 고정한 값에 다시 경고를 달면
            # 따를 수 있는 조언이 남지 않는다.
            if abs(angle_deg) > ANGLE_SANITY_DEG:
                extra_warnings.append(
                    f"갭 축 각도가 {angle_deg:.1f}도로 추정됐습니다 — ROI가 갭을 "
                    f"제대로 가로지르는지 확인하고, 필요하면 각도를 직접 고정하세요"
                )

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

    lines = mark_outliers(lines, mad_k=params.mad_k,
                          resolution_nm=scale.nm_per_px)
    return summarize(lines, scale=scale, angle_deg=angle_deg,
                     extra_warnings=tuple(extra_warnings))
