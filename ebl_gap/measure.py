"""ROI 하나를 측정하는 단일 진입점.

GUI, CLI, 노트북 어디서든 이 함수 하나만 부르면 된다. 여기 아래로는 Qt가 전혀
등장하지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ebl_gap.classify import classify_line
from ebl_gap.edges import analyze_profile
from ebl_gap.orientation import (
    InsufficientEdgesError,
    detect_base_angle_deg,
    estimate_angle_deg,
)
from ebl_gap.profile import (
    extract_profiles,
    measurement_runs_down,
    uv_extent,
)
from ebl_gap.stats import mark_outliers, summarize
from ebl_gap.types import LineResult, Roi, RoiResult, ScaleInfo

if TYPE_CHECKING:  # 런타임 import는 하지 않는다. loader가 tifffile/Pillow를
    # 끌고 오는데, ROI 측정만 하려는 호출자에게 그 값을 물릴 이유가 없다.
    from ebl_gap.loader import LoadedImage

#: 갭 축 각도가 이 절댓값을 넘으면 측정을 믿을 수 없는 것으로 본다. 추정값과
#: 사용자가 고정한 값에 똑같이 적용한다.
#: 스펙 4.2와 8절이 상정한 시야는 기울기 0~10도이고, 정확도 게이트도 그 범위에서만
#: 참값을 보증한다. 거기에 스펙 상한의 절반인 5도를 여유로 얹어 15도로 둔다.
#: 15도를 넘는 추정치는 둘 중 하나다 — 시료가 스펙이 상정한 촬영 조건 밖이거나,
#: 갭이 ROI 가장자리에 붙어 Theil-Sen 적합이 접혔거나(리뷰어 실측: -18.5도,
#: 참값 70.0 nm를 74.1 nm로, 표준편차 0.4 nm라 정밀해 보인다). 어느 쪽이든
#: 사용자가 ROI를 확인해야 하므로 경고가 맞는 대응이다. 이 값은 특정 테스트를
#: 통과시키려고 맞춘 것이 아니라 스펙이 적은 기울기 범위에서 끌어낸 것이다.
ANGLE_SANITY_DEG = 15.0

#: 도달 행 계산에서 부동소수점 먼지를 걷어내는 여유(px). `cos(radians(90))`은
#: 0이 아니라 6e-17이고, 그대로 올림하면 정확히 90도에서 한 행이 덤으로 붙어
#: 데이터바 바로 위의 멀쩡한 ROI가 거부된다. 실제 도달 범위에 1 nm 단위로
#: 영향을 줄 수 있는 크기가 아니다.
_REACH_EPS_PX = 1e-9


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


def _lowest_scanned_row(roi: Roi, angle_deg: float) -> int:
    """회전한 ROI가 실제로 훑는 가장 아래 행.

    `extract_profiles`는 ROI 상자를 그대로 읽는 것이 아니라 `angle_deg`만큼
    돌린 사각형을 훑는다. 측정 방향 오프셋 u는 y에 `-u·sin`으로, 갭 축 오프셋
    v는 `+v·cos`으로 들어간다. 두 항 중 큰 쪽은 기준선 `roi.y1`이 이미 품고
    있으므로, 상자 밖으로 더 내려가는 몫은 남은 항 하나다.
    `ceil`로 올림해 픽셀 한 칸도 넘겨주지 않는다.

    회전으로 오히려 줄어드는 몫은 빼지 않는다 — 데이터바 쪽으로는 넉넉하게
    보는 편이 안전하고, 그래야 0도에서 값이 정확히 `roi.y1`이 되어 회전 없는
    기존 동작과 한 치도 달라지지 않는다. 각도 90도(가로 갭의 정답)에서도
    정확히 `roi.y1`이다: 상자가 측정 방향과 함께 돌았으므로 도달 범위는 화면에
    그려진 상자 그대로다.

    표본 수와 방향 판단을 `profile` 모듈에서 받아 온다. 여기서 따로 계산하면
    데이터바 검사와 실제 표본 추출이 서로 다른 상자를 보게 된다.
    """
    angle_rad = math.radians(float(angle_deg))
    u_extent, v_extent = uv_extent(roi, angle_deg)
    if measurement_runs_down(angle_deg):
        swing = (v_extent / 2) * abs(math.cos(angle_rad))
    else:
        swing = (u_extent / 2) * abs(math.sin(angle_rad))
    return roi.y1 + math.ceil(swing - _REACH_EPS_PX)


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
        DatabarOverlapError: 회전까지 감안한 ROI가 데이터바 영역에 걸칠 때.
            스펙 4.1이 요구하는 거부이며, 화면이 아니라 여기 있어야 한다 —
            노트북에서 직접 부르는 경로도 같은 보호를 받아야 하기 때문이다.
            데이터바의 균일한 띠는 라인마다 short로 판정돼 날조된 이상 비율을
            만든다. 검사는 각도가 정해진 뒤에 한다(`_lowest_scanned_row` 참조).
    """
    extra_warnings: list[str] = []
    locked = angle_deg is not None

    # 기준 방향은 자동/고정 양쪽에서 한 번만 판별한다. 고정한 각도에도 기준이
    # 필요하다 — 타당성 검사가 절댓값이 아니라 기준 대비로 판정하기 때문이다.
    base_deg = detect_base_angle_deg(image, roi)

    if angle_deg is None:
        try:
            angle_deg, _, _ = estimate_angle_deg(image, roi, base_deg=base_deg,
                                                 **params.edge_kwargs)
        except InsufficientEdgesError as exc:
            angle_deg = base_deg
            extra_warnings.append(
                f"각도 자동 추정 실패, {base_deg:.0f}도로 측정함 ({exc})"
            )

    # 범위 검사는 각도의 출처를 가리지 않는다. 고정한 -70도는 추정한 -70도와
    # 똑같이 틀린 폭을 내므로, 고정을 검사 면제로 두면 사용자가 손으로 넣은
    # 오타만 조용히 통과한다. 권하는 행동이 다르므로 문구는 나눈다 — 추정이
    # 무너졌으면 ROI를 고쳐야 하고, 고정값이 이상하면 입력한 숫자를 고쳐야
    # 한다. 이미 고정한 사람에게 고정을 다시 권하면 따를 조언이 남지 않는다.
    #
    # 절댓값이 아니라 기준에서 벗어난 정도로 본다. 가로 갭의 정답인 90도를
    # 절댓값으로 재면 언제나 경고가 붙어 경고가 잡음이 된다. 문구에 판별 결과를
    # 넣는 이유는, 판별이 틀렸을 때 사용자가 그 사실을 알아야 각도를 고정할
    # 마음을 먹기 때문이다.
    if abs(angle_deg - base_deg) > ANGLE_SANITY_DEG:
        # 판별 결과를 부르는 이름은 결과 패널과 같은 술어에서 온다.
        orientation = "가로" if measurement_runs_down(base_deg) else "세로"
        judged = f"{orientation} 갭으로 판별했습니다(기준 {base_deg:.0f}도). "
        if locked:
            extra_warnings.append(
                f"{judged}고정한 각도가 {angle_deg:.1f}도입니다 — 의도한 값인지 "
                f"확인하세요"
            )
        else:
            extra_warnings.append(
                f"{judged}갭 축 각도가 {angle_deg:.1f}도로 추정됐습니다 — ROI가 "
                f"갭을 제대로 가로지르는지 확인하고, 필요하면 각도를 직접 "
                f"고정하세요"
            )

    # 데이터바 검사는 여기서 한다. 각도가 확정된 뒤라야 ROI가 실제로 훑는
    # 마지막 행을 알 수 있고, 엔진은 각도를 아는 첫 번째 자리다.
    if databar_top is not None:
        reach = _lowest_scanned_row(roi, angle_deg)
        if reach >= databar_top:
            tilt = ("" if reach == roi.y1 else
                    f"{angle_deg:.1f}도 기운 ROI가 {reach}행까지 훑습니다. ")
            raise DatabarOverlapError(
                f"{tilt}ROI가 데이터바 영역({databar_top}행 이하)을 "
                f"침범했습니다 — ROI를 데이터바 위쪽으로 다시 잡으세요"
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


def measure_loaded(loaded: "LoadedImage", roi: Roi, *,
                   angle_deg: float | None = None,
                   params: MeasureParams = MeasureParams()) -> RoiResult:
    """`load_image`가 돌려준 것을 그대로 받아 측정한다.

    `measure_roi`는 스케일과 데이터바 행을 따로 받는데, 그 둘은 이미
    `LoadedImage` 안에 있다. 손으로 옮겨 적게 두면 언젠가 `databar_top`이
    빠지고, 그러면 거부가 통째로 꺼져 데이터바의 균일한 띠가 날조된 short
    비율로 보고된다(리뷰어 실측 26.9%). 노트북에서는 이 함수를 쓴다.

    `measure_roi`의 서명은 건드리지 않는다 — 정확도 게이트가 그쪽을 쓴다.

    Raises:
        ValueError: 스케일이 확정되지 않은 이미지일 때. 길이 단위를 모르는 채
            측정하면 나오는 nm 값에 아무 뜻이 없다.
        DatabarOverlapError: ROI가 데이터바 영역에 걸칠 때.
    """
    scale = loaded.record.scale
    if scale is None:
        raise ValueError(
            f"{loaded.record.path.name}: 스케일이 확정되지 않았습니다 — "
            f"스케일 캘리브레이션으로 nm/px를 먼저 정하세요"
        )
    return measure_roi(loaded.pixels, roi, scale, angle_deg=angle_deg,
                       params=params, databar_top=loaded.databar_top)
