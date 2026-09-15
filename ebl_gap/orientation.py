"""ROI 안의 갭이 얼마나 기울어져 있는지 추정한다.

기울어진 갭을 수평 스캔라인으로 재면 폭이 1/cos(theta)만큼 과대평가된다. 5도에서
0.4%, 10도에서 1.5%다. 100 nm 이하를 재는 상황에서 1.5%는 무시할 수 없다.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import theilslopes

from ebl_gap.edges import analyze_profile
from ebl_gap.profile import extract_profiles
from ebl_gap.types import Roi


class InsufficientEdgesError(RuntimeError):
    """각도를 추정할 만큼 에지를 찾지 못했을 때."""


def estimate_angle_deg(image, roi: Roi, *, min_rows: int = 5,
                       **profile_kwargs) -> tuple[float, int]:
    """갭 축 각도를 Theil-Sen 추정으로 구한다.

    각도 0으로 뽑은 프로파일에서 행마다 좌/우 에지 위치를 구하면, 에지 위치는 행
    번호의 선형 함수다. 그 기울기가 곧 tan(theta)이므로 한 번의 패스로 충분하다.

    최소자승 대신 Theil-Sen을 쓰는 이유는 에지 검출이 몇 줄 실패하거나 엉뚱한 값을
    내놓아도 추정치가 끌려가지 않기 때문이다.

    Returns:
        (각도(도), 추정에 쓰인 행 수)
    """
    profiles = extract_profiles(image, roi, 0.0)

    rows: list[int] = []
    lefts: list[float] = []
    rights: list[float] = []
    for index, line in enumerate(profiles):
        try:
            analysis = analyze_profile(line, **profile_kwargs)
        except ValueError:
            continue
        if analysis.left_px is None or analysis.right_px is None:
            continue
        rows.append(index)
        lefts.append(analysis.left_px)
        rights.append(analysis.right_px)

    if len(rows) < min_rows:
        raise InsufficientEdgesError(
            f"각도 추정에 쓸 에지가 부족하다: {len(rows)}행 (최소 {min_rows}행)"
        )

    row_array = np.asarray(rows, dtype=np.float64)
    slope_left = float(theilslopes(np.asarray(lefts), row_array)[0])
    slope_right = float(theilslopes(np.asarray(rights), row_array)[0])

    # 각도를 각각 구해 평균내는 대신 기울기를 평균낸 뒤 한 번만 arctan을 취한다.
    angle_deg = float(np.degrees(np.arctan(0.5 * (slope_left + slope_right))))
    return angle_deg, len(rows)
