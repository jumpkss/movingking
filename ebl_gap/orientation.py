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


def detect_base_angle_deg(image, roi: Roi) -> float:
    """갭이 세로(0도)인지 가로(90도)인지 판별한다.

    갭은 어둡고 전극은 밝다. 세로 갭이면 열 평균이 깊게 파이고 행 평균은
    평탄하다. 가로 갭이면 반대다. 대비가 더 큰 쪽이 측정 방향이다.

    기울기까지 여기서 재지 않는다. 방향만 고르고, 그 방향 안에서의 미세
    기울기는 Theil-Sen이 맡는다 — 두 가지를 한 추정에 섞으면 둘 다 나빠진다.

    동률이면 0도를 고른다. 정사각 ROI에 대각 갭을 놓은 경우가 그런데, 어느
    쪽도 맞지 않는 상황이라 임의 선택이다. 임의이되 고정이어야 한다 — 같은
    이미지를 두 번 재면서 측정 방향이 바뀌면 안 된다.
    """
    profiles = extract_profiles(image, roi, 0.0)
    # numpy 2.x에는 ndarray.ptp() 메서드가 없다. np.ptp 함수를 쓴다.
    contrast_x = float(np.ptp(profiles.mean(axis=0)))  # 가로 방향 변화 = 세로 갭
    contrast_y = float(np.ptp(profiles.mean(axis=1)))  # 세로 방향 변화 = 가로 갭
    return 90.0 if contrast_y > contrast_x else 0.0


def estimate_angle_deg(image, roi: Roi, *, min_rows: int = 5,
                       base_deg: float | None = None,
                       **profile_kwargs) -> tuple[float, int, float]:
    """갭 축 각도를 Theil-Sen 추정으로 구한다.

    기준 방향(`base_deg`)으로 뽑은 프로파일에서 행마다 좌/우 에지 위치를 구하면,
    에지 위치는 행 번호의 선형 함수다. 그 기울기가 곧 tan(theta)이므로 한 번의
    패스로 충분하다. 돌려주는 각도는 `base_deg + 기울기`다.

    기준을 먼저 고르는 이유는 이 추정이 ±45도 안의 미세 기울기만 잡을 수 있기
    때문이다. 가로 갭(기준 90도)을 세로 기준으로 재면 31.5도 같은 값이 나오고,
    58도 틀린 각도는 폭을 1/cos만큼 부풀린다 — 사용자 제보에서 참값 70 nm가
    137.39 nm가 된 경로다.

    최소자승 대신 Theil-Sen을 쓰는 이유는 에지 검출이 몇 줄 실패하거나 엉뚱한 값을
    내놓아도 추정치가 끌려가지 않기 때문이다.

    Args:
        base_deg: 기준 방향. None이면 `detect_base_angle_deg`로 정한다. 호출자가
            이미 기준을 알고 있으면(예: `measure_roi`의 타당성 경고) 넘겨서
            판별이 두 번 일어나지 않게 한다.

    Returns:
        (각도(도), 추정에 쓰인 행 수, 기준 각도(도))
    """
    if base_deg is None:
        base_deg = detect_base_angle_deg(image, roi)
    base_deg = float(base_deg)
    # 측정 경로와 별개로 한 번 더 뽑는다. Task 19가 합치지 않기로 판정했다.
    # 여기에는 along_average를 넘기지 않는다 — 각도 추정은 언제나 평균 없는
    # 원본 해상도 프로파일에서 한다. 이것은 실수가 아니라 약속이다: 갭 축 방향
    # 이동평균은 기울어진 갭의 에지 위치를 행 방향으로 번지게 해서 바로 지금
    # 재려는 그 기울기를 흐린다. 그래서 사용자가 이동평균을 올려도 측정만
    # 매끈해지고 각도 추정은 그대로다.
    #
    # 나중에 "중복 호출"을 보고 measure_roi와 배열 하나를 공유하게 고치면,
    # along_average가 1이 아닌 순간 각도 추정이 조용히 바뀐다. 테스트는 전부
    # 기본값(1)으로 돌기 때문에 초록인 채로 바뀐다. 합치려면 여기에 반드시
    # along_average=1을 명시해서 넘겨라.
    profiles = extract_profiles(image, roi, base_deg)

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
    tilt_deg = float(np.degrees(np.arctan(0.5 * (slope_left + slope_right))))
    return base_deg + tilt_deg, len(rows), base_deg
