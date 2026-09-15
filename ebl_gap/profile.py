"""ROI를 갭 축에 맞춰 회전 정렬하고 스캔라인 프로파일을 뽑는다.

좌표 약속 (프로젝트 전체 공통):
    angle_deg는 갭 축이 이미지 세로축(+y, 아래 방향)과 이루는 각도이고 반시계
    방향이 양수다. 갭 축 단위벡터는 (sin a, cos a), 측정 방향 단위벡터는
    (cos a, -sin a)이다.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter1d

from ebl_gap.types import Roi


def extract_profiles(image, roi: Roi, angle_deg: float, *,
                     along_average: int = 1) -> np.ndarray:
    """ROI를 -angle_deg 회전시켜 정렬한 (height, width) 프로파일 배열을 만든다.

    반환 배열의 각 행이 갭 축에 수직인 방향의 밝기 프로파일이다. 행 수는 ROI 세로
    픽셀 수와 같다 — 이미지 최대 해상도를 그대로 쓴다는 뜻이다.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"이미지는 2차원이어야 한다 (받은 차원: {img.ndim})")
    if along_average < 1:
        raise ValueError(f"along_average는 1 이상이어야 한다: {along_average}")

    a = np.radians(float(angle_deg))
    w, h = roi.width, roi.height

    u = np.arange(w, dtype=np.float64) - (w - 1) / 2.0  # 측정 방향 오프셋
    v = np.arange(h, dtype=np.float64) - (h - 1) / 2.0  # 갭 축 방향 오프셋
    uu, vv = np.meshgrid(u, v)

    xx = roi.cx + uu * np.cos(a) + vv * np.sin(a)
    yy = roi.cy - uu * np.sin(a) + vv * np.cos(a)

    # mode="nearest": ROI가 이미지 경계를 살짝 벗어나도 예외 대신 가장자리 값을 쓴다.
    profiles = map_coordinates(img, [yy, xx], order=1, mode="nearest")

    if along_average > 1:
        profiles = uniform_filter1d(profiles, size=along_average, axis=0,
                                    mode="nearest")
    return np.ascontiguousarray(profiles, dtype=np.float64)
