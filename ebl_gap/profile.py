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


def measurement_runs_down(angle_deg: float) -> bool:
    """측정 방향이 가로보다 세로에 가까운가.

    측정 방향 단위벡터는 (cos a, -sin a)이므로 |cos a| < |sin a|이면 세로 쪽이다.
    각도 45도(동률)는 가로 측정으로 친다 — `detect_base_angle_deg`의 동률 처리와
    같은 쪽이다.

    `uv_extent`와 `measure._lowest_scanned_row`가 이 판단을 함께 쓴다. 판단이 두
    군데로 갈라지면 데이터바 검사와 실제 표본 추출이 서로 다른 상자를 본다.
    """
    a_rad = np.radians(float(angle_deg))
    return abs(np.cos(a_rad)) < abs(np.sin(a_rad))


def uv_extent(roi: Roi, angle_deg: float) -> tuple[int, int]:
    """(측정 방향 표본 수, 갭 축 방향 표본 수).

    ROI 상자는 화면에 그려진 그대로이고, 측정 방향이 바뀌면 상자도 같이 돈다.
    측정 방향이 세로에 가까우면 ROI의 세로 길이가 측정 범위가 되어야 한다.
    이것을 안 바꾸면 가로로 납작한 ROI가 각도 90도에서 세로로 ROI 밖까지
    훑는다 — 상자는 화면에 그려진 그대로인데 표본은 딴 데서 온다.

    `extract_profiles`와 `aligned_to_image`가 이 한 자리에서 같은 값을 받는다.
    둘이 갈라지면 오버레이의 초록 에지가 실제 잰 자리와 다른 곳에 그려진다.
    """
    if measurement_runs_down(angle_deg):
        return roi.height, roi.width
    return roi.width, roi.height


def extract_profiles(image, roi: Roi, angle_deg: float, *,
                     along_average: int = 1) -> np.ndarray:
    """ROI를 -angle_deg 회전시켜 정렬한 (갭 축, 측정 방향) 프로파일 배열을 만든다.

    반환 배열의 각 행이 갭 축에 수직인 방향의 밝기 프로파일이다. 행 수는 갭 축
    방향의 ROI 픽셀 수와 같다 — 이미지 최대 해상도를 그대로 쓴다는 뜻이다.

    측정 방향이 세로에 가까우면(각도 45도 초과) ROI의 가로/세로가 맞바뀐다.
    `uv_extent` 참조.
    """
    img = np.asarray(image, dtype=np.float64)
    # size == 0을 따로 막는다. ndim만 보면 (0, 0) 배열이 통과하고,
    # map_coordinates가 예외도 NaN도 아닌 초기화되지 않은 메모리를 돌려준다.
    # 계측 툴에서 조용한 데이터 오염은 예외보다 나쁘다.
    if img.ndim != 2 or img.size == 0:
        raise ValueError(
            f"이미지는 비어 있지 않은 2차원 배열이어야 한다 "
            f"(차원 {img.ndim}, 원소 수 {img.size})"
        )
    if along_average < 1:
        raise ValueError(f"along_average는 1 이상이어야 한다: {along_average}")

    a_rad = np.radians(float(angle_deg))
    w, h = uv_extent(roi, angle_deg)

    u = np.arange(w, dtype=np.float64) - (w - 1) / 2.0  # 측정 방향 오프셋(px)
    v = np.arange(h, dtype=np.float64) - (h - 1) / 2.0  # 갭 축 방향 오프셋(px)
    uu, vv = np.meshgrid(u, v)

    xx = roi.cx + uu * np.cos(a_rad) + vv * np.sin(a_rad)
    yy = roi.cy - uu * np.sin(a_rad) + vv * np.cos(a_rad)

    # mode="nearest": ROI가 이미지 경계를 살짝 벗어나도 예외 대신 가장자리 값을 쓴다.
    profiles = map_coordinates(img, [yy, xx], order=1, mode="nearest")

    if along_average > 1:
        profiles = uniform_filter1d(profiles, size=along_average, axis=0,
                                    mode="nearest")
    return np.ascontiguousarray(profiles, dtype=np.float64)


def aligned_to_image(roi: Roi, angle_deg: float, u_px: float,
                     v_px: float) -> tuple[float, float]:
    """정렬 좌표계의 (열, 행)을 원본 이미지 좌표 (x, y)로 되돌린다.

    extract_profiles가 쓰는 변환과 반드시 같은 식이어야 한다. 오버레이에 에지를
    그리려면 이 역변환이 필요하다.
    """
    a_rad = np.radians(float(angle_deg))
    w, h = uv_extent(roi, angle_deg)
    u = float(u_px) - (w - 1) / 2.0
    v = float(v_px) - (h - 1) / 2.0
    x = roi.cx + u * np.cos(a_rad) + v * np.sin(a_rad)
    y = roi.cy - u * np.sin(a_rad) + v * np.cos(a_rad)
    return float(x), float(y)
