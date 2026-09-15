"""정답 갭 폭을 아는 합성 SEM 이미지 생성기.

실제 SEM 이미지로는 "측정값이 맞는가"를 확인할 수 없다. 참값을 모르기 때문이다.
여기서 만드는 이미지는 50% 문턱의 참값 위치가 정확히 갭 중심에서 +-gap_px/2가
되도록 erf 프로파일로 구성되어 있어, 측정 엔진의 정확도를 숫자로 고정할 수 있다.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def half_max_centre(row, i_metal: float = 200.0, i_gap: float = 40.0) -> float:
    """50% 문턱을 지나는 두 지점의 중점으로 갭 중심을 서브픽셀로 잡는다.

    np.argmin을 쓰면 안 된다. 갭 바닥은 erf 전이가 완전히 포화된 평탄부라서
    (30픽셀 갭, sigma=1.5에서 4픽셀이 모두 같은 최소값) argmin이 평탄부의 왼쪽
    끝을 돌려주고, 참값에서 1.5~2.2픽셀 어긋난다. 회전 정렬된 프로파일에서는
    행마다 샘플링 위상이 달라 그 왼쪽 끝이 4픽셀까지 흔들린다. 이 중점 추정은
    같은 조건에서 행 간 편차가 0.012픽셀이다.
    """
    mid = (i_metal + i_gap) / 2.0
    x = np.arange(row.size, dtype=float)
    lo = int(np.argmin(row))  # 평탄부 어딘가 — 좌우를 가르는 용도로만 쓴다
    left = np.interp(mid, row[:lo + 1][::-1], x[:lo + 1][::-1])
    right = np.interp(mid, row[lo:], x[lo:])
    return float((left + right) / 2.0)


def gap_center_x_at_row(row: int, *, width: int, height: int,
                        angle_deg: float) -> float:
    """주어진 행에서 갭 중심선이 지나는 x 좌표.

    갭 축 방향이 (sin a, cos a)이므로 중심선은 x = cx + (y - cy) * tan(a)이다.
    """
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return cx + (row - cy) * np.tan(np.radians(angle_deg))


def synth_gap_image(
    *,
    width: int = 512,
    height: int = 512,
    gap_nm: float = 50.0,
    nm_per_px: float = 1.0,
    angle_deg: float = 0.0,
    edge_sigma_px: float = 1.5,
    noise_sigma: float = 0.0,
    i_metal: float = 200.0,
    i_gap: float = 40.0,
    edge_bright: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """합성 SEM 이미지를 만든다.

    angle_deg는 엔진과 같은 정의를 쓴다: 갭 축이 이미지 세로축(+y)과 이루는 각도,
    반시계 방향이 양수.
    """
    gap_px = gap_nm / nm_per_px
    half = gap_px / 2.0
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    a = np.radians(angle_deg)

    y, x = np.mgrid[0:height, 0:width]
    # u = 갭 축에 수직인 방향(측정 방향)의 부호 있는 거리.
    u = (x - cx) * np.cos(a) - (y - cy) * np.sin(a)
    d = np.abs(u)

    s = max(float(edge_sigma_px), 1e-6)
    # d == half 에서 t == 0.5 가 되도록 맞춘 erf 전이. 이것이 참값의 정의다.
    t = 0.5 * (erf((d - half) / (np.sqrt(2.0) * s)) + 1.0)
    img = i_gap + (i_metal - i_gap) * t

    if edge_bright:
        img = img + edge_bright * np.exp(-((d - half - 1.5 * s) ** 2) / (2.0 * s**2))

    if noise_sigma:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, float(noise_sigma), img.shape)

    return np.ascontiguousarray(img, dtype=np.float64)
