"""1차원 밝기 프로파일에서 50% 문턱 서브픽셀 에지를 찾는다.

이 모듈은 이미지, ROI, 회전, 스케일을 전혀 모른다. 1차원 배열 하나만 받는다.
그 덕분에 측정 정확도를 다른 모든 요소와 분리해 검증할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class ProfileAnalysis:
    """프로파일 한 줄에서 뽑아낸 밝기 레벨과 에지 위치."""

    i_hi_left: float
    i_hi_right: float
    i_lo: float
    sigma_noise: float
    min_index: int
    left_px: float | None
    right_px: float | None
    n_cross_left: int
    n_cross_right: int
    threshold_fraction: float

    @property
    def threshold_left(self) -> float:
        """왼쪽 에지를 잡은 실제 밝기 문턱. _locate와 같은 식이어야 한다."""
        return self.i_lo + self.threshold_fraction * (self.i_hi_left - self.i_lo)

    @property
    def threshold_right(self) -> float:
        return self.i_lo + self.threshold_fraction * (self.i_hi_right - self.i_lo)

    @property
    def width_px(self) -> float | None:
        if self.left_px is None or self.right_px is None:
            return None
        return self.right_px - self.left_px

    @property
    def contrast(self) -> float:
        """갭 바닥과 어두운 쪽 전극의 밝기 차이."""
        return min(self.i_hi_left, self.i_hi_right) - self.i_lo


def _rising_crossing(p: np.ndarray, j_low: int, j_high: int,
                     threshold: float) -> float:
    """p[j_low] < threshold <= p[j_high] 인 인접 쌍에서 선형보간 위치를 구한다."""
    a = float(p[j_low])
    b = float(p[j_high])
    if abs(b - a) < _EPS:
        return float(j_low)
    frac = (threshold - a) / (b - a)
    return j_low + frac * (j_high - j_low)


def _count_rising(seg: np.ndarray, t_hi: float, t_lo: float) -> int:
    """히스테리시스를 적용해 상향 교차 횟수를 센다.

    t_lo 아래로 내려가야 다음 교차를 셀 수 있게 무장(arm)된다. 잡음이 문턱 근처에서
    여러 번 넘나드는 것을 다중 패턴으로 오판하지 않기 위한 장치다.
    """
    if seg.size == 0:
        return 0
    count = 0
    armed = float(seg[0]) <= t_lo
    for value in seg:
        v = float(value)
        if armed and v >= t_hi:
            count += 1
            armed = False
        elif not armed and v <= t_lo:
            armed = True
    return count


def _locate(p: np.ndarray, min_index: int, i_hi_left: float, i_hi_right: float,
            i_lo: float, threshold_fraction: float, hysteresis: float):
    """주어진 밝기 레벨로 좌/우 에지와 교차 횟수를 구한다."""
    t_left = i_lo + threshold_fraction * (i_hi_left - i_lo)
    t_right = i_lo + threshold_fraction * (i_hi_right - i_lo)

    left: float | None = None
    for j in range(min_index - 1, -1, -1):
        if p[j] >= t_left:
            left = _rising_crossing(p, j + 1, j, t_left)
            break

    right: float | None = None
    for j in range(min_index + 1, p.size):
        if p[j] >= t_right:
            right = _rising_crossing(p, j - 1, j, t_right)
            break

    band_left = hysteresis * (i_hi_left - i_lo)
    band_right = hysteresis * (i_hi_right - i_lo)
    n_left = _count_rising(p[: min_index + 1][::-1],
                           t_left + band_left, t_left - band_left)
    n_right = _count_rising(p[min_index:],
                            t_right + band_right, t_right - band_right)
    return left, right, n_left, n_right


def analyze_profile(
    profile,
    *,
    threshold_fraction: float = 0.5,
    flat_fraction: float = 0.2,
    center_fraction: float = 0.6,
    hysteresis: float = 0.1,
) -> ProfileAnalysis:
    """프로파일 한 줄을 분석해 밝기 레벨과 서브픽셀 에지 위치를 돌려준다."""
    p = np.asarray(profile, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"프로파일은 1차원이어야 한다 (받은 차원: {p.ndim})")
    n = p.size
    if n < 9:
        raise ValueError(f"프로파일이 너무 짧다: {n} 샘플 (최소 9)")

    # 전극 평탄부는 프로파일 양 끝에서 잡는다. 에지 근처에서 잡으면
    # edge-brightening 때문에 문턱이 위로 밀려 갭이 좁게 측정된다.
    k = max(2, int(round(n * flat_fraction)))
    i_hi_left = float(np.median(p[:k]))
    i_hi_right = float(np.median(p[-k:]))
    sigma_noise = float(0.5 * (np.std(p[:k]) + np.std(p[-k:])))

    margin = int(round(n * (1.0 - center_fraction) / 2.0))
    margin = min(margin, (n - 3) // 2)
    center = p[margin : n - margin]
    min_index = int(margin + np.argmin(center))

    # 1차: 국소 최소값을 갭 바닥으로 보고 갭 위치를 대략 잡는다.
    i_lo = float(np.min(center))

    # 대비가 없으면 에지를 찾을 수 없다.
    if i_lo >= min(i_hi_left, i_hi_right) - _EPS:
        return ProfileAnalysis(
            i_hi_left=i_hi_left,
            i_hi_right=i_hi_right,
            i_lo=i_lo,
            sigma_noise=sigma_noise,
            min_index=min_index,
            left_px=None,
            right_px=None,
            n_cross_left=0,
            n_cross_right=0,
            threshold_fraction=threshold_fraction,
        )

    left, right, n_left, n_right = _locate(
        p, min_index, i_hi_left, i_hi_right, i_lo, threshold_fraction, hysteresis
    )

    # 2차: 잡아낸 갭의 중앙 절반에서 바닥 밝기를 다시 구해 문턱을 보정한다.
    # 단순 분위수를 쓰면 좁은 갭에서 금속 밝기가 섞여 들어와 문턱이 통째로 틀어진다.
    if left is not None and right is not None:
        w = right - left
        lo_a = int(np.ceil(left + 0.25 * w))
        lo_b = int(np.floor(right - 0.25 * w))
        if lo_b >= lo_a:
            i_lo = float(np.median(p[lo_a : lo_b + 1]))
            left, right, n_left, n_right = _locate(
                p, min_index, i_hi_left, i_hi_right, i_lo,
                threshold_fraction, hysteresis,
            )

    return ProfileAnalysis(
        i_hi_left=i_hi_left,
        i_hi_right=i_hi_right,
        i_lo=i_lo,
        sigma_noise=sigma_noise,
        min_index=min_index,
        left_px=left,
        right_px=right,
        n_cross_left=n_left,
        n_cross_right=n_right,
        threshold_fraction=threshold_fraction,
    )
