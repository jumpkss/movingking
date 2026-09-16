"""라인 결과를 모아 ROI 하나의 통계와 경고를 만든다."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from ebl_gap.types import UNCERTAIN_STATUSES, LineResult, RoiResult, ScaleInfo

SHORT_RATIO_WARN = 0.05
UNCERTAIN_RATIO_WARN = 0.20
MIN_VALID_LINES = 10
CV_WARN = 0.20


def mark_outliers(lines: list[LineResult], *, mad_k: float = 3.5,
                  resolution_nm: float = 0.0) -> list[LineResult]:
    """valid 라인 중 중앙값에서 크게 벗어난 것을 outlier로 다시 라벨링한다.

    표준편차 대신 MAD를 쓰는 이유는, 이상치 자체가 표준편차를 부풀려서 자기 자신을
    정상으로 만들어 버리기 때문이다.

    편차 판정의 척도는 `max(MAD, 측정 분해능)`이다. 둘 중 하나만 쓰면 반쪽이 된다.
    MAD만 쓰면 값 대부분이 정확히 같을 때(노이즈 없는 이미지, 아주 깨끗한 실측)
    MAD가 0이 되어 척도가 사라지고, 분해능보다 작은 1픽셀 차이까지 이상치로
    배제한다. 반대로 그때 판정을 통째로 포기하면 295개가 40 nm이고 5개가 112 nm인
    상황 — 이상치 제거가 가장 필요한 바로 그 상황 — 에서 기능이 꺼진다. 둘 중 큰
    쪽을 척도로 쓰면 "측정이 구분할 수 없는 차이는 이상치가 아니다"와 "실제 산포보다
    크게 벗어나면 이상치다"를 동시에 만족한다.

    `resolution_nm`은 보통 `ScaleInfo.nm_per_px`다. 0이면 분해능을 모른다는 뜻이고,
    MAD도 0이면 판정 기준 자체가 없으므로 전부 남긴다.
    """
    widths = [ln.width_nm for ln in lines
              if ln.status == "valid" and ln.width_nm is not None]
    if len(widths) < 3:
        return list(lines)

    median = float(np.median(widths))
    mad = float(np.median(np.abs(np.asarray(widths) - median)))
    scale_nm = max(mad, float(resolution_nm))
    if scale_nm <= 0.0:
        return list(lines)

    limit = mad_k * scale_nm
    out: list[LineResult] = []
    for ln in lines:
        if ln.status == "valid" and ln.width_nm is not None \
                and abs(ln.width_nm - median) > limit:
            out.append(replace(
                ln,
                status="outlier",
                flags=frozenset(),
                reason=(f"중앙값 {median:.2f}nm에서 "
                        f"{abs(ln.width_nm - median):.2f}nm 벗어남 "
                        f"(> {mad_k:g} x 척도 {scale_nm:.2f}nm, "
                        f"MAD {mad:.2f}nm / 분해능 {resolution_nm:.2f}nm)"),
            ))
        else:
            out.append(ln)
    return out


def summarize(lines, *, scale: ScaleInfo, angle_deg: float,
              extra_warnings: tuple[str, ...] = ()) -> RoiResult:
    """라인 결과를 RoiResult로 집계하고 사용자가 볼 경고를 만든다."""
    lines = tuple(lines)
    valid = [ln for ln in lines if ln.status == "valid" and ln.width_nm is not None]
    widths = np.asarray([ln.width_nm for ln in valid], dtype=np.float64)

    n_valid = len(valid)
    n_short = sum(1 for ln in lines if ln.status == "short")
    n_uncertain = sum(1 for ln in lines if ln.status in UNCERTAIN_STATUSES)
    n_low_conf = sum(1 for ln in valid if "low_confidence" in ln.flags)
    n_total = n_valid + n_short + n_uncertain

    mean_nm = float(widths.mean()) if n_valid >= 1 else None
    std_nm = float(widths.std(ddof=1)) if n_valid >= 2 else None

    warnings: list[str] = []
    if n_total and n_short / n_total >= SHORT_RATIO_WARN:
        warnings.append(
            f"short 발생 구간 있음, 확인 필요 ({n_short}/{n_total} 라인, "
            f"{100 * n_short / n_total:.1f}%)"
        )
    if n_total and n_uncertain / n_total >= UNCERTAIN_RATIO_WARN:
        warnings.append(
            f"측정 신뢰도 낮음, ROI 재설정 권장 (판정보류 {n_uncertain}/{n_total} 라인, "
            f"{100 * n_uncertain / n_total:.1f}%)"
        )
    if n_valid < MIN_VALID_LINES:
        warnings.append(f"유효 라인 부족, 평균 신뢰 불가 ({n_valid} 라인)")
    if mean_nm and std_nm and mean_nm > 0 and std_nm / mean_nm > CV_WARN:
        warnings.append(
            f"갭 폭 편차 큼, 패턴 불균일 의심 (변동계수 {100 * std_nm / mean_nm:.1f}%)"
        )
    if scale.source != "fei_metadata":
        warnings.append(
            f"스케일 출처가 자동 메타데이터가 아님 ({scale.source}, "
            f"{scale.nm_per_px:.4f} nm/px)"
        )
    if n_low_conf:
        warnings.append(
            f"정밀도 주의: {n_low_conf} 라인이 10픽셀 미만. "
            "배율을 올려 다시 촬영하면 정밀도가 개선된다"
        )
    warnings.extend(extra_warnings)

    return RoiResult(
        mean_nm=mean_nm,
        std_nm=std_nm,
        n_valid=n_valid,
        n_short=n_short,
        n_uncertain=n_uncertain,
        n_low_confidence=n_low_conf,
        angle_deg=float(angle_deg),
        lines=lines,
        warnings=tuple(warnings),
        scale=scale,
    )


def representative_line(lines: Sequence[LineResult]) -> LineResult | None:
    """평균을 대표하는 라인 하나. 폭이 중앙값에 가장 가까운 valid 라인이다.

    첫 줄이 아니라 이쪽인 이유: 이 라인의 프로파일을 보고 사용자가 '평균이
    어디서 나왔나'를 판단한다. valid가 없으면 첫 줄이라도 돌려준다 — 전부
    short인 이미지에서 왜 short인지 볼 수단이 필요하기 때문이다.
    """
    if not lines:
        return None
    valid = [ln for ln in lines
             if ln.status == "valid" and ln.width_nm is not None]
    if not valid:
        return lines[0]
    widths_nm = sorted(ln.width_nm for ln in valid)
    median_nm = widths_nm[len(widths_nm) // 2]
    return min(valid, key=lambda ln: abs(ln.width_nm - median_nm))
