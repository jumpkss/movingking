"""라인 결과를 모아 ROI 하나의 통계와 경고를 만든다."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ebl_gap.types import UNCERTAIN_STATUSES, LineResult, RoiResult, ScaleInfo

SHORT_RATIO_WARN = 0.05
UNCERTAIN_RATIO_WARN = 0.20
MIN_VALID_LINES = 10
CV_WARN = 0.20


def mark_outliers(lines: list[LineResult], *, mad_k: float = 3.5) -> list[LineResult]:
    """valid 라인 중 중앙값에서 크게 벗어난 것을 outlier로 다시 라벨링한다.

    표준편차 대신 MAD를 쓰는 이유는, 이상치 자체가 표준편차를 부풀려서 자기 자신을
    정상으로 만들어 버리기 때문이다. MAD가 0인 경우에도 limit=0이 되어 중앙값과
    다른 모든 값을 outlier로 표시하므로, 이는 올바른 동작이다.
    """
    widths = [ln.width_nm for ln in lines
              if ln.status == "valid" and ln.width_nm is not None]
    if len(widths) < 3:
        return list(lines)

    median = float(np.median(widths))
    mad = float(np.median(np.abs(np.asarray(widths) - median)))
    limit = mad_k * mad
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
                        f"(> {mad_k:g} x MAD {mad:.2f}nm)"),
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
