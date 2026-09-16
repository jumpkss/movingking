"""이미지 여러 장을 한 세션으로 묶고 dose-gap 관계를 집계한다."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ebl_gap.types import ImageRecord

#: 파일명에서 dose를 읽는 기본 패턴. pattern_320uC_01.tif -> 320.0
DEFAULT_DOSE_PATTERN = r"(?i)(\d+(?:\.\d+)?)\s*uc"

#: 픽셀 크기가 이 비율 이상 다르면 배율이 섞인 것으로 본다.
SCALE_MIX_TOLERANCE = 0.01


def parse_dose(filename: str, pattern: str = DEFAULT_DOSE_PATTERN) -> float | None:
    """파일명에서 dose 값을 읽는다. 찾지 못하면 None."""
    match = re.search(pattern, str(filename))
    if match is None:
        return None
    try:
        return float(match.group(1))
    except (IndexError, ValueError):
        return None


@dataclass(frozen=True)
class DosePoint:
    """dose-gap 곡선의 점 하나."""

    dose: float
    mean_nm: float
    std_nm: float | None
    n_valid: int
    n_short: int
    path: Path


@dataclass(frozen=True)
class ClosedDose:
    """갭이 전 구간에서 닫힌 dose. 측정된 갭 폭이 없다.

    `DosePoint`와 따로 두는 이유: 여기에는 잴 수 있는 갭이 없다. 같은 목록에
    `mean_nm=0.0`으로 섞어 넣으면 평균, 기울기, 오차 막대를 계산하는 모든
    소비자가 재지 않은 0을 측정값으로 받는다. 따로 두면 소비자는 이 목록을
    쓰겠다고 밝혀야 하고, 빠뜨렸을 때 나오는 것은 틀린 숫자가 아니라 예전
    화면이다.
    """

    dose: float
    n_short: int
    n_total: int
    path: Path


@dataclass
class Session:
    """한 번의 dose test에서 다루는 이미지들."""

    records: list[ImageRecord] = field(default_factory=list)

    def add(self, record: ImageRecord) -> ImageRecord:
        self.records.append(record)
        return record

    def dose_curve(self) -> list[DosePoint]:
        """dose가 있고 측정값이 나온 이미지만 모아 dose 오름차순으로 돌려준다.

        한 이미지에 ROI가 여러 개면 유효 라인 수로 가중 평균한다. 라인이 많은 ROI가
        더 믿을 만하기 때문이다.
        """
        points: list[DosePoint] = []
        for record in self.records:
            if record.dose is None:
                continue
            usable = [r for r in record.roi_results
                      if r.mean_nm is not None and r.n_valid > 0]
            if not usable:
                continue
            total_valid = sum(r.n_valid for r in usable)
            mean_nm = sum(r.mean_nm * r.n_valid for r in usable) / total_valid
            spreads = [r.std_nm for r in usable if r.std_nm is not None]
            points.append(DosePoint(
                dose=record.dose,
                mean_nm=mean_nm,
                std_nm=(sum(spreads) / len(spreads)) if spreads else None,
                n_valid=total_valid,
                n_short=sum(r.n_short for r in record.roi_results),
                path=record.path,
            ))
        return sorted(points, key=lambda p: p.dose)

    def closed_doses(self) -> list[ClosedDose]:
        """갭이 닫혀 측정값이 나오지 않은 dose를 dose 오름차순으로 돌려준다.

        "이 dose에서 갭이 닫힌다"가 dose test의 답이므로, 곡선에서 이 점이 빠지면
        답의 절반이 지워진다. 표와 CSV에는 남지만 사용자가 dose를 고르는 곳은
        곡선이다.

        "닫혔다"와 "못 쟀다"를 가른다. 유효 라인이 하나도 없고 short가 판정보류보다
        많을 때만 닫힌 것으로 본다. ROI를 엉뚱한 데 놓아 전부 no_edge가 난 것까지
        갭 0으로 찍으면, 화면이 측정하지 않은 결론을 대신 말하게 된다.
        """
        closed: list[ClosedDose] = []
        for record in self.records:
            if record.dose is None or not record.roi_results:
                continue
            if any(r.mean_nm is not None and r.n_valid > 0
                   for r in record.roi_results):
                continue
            n_short = sum(r.n_short for r in record.roi_results)
            n_uncertain = sum(r.n_uncertain for r in record.roi_results)
            if n_short <= n_uncertain:
                continue
            closed.append(ClosedDose(
                dose=record.dose,
                n_short=n_short,
                n_total=n_short + n_uncertain,
                path=record.path,
            ))
        return sorted(closed, key=lambda c: c.dose)

    def scale_warnings(self) -> list[str]:
        """세션 안에서 배율이 섞였는지 확인한다."""
        sizes = [r.scale.nm_per_px for r in self.records if r.scale is not None]
        if len(sizes) < 2:
            return []
        smallest, largest = min(sizes), max(sizes)
        if (largest - smallest) / smallest > SCALE_MIX_TOLERANCE:
            return [
                f"세션 안에 배율이 섞여 있다 ({smallest:.4f} ~ {largest:.4f} nm/px). "
                "dose 곡선을 해석할 때 주의할 것"
            ]
        return []
