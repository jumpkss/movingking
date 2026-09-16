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
    n_uncertain: int
    path: Path

    @property
    def n_total(self) -> int:
        """이 dose에서 판정된 라인 전부. short 비율의 분모다.

        `RoiResult.n_total`과 같은 정의여야 한다. 곡선과 결과 패널이 다른
        분모를 쓰면 같은 한 장을 두고 두 화면이 다른 답을 준다 — 50 valid /
        3 short / 47 판정보류에서 패널은 3.0%로 조용한데 곡선은 5.66%로
        빨간 X를 붙인다. 사용자가 dose를 고르는 곳은 곡선이다.
        """
        return self.n_valid + self.n_short + self.n_uncertain


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
                n_uncertain=sum(r.n_uncertain for r in record.roi_results),
                path=record.path,
            ))
        return sorted(points, key=lambda p: p.dose)

    def closed_doses(self) -> list[ClosedDose]:
        """갭이 닫혀 측정값이 나오지 않은 dose를 dose 오름차순으로 돌려준다.

        "이 dose에서 갭이 닫힌다"가 dose test의 답이므로, 곡선에서 이 점이 빠지면
        답의 절반이 지워진다. 표와 CSV에는 남지만 사용자가 dose를 고르는 곳은
        곡선이다.

        유효 라인이 하나도 없고 short가 판정보류보다 많을 때만 여기 들어온다.
        이 규칙이 가르는 것은 "전 구간 short"와 "못 쟀다(판정보류뿐)"이지,
        "갭이 닫혔다"와 "ROI가 빗나갔다"가 아니다. 엔진은 픽셀만으로 그 둘을
        구별할 수 없다 — 평탄한 금속과 닫힌 갭은 둘 다 대비가 없고, 실측하면
        빗나간 ROI도 no_edge가 아니라 전 구간 short를 낸다(갭이 x=256인 합성
        이미지에서 ROI를 왼쪽 금속 위에 놓으면 301줄 전부 short다).

        그러므로 이 목록의 이름을 곧이곧대로 "갭이 닫힌 dose"로 읽으면 안 된다.
        소비자(리포트, dose 곡선)는 단정하지 말고 두 가능성을 함께 적어야 한다.
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

    def session_warnings(self) -> list[str]:
        """한 장만 봐서는 알 수 없고 세션 전체를 봐야 보이는 경고.

        한 dose에서 갭이 닫히는 것은 정상이고, 그것이 dose test의 답이다.
        그런데 **모든** dose가 전 구간 short인 것은 dose test로서 말이 되지
        않는다 — ROI가 패턴을 벗어났거나 스케일·문턱 설정이 틀렸을 가능성이
        훨씬 높다. `closed_doses()`가 한 장 단위로는 구별할 수 없는 것을,
        여기서는 "측정된 점이 하나도 없다"는 정황으로 한 번 말할 수 있다.

        틀렸을 때의 비용은 진짜로 전부 닫힌 시리즈에 한 줄이 더 붙는 것뿐이다.
        """
        if self.dose_curve() or not self.closed_doses():
            return []
        return [
            "세션의 모든 dose가 전 구간 short입니다 — dose test라면 나올 수 없는 "
            "결과이므로, ROI가 패턴을 벗어나지 않았는지와 스케일·문턱 설정을 "
            "먼저 확인하세요"
        ]

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
