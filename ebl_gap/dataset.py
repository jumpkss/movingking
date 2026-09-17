"""이미지 여러 장을 한 세션으로 묶고 dose-gap 관계를 집계한다."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ebl_gap.types import ImageRecord, RoiResult

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
    """dose-gap 곡선의 점 하나. 같은 dose로 찍은 여러 장이 여기 하나로 모인다."""

    dose: float
    mean_nm: float
    std_nm: float | None
    n_valid: int
    n_short: int
    n_uncertain: int
    #: 이 점에 들어간 이미지 전부. 갭이 나온 장과 전 구간 short인 장을 모두 담는다.
    #: `path` 하나로는 더 이상 진실을 적을 수 없다 — 반복 촬영이 한 점이 되므로.
    paths: tuple[Path, ...]
    #: 그중 갭이 하나도 나오지 않은(전 구간 short) 장 수. 같은 dose에서 한 장은
    #: 재지고 한 장은 닫혔다면 그 어긋남 자체가 사용자가 알아야 할 사실이다.
    n_closed_images: int

    @property
    def n_images(self) -> int:
        """이 점을 만든 이미지 장 수. `paths`에서 센다 — 둘이 갈라질 수 없다."""
        return len(self.paths)

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
    #: 이 dose로 찍은 이미지 전부. 반복 촬영이면 여러 장이다.
    paths: tuple[Path, ...]

    @property
    def n_images(self) -> int:
        return len(self.paths)


@dataclass(frozen=True)
class _ImageGap:
    """이미지 한 장을 한 줄로 요약한 것. dose로 묶기 전 단계다."""

    mean_nm: float | None
    std_nm: float | None
    n_valid: int
    n_short: int
    n_uncertain: int

    @property
    def is_closed(self) -> bool:
        """갭이 하나도 안 나왔고 short가 판정보류보다 많다 = 전 구간 short.

        `closed_doses()`와 같은 규칙이어야 한다. 갈라지면 한 장짜리 dose가
        곡선에서는 닫힘으로, 점의 장 수 세기에서는 아닌 것으로 잡힌다.
        """
        return self.mean_nm is None and self.n_short > self.n_uncertain


def _image_gap(record: ImageRecord) -> _ImageGap:
    """이미지 한 장의 ROI들을 유효 라인 수로 가중 평균한다.

    라인이 많은 ROI가 더 믿을 만하기 때문이다.
    """
    results: list[RoiResult] = list(record.roi_results)
    usable = [r for r in results if r.mean_nm is not None and r.n_valid > 0]
    n_short = sum(r.n_short for r in results)
    n_uncertain = sum(r.n_uncertain for r in results)
    if not usable:
        return _ImageGap(mean_nm=None, std_nm=None, n_valid=0,
                         n_short=n_short, n_uncertain=n_uncertain)
    n_valid = sum(r.n_valid for r in usable)
    spreads = [r.std_nm for r in usable if r.std_nm is not None]
    return _ImageGap(
        mean_nm=sum(r.mean_nm * r.n_valid for r in usable) / n_valid,
        std_nm=(sum(spreads) / len(spreads)) if spreads else None,
        n_valid=n_valid, n_short=n_short, n_uncertain=n_uncertain,
    )


def _pooled_std_nm(shots: list[_ImageGap]) -> float | None:
    """장 안의 산포와 장 사이의 산포를 합친 합동 표준편차.

    원래 라인들을 전부 모아 한 번에 계산한 것과 같은 값이다. 장 안의 산포만
    쓰면 장 사이가 크게 어긋나도 오차 막대가 작게 나온다 — dose를 고르는
    사람에게 없는 재현성을 있다고 말하는 셈이다. 60 nm와 64 nm로 찍힌 두 장
    (각 200줄, 장 안 1.0 nm)에서 장 안만 쓰면 1.0 nm, 합동으로는 2.24 nm다.
    """
    if len(shots) == 1:
        # 장이 하나면 합동 분산은 그 장의 분산 그대로다. 굳이 다시 계산해
        # 부동소수점 잡음을 얹지 않는다.
        return shots[0].std_nm
    total = sum(shot.n_valid for shot in shots)
    if total < 2:
        return None
    grand_nm = sum(shot.n_valid * shot.mean_nm for shot in shots) / total
    within = sum((shot.n_valid - 1) * (shot.std_nm or 0.0) ** 2
                 for shot in shots)
    between = sum(shot.n_valid * (shot.mean_nm - grand_nm) ** 2
                  for shot in shots)
    return math.sqrt((within + between) / (total - 1))


@dataclass
class Session:
    """한 번의 dose test에서 다루는 이미지들."""

    records: list[ImageRecord] = field(default_factory=list)

    def add(self, record: ImageRecord) -> ImageRecord:
        self.records.append(record)
        return record

    def _by_dose(self) -> dict[float, list[ImageRecord]]:
        """dose가 있고 측정을 시도한 이미지를 dose별로 모은다.

        아직 재지 않은 장(ROI 결과가 없다)은 빼둔다. 그 장은 이 dose에 아무것도
        보태지 않았으므로, 세면 "두 장을 평균했다"는 거짓이 된다.
        """
        groups: dict[float, list[ImageRecord]] = {}
        for record in self.records:
            if record.dose is None or not record.roi_results:
                continue
            groups.setdefault(record.dose, []).append(record)
        return groups

    def dose_curve(self) -> list[DosePoint]:
        """dose마다 점 하나를 dose 오름차순으로 돌려준다.

        같은 dose로 찍은 반복 촬영(`_001`, `_002`)은 한 점으로 묶는다. 두 점으로
        찍으면 곡선이 같은 x에서 두 번 꺾이고, 어느 쪽이 그 dose의 답인지 화면이
        말해 주지 않는다. 평균은 유효 라인 수 가중이고, 오차 막대는 장 안과 장
        사이의 산포를 합친 합동 표준편차다.

        한 장이라도 갭이 나왔으면 그 dose는 측정된 점이다. 같은 dose의 다른 장이
        전 구간 short였다면 그 장 수를 `n_closed_images`로 들고 간다 — 지우지
        않는다. 한 장은 재지고 한 장은 닫혔다는 것이 바로 사용자가 봐야 할 사실이다.
        """
        points: list[DosePoint] = []
        for dose, records in self._by_dose().items():
            shots: list[_ImageGap] = []
            n_short = n_uncertain = n_closed = 0
            for record in records:
                gap = _image_gap(record)
                n_short += gap.n_short
                n_uncertain += gap.n_uncertain
                if gap.mean_nm is None:
                    n_closed += 1 if gap.is_closed else 0
                else:
                    shots.append(gap)
            if not shots:
                continue
            total_valid = sum(shot.n_valid for shot in shots)
            points.append(DosePoint(
                dose=dose,
                mean_nm=sum(shot.mean_nm * shot.n_valid
                            for shot in shots) / total_valid,
                std_nm=_pooled_std_nm(shots),
                n_valid=total_valid,
                n_short=n_short,
                n_uncertain=n_uncertain,
                paths=tuple(record.path for record in records),
                n_closed_images=n_closed,
            ))
        return sorted(points, key=lambda p: p.dose)

    def closed_doses(self) -> list[ClosedDose]:
        """갭이 닫혀 측정값이 나오지 않은 dose를 dose 오름차순으로 돌려준다.

        "이 dose에서 갭이 닫힌다"가 dose test의 답이므로, 곡선에서 이 점이 빠지면
        답의 절반이 지워진다. 표와 CSV에는 남지만 사용자가 dose를 고르는 곳은
        곡선이다.

        같은 dose의 반복 촬영은 한 점으로 묶는다. **그중 한 장이라도 갭이 나왔으면
        이 목록에 넣지 않는다** — 그 dose는 측정된 점이고, 닫힌 장이 몇 장인지는
        `DosePoint.n_closed_images`가 들고 간다.

        유효 라인이 하나도 없고, 적어도 한 장이 short가 판정보류보다 많을 때만
        여기 들어온다. 이 규칙이 가르는 것은 "전 구간 short"와 "못 쟀다
        (판정보류뿐)"이지, "갭이 닫혔다"와 "ROI가 빗나갔다"가 아니다. 엔진은
        픽셀만으로 그 둘을 구별할 수 없다 — 평탄한 금속과 닫힌 갭은 둘 다 대비가
        없고, 실측하면 빗나간 ROI도 no_edge가 아니라 전 구간 short를 낸다(갭이
        x=256인 합성 이미지에서 ROI를 왼쪽 금속 위에 놓으면 301줄 전부 short다).

        그러므로 이 목록의 이름을 곧이곧대로 "갭이 닫힌 dose"로 읽으면 안 된다.
        소비자(리포트, dose 곡선)는 단정하지 말고 두 가능성을 함께 적어야 한다.
        """
        closed: list[ClosedDose] = []
        for dose, records in self._by_dose().items():
            gaps = [_image_gap(record) for record in records]
            if any(gap.mean_nm is not None for gap in gaps):
                continue
            if not any(gap.is_closed for gap in gaps):
                continue
            n_short = sum(gap.n_short for gap in gaps)
            n_uncertain = sum(gap.n_uncertain for gap in gaps)
            closed.append(ClosedDose(
                dose=dose,
                n_short=n_short,
                n_total=n_short + n_uncertain,
                paths=tuple(record.path for record in records),
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
