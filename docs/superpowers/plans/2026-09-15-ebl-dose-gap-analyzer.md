# EBL Dose Test 갭 분석 툴 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SEM 이미지에서 드래그한 구간의 S/D 패턴 갭 폭을 서브픽셀 정밀도로 측정하고, short/미형성 구간을 분리 보고하며, dose–gap 곡선까지 뽑는 PySide6 데스크톱 툴을 만든다.

**Architecture:** Qt에 전혀 의존하지 않는 순수 파이썬 측정 엔진(`ebl_gap`)과 그 위에 얹는 얇은 GUI(`ebl_gap_gui`)로 분리한다. 엔진은 1차원 프로파일 연산(`edges`), 2차원 리샘플링(`profile`), 각도 추정(`orientation`), 판정(`classify`), 통계(`stats`)로 쪼개져 각각 단독 테스트된다. 정확도는 정답 갭 폭을 아는 합성 SEM 이미지로 회귀 테스트에 고정한다.

**Tech Stack:** Python 3.10+, numpy, scipy, tifffile, Pillow, PySide6, pyqtgraph, pytest

**Spec:** `docs/superpowers/specs/2026-09-15-ebl-dose-gap-analyzer-design.md`

## Global Constraints

- Python `>=3.10` (`X | None` 타입 문법을 쓴다)
- `ebl_gap/` 아래 어떤 모듈도 `PySide6`, `pyqtgraph`, `ebl_gap_gui`를 import하지 않는다. 이것이 깨지면 엔진을 화면 없이 테스트할 수 없다.
- 엔진 런타임 의존성은 `numpy`, `scipy`, `tifffile`, `Pillow` 네 개로 한정한다.
- 길이 단위는 엔진 내부에서 항상 **픽셀(px)** 또는 **나노미터(nm)** 이며, **공개 인터페이스·반환값·dataclass 필드·단위가 변환된 값**의 이름에 단위를 붙인다 (`width_px`, `width_nm`, `nm_per_px`). 미터 단위는 `metadata.py`가 읽는 즉시 nm로 바꾼다. 이 규칙의 목적은 nm/px/m가 만나는 **경계**에서 혼동을 막는 것이므로, 하나의 수식 안에서만 살고 전부 픽셀인 것이 자명한 지역 변수(회전 커널의 `u`, `v`, `xx`, `yy` 등)에는 적용하지 않는다 — 그런 곳에 접미사를 붙이면 수식만 읽기 어려워지고 잡아내는 버그는 없다.
- **각도는 예외 없이** 변수명에 `_deg` 또는 `_rad`를 붙인다. 지역 변수도 포함한다. 도/라디안 혼동은 조용히 틀린 값을 내는 실제 버그 유형이고, 이름 한 글자로 막을 수 있다.
- `angle_deg`의 정의: **갭이 뻗어나가는 축이 이미지 세로축(+y, 아래 방향)과 이루는 각도**, 반시계 방향이 양수. 0이면 갭이 정확히 세로로 뻗고 측정 방향은 가로다. 모든 모듈이 이 정의를 공유한다.
- `status` 문자열은 `"valid"`, `"short"`, `"no_edge"`, `"multi_edge"`, `"sub_resolution"`, `"outlier"` 여섯 개뿐이다. 통계에 포함되는 것은 `"valid"` 하나다.
- `flags`는 `frozenset[str]`이고 현재 값은 `"low_confidence"` 하나다. 플래그는 통계 포함 여부를 바꾸지 않는다.
- 사용자에게 보이는 경고 메시지는 한국어로 쓴다.
- 모든 커밋은 테스트가 통과한 상태에서 한다.

## 파일 구조

| 파일 | 책임 |
|---|---|
| `ebl_gap/types.py` | 엔진 전체가 주고받는 데이터 타입. numpy 외 의존성 없음 |
| `ebl_gap/edges.py` | 1차원 프로파일 → 50% 문턱 서브픽셀 에지. 이미지/ROI를 모름 |
| `ebl_gap/profile.py` | 이미지 + ROI + 각도 → 정렬된 2차원 프로파일 배열 |
| `ebl_gap/orientation.py` | ROI → 갭 축 각도 (Theil-Sen) |
| `ebl_gap/classify.py` | ProfileAnalysis → (status, flags, reason) |
| `ebl_gap/stats.py` | LineResult 리스트 → RoiResult (이상치 제거, 경고 생성) |
| `ebl_gap/measure.py` | 위 전부를 엮는 단일 진입점 `measure_roi` |
| `ebl_gap/metadata.py` | FEI TIFF INI 블록 파싱 → ScaleInfo, 데이터바 경계 |
| `ebl_gap/scalebar.py` | 스케일바 자동 검출 / 수동 2점 캘리브레이션 |
| `ebl_gap/dataset.py` | 세션 관리, 파일명 dose 파싱, dose–gap 집계 |
| `ebl_gap/export.py` | CSV 2종, 오버레이 PNG, 요약 텍스트 |
| `ebl_gap_gui/app.py` | 메인 윈도우, 레이아웃, 시그널 배선 |
| `ebl_gap_gui/image_view.py` | pyqtgraph 이미지 표시 + 드래그 ROI + 오버레이 |
| `ebl_gap_gui/panels.py` | 파일 목록, 결과 패널, 결과 테이블 |
| `ebl_gap_gui/dose_plot.py` | dose–gap 곡선 |
| `tests/synth.py` | 정답 갭 폭을 아는 합성 SEM 이미지 생성기 |

---

### Task 1: 프로젝트 스캐폴딩과 데이터 타입

**Files:**
- Create: `pyproject.toml`
- Create: `ebl_gap/__init__.py`
- Create: `ebl_gap/types.py`
- Create: `tests/__init__.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: `ScaleInfo(nm_per_px: float, source: str)`, `Roi(x0: int, y0: int, x1: int, y1: int)` (속성 `width`, `height`, `cx`, `cy`), `LineResult(row, left_px, right_px, width_px, width_nm, status, flags, reason)` (속성 `counts_in_stats`), `RoiResult(mean_nm, std_nm, n_valid, n_short, n_uncertain, n_low_confidence, angle_deg, lines, warnings, scale)`, `ImageRecord(path, scale, dose, roi_results, error)` — `error`는 로딩 실패나 메타데이터 경고를 담으며 Task 12·13·15·18이 읽고 쓴다, 상수 `SCALE_SOURCES`, `LINE_STATUSES`, `UNCERTAIN_STATUSES`, `LINE_FLAGS`

- [ ] **Step 1: `pyproject.toml` 작성**

```toml
[project]
name = "ebl-gap"
version = "0.1.0"
description = "EBL dose test SEM 이미지 S/D 갭 측정 툴"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "tifffile>=2023.7.10",
    "Pillow>=10.0",
]

[project.optional-dependencies]
gui = ["PySide6>=6.5", "pyqtgraph>=0.13"]
dev = ["pytest>=7.4"]

[project.scripts]
ebl-gap = "ebl_gap_gui.app:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["ebl_gap*", "ebl_gap_gui*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_types.py`:

```python
import pytest

from ebl_gap.types import (
    LINE_STATUSES,
    UNCERTAIN_STATUSES,
    LineResult,
    Roi,
    RoiResult,
    ScaleInfo,
)


def test_scale_info_rejects_unknown_source():
    with pytest.raises(ValueError, match="source"):
        ScaleInfo(nm_per_px=3.0, source="guess")


def test_scale_info_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="nm_per_px"):
        ScaleInfo(nm_per_px=0.0, source="manual")


def test_roi_normalizes_reversed_drag():
    roi = Roi(x0=400, y0=300, x1=100, y1=50)
    assert (roi.x0, roi.y0, roi.x1, roi.y1) == (100, 50, 400, 300)
    assert roi.width == 301
    assert roi.height == 251
    assert roi.cx == pytest.approx(250.0)
    assert roi.cy == pytest.approx(175.0)


def test_roi_rejects_degenerate_box():
    with pytest.raises(ValueError, match="ROI"):
        Roi(x0=10, y0=10, x1=10, y1=40)


def test_line_result_counts_in_stats_only_when_valid():
    base = dict(row=0, left_px=10.0, right_px=30.0, width_px=20.0,
                width_nm=60.0, flags=frozenset(), reason="")
    assert LineResult(status="valid", **base).counts_in_stats is True
    for status in ("short", "no_edge", "multi_edge", "sub_resolution", "outlier"):
        assert LineResult(status=status, **base).counts_in_stats is False


def test_line_result_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        LineResult(row=0, left_px=None, right_px=None, width_px=None,
                   width_nm=None, status="weird", flags=frozenset(), reason="")


def test_uncertain_statuses_is_subset_of_line_statuses():
    assert set(UNCERTAIN_STATUSES) < set(LINE_STATUSES)
    assert "valid" not in UNCERTAIN_STATUSES
    assert "short" not in UNCERTAIN_STATUSES


def test_roi_result_holds_lines_and_warnings():
    scale = ScaleInfo(nm_per_px=3.0, source="fei_metadata")
    result = RoiResult(mean_nm=60.0, std_nm=2.0, n_valid=100, n_short=3,
                       n_uncertain=5, n_low_confidence=12, angle_deg=1.5,
                       lines=(), warnings=("확인 필요",), scale=scale)
    assert result.n_total == 108
    assert result.warnings == ("확인 필요",)
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python -m pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap'`

- [ ] **Step 4: `ebl_gap/__init__.py`와 `tests/__init__.py` 생성**

`ebl_gap/__init__.py`:

```python
"""EBL dose test SEM 이미지에서 S/D 패턴 갭을 측정하는 엔진.

이 패키지는 Qt에 의존하지 않는다. GUI 없이 단독으로 import하고 테스트할 수 있어야
한다는 것이 이 패키지의 핵심 제약이다.
"""

__version__ = "0.1.0"
```

`tests/__init__.py`는 빈 파일로 만든다.

- [ ] **Step 5: `ebl_gap/types.py` 구현**

```python
"""엔진 전체가 주고받는 데이터 타입.

이 모듈은 numpy조차 import하지 않는다. 타입 정의가 계산 코드에 끌려가지 않도록
의도적으로 비워 둔 것이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SCALE_SOURCES = ("fei_metadata", "scalebar_auto", "manual")

LINE_STATUSES = (
    "valid",
    "short",
    "no_edge",
    "multi_edge",
    "sub_resolution",
    "outlier",
)

#: 통계에서 제외되고 "판정보류"로 집계되는 상태들.
#: "short"는 판정보류가 아니라 별도 카운트이므로 여기 들어가지 않는다.
UNCERTAIN_STATUSES = ("no_edge", "multi_edge", "sub_resolution", "outlier")

LINE_FLAGS = ("low_confidence",)


@dataclass(frozen=True)
class ScaleInfo:
    """픽셀 크기와 그 출처.

    출처를 값과 함께 들고 다니는 것이 이 타입의 존재 이유다. 스케일 출처를 모르는
    계측값은 나중에 재현할 수 없다.
    """

    nm_per_px: float
    source: str

    def __post_init__(self) -> None:
        if self.source not in SCALE_SOURCES:
            raise ValueError(
                f"알 수 없는 source: {self.source!r}. 허용값: {SCALE_SOURCES}"
            )
        if not self.nm_per_px > 0:
            raise ValueError(f"nm_per_px는 양수여야 한다: {self.nm_per_px!r}")


@dataclass(frozen=True)
class Roi:
    """이미지 좌표계의 축 정렬 사각형. 양 끝 픽셀을 모두 포함한다."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        x0, x1 = sorted((int(self.x0), int(self.x1)))
        y0, y1 = sorted((int(self.y0), int(self.y1)))
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "y1", y1)
        if x1 - x0 < 8 or y1 - y0 < 4:
            raise ValueError(
                f"ROI가 너무 작다: 가로 {x1 - x0 + 1}px, 세로 {y1 - y0 + 1}px "
                "(최소 가로 9px, 세로 5px)"
            )

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class LineResult:
    """스캔라인 한 줄의 측정 결과."""

    row: int
    left_px: float | None
    right_px: float | None
    width_px: float | None
    width_nm: float | None
    status: str
    flags: frozenset[str]
    reason: str

    def __post_init__(self) -> None:
        if self.status not in LINE_STATUSES:
            raise ValueError(
                f"알 수 없는 status: {self.status!r}. 허용값: {LINE_STATUSES}"
            )
        unknown = set(self.flags) - set(LINE_FLAGS)
        if unknown:
            raise ValueError(f"알 수 없는 flag: {sorted(unknown)}")

    @property
    def counts_in_stats(self) -> bool:
        return self.status == "valid"


@dataclass(frozen=True)
class RoiResult:
    """ROI 하나의 측정 결과 전체."""

    mean_nm: float | None
    std_nm: float | None
    n_valid: int
    n_short: int
    n_uncertain: int
    n_low_confidence: int
    angle_deg: float
    lines: tuple[LineResult, ...]
    warnings: tuple[str, ...]
    scale: ScaleInfo

    @property
    def n_total(self) -> int:
        return self.n_valid + self.n_short + self.n_uncertain


@dataclass
class ImageRecord:
    """이미지 한 장과 거기서 나온 측정 결과들."""

    path: Path
    scale: ScaleInfo | None = None
    dose: float | None = None
    roi_results: list[RoiResult] = field(default_factory=list)
    error: str | None = None
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `python -m pytest tests/test_types.py -v`
Expected: PASS (8 passed)

- [ ] **Step 7: 커밋**

```bash
git add pyproject.toml ebl_gap/ tests/
git commit -m "feat: add project scaffolding and engine data types"
```

---

### Task 2: 합성 SEM 이미지 생성기

정답을 아는 이미지가 없으면 10픽셀짜리 갭 측정이 맞는지 확인할 방법이 없다. 이 태스크가 이후 모든 정확도 검증의 기준이 된다.

**Files:**
- Create: `tests/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Consumes: 없음
- Produces: `synth_gap_image(*, width=512, height=512, gap_nm=50.0, nm_per_px=1.0, angle_deg=0.0, edge_sigma_px=1.5, noise_sigma=0.0, i_metal=200.0, i_gap=40.0, edge_bright=0.0, seed=0) -> np.ndarray` (float64 2차원), `gap_center_x_at_row(row, *, width, height, angle_deg) -> float`, `half_max_centre(row, i_metal=200.0, i_gap=40.0) -> float` (Task 4도 소비하는 공유 헬퍼)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_synth.py`:

```python
import numpy as np
import pytest

from tests.synth import gap_center_x_at_row, half_max_centre, synth_gap_image


def test_gap_center_is_dark_and_far_field_is_bright():
    img = synth_gap_image(width=256, height=64, gap_nm=40.0, nm_per_px=1.0)
    assert img[32, 128] == pytest.approx(40.0, abs=0.5)
    assert img[32, 5] == pytest.approx(200.0, abs=0.5)
    assert img[32, 250] == pytest.approx(200.0, abs=0.5)


def test_fifty_percent_crossing_sits_exactly_at_half_gap():
    """참값 정의: |u| == gap_px/2 인 지점의 밝기가 정확히 중간값이다."""
    img = synth_gap_image(width=256, height=8, gap_nm=40.0, nm_per_px=1.0,
                          edge_sigma_px=2.0)
    cx = (256 - 1) / 2.0
    half = 20.0
    mid = (200.0 + 40.0) / 2.0
    row = img[4]
    # 서브픽셀 위치이므로 선형보간으로 읽는다.
    left_value = np.interp(cx - half, np.arange(256), row)
    right_value = np.interp(cx + half, np.arange(256), row)
    assert left_value == pytest.approx(mid, abs=1.0)
    assert right_value == pytest.approx(mid, abs=1.0)


@pytest.mark.parametrize("angle_deg", [0.0, 2.0, 5.0, 10.0])
def test_rotation_shifts_gap_center_by_tangent(angle_deg):
    img = synth_gap_image(width=512, height=512, gap_nm=30.0, nm_per_px=1.0,
                          angle_deg=angle_deg)
    for row in (180, 256, 330):
        observed = half_max_centre(img[row])
        expected = gap_center_x_at_row(row, width=512, height=512,
                                       angle_deg=angle_deg)
        assert observed == pytest.approx(expected, abs=0.05)


def test_noise_is_reproducible_by_seed():
    a = synth_gap_image(width=64, height=16, noise_sigma=5.0, seed=7)
    b = synth_gap_image(width=64, height=16, noise_sigma=5.0, seed=7)
    c = synth_gap_image(width=64, height=16, noise_sigma=5.0, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_edge_brightening_raises_intensity_just_outside_the_gap():
    plain = synth_gap_image(width=256, height=8, gap_nm=40.0, nm_per_px=1.0,
                            edge_sigma_px=2.0, edge_bright=0.0)
    bright = synth_gap_image(width=256, height=8, gap_nm=40.0, nm_per_px=1.0,
                             edge_sigma_px=2.0, edge_bright=30.0)
    cx = int((256 - 1) / 2.0)
    just_outside = cx + 20 + 3
    assert bright[4, just_outside] > plain[4, just_outside] + 5.0
    # 먼 평탄부는 영향을 받지 않아야 문턱 계산이 흔들리지 않는다.
    assert bright[4, 5] == pytest.approx(plain[4, 5], abs=0.5)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_synth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.synth'`

- [ ] **Step 3: `tests/synth.py` 구현**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_synth.py -v`
Expected: PASS (8 passed — parametrize 4개 포함)

- [ ] **Step 5: 커밋**

```bash
git add tests/synth.py tests/test_synth.py
git commit -m "test: add synthetic SEM gap image generator with known ground truth"
```

---

### Task 3: 50% 문턱 서브픽셀 에지 검출 (1차원)

측정 정확도의 핵심이다. 이 모듈은 이미지도 ROI도 모르고 1차원 배열만 다루므로 회전·스케일 문제와 섞이지 않은 상태에서 단독으로 검증할 수 있다.

**Files:**
- Create: `ebl_gap/edges.py`
- Test: `tests/test_edges.py`

**Interfaces:**
- Consumes: 없음
- Produces: `ProfileAnalysis` (필드 `i_hi_left`, `i_hi_right`, `i_lo`, `sigma_noise`, `min_index`, `left_px`, `right_px`, `n_cross_left`, `n_cross_right`; 속성 `width_px`, `contrast`), `analyze_profile(profile, *, threshold_fraction=0.5, flat_fraction=0.2, center_fraction=0.6, hysteresis=0.1) -> ProfileAnalysis`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_edges.py`:

```python
import numpy as np
import pytest
from scipy.special import erf

from ebl_gap.edges import ProfileAnalysis, analyze_profile


def erf_profile(n=301, gap_px=20.0, sigma=2.0, i_metal=200.0, i_gap=40.0,
                noise=0.0, seed=0):
    x = np.arange(n, dtype=float)
    d = np.abs(x - (n - 1) / 2.0)
    t = 0.5 * (erf((d - gap_px / 2.0) / (np.sqrt(2.0) * sigma)) + 1.0)
    p = i_gap + (i_metal - i_gap) * t
    if noise:
        p = p + np.random.default_rng(seed).normal(0.0, noise, n)
    return p


def test_recovers_known_width_within_a_fifth_of_a_pixel():
    p = erf_profile(gap_px=20.0)
    a = analyze_profile(p)
    assert a.width_px == pytest.approx(20.0, abs=0.2)


def test_edges_are_symmetric_about_the_profile_centre():
    p = erf_profile(n=301, gap_px=30.0)
    a = analyze_profile(p)
    centre = 150.0
    assert (centre - a.left_px) == pytest.approx(a.right_px - centre, abs=0.2)


def test_narrow_gap_still_measured_when_it_is_a_tiny_fraction_of_the_roi():
    """스펙 4.4의 2단계 I_lo 추정이 실제로 필요한 상황.

    폭 301 프로파일에 12픽셀 갭이면 중앙 60% 구간(180샘플)의 하위 20%에도 금속이
    대부분 들어온다. 단순 분위수로 I_lo를 잡으면 여기서 무너진다.
    """
    p = erf_profile(n=301, gap_px=12.0, sigma=1.5)
    a = analyze_profile(p)
    assert a.width_px == pytest.approx(12.0, abs=0.5)
    assert a.i_lo == pytest.approx(40.0, abs=5.0)


def test_lower_threshold_fraction_gives_narrower_gap():
    p = erf_profile(gap_px=20.0, sigma=2.0)
    narrow = analyze_profile(p, threshold_fraction=0.3).width_px
    wide = analyze_profile(p, threshold_fraction=0.7).width_px
    assert narrow < 20.0 < wide


def test_flat_profile_reports_no_edges_and_zero_contrast():
    p = np.full(201, 180.0)
    a = analyze_profile(p)
    assert a.left_px is None
    assert a.right_px is None
    assert a.width_px is None
    assert a.contrast == pytest.approx(0.0, abs=1e-6)


def test_hysteresis_suppresses_noise_induced_extra_crossings():
    """문턱 근처에서 잡음이 여러 번 넘나들어도 교차는 한 번으로 센다."""
    p = erf_profile(n=301, gap_px=20.0, sigma=2.0, noise=6.0, seed=3)
    a = analyze_profile(p)
    assert a.n_cross_left == 1
    assert a.n_cross_right == 1


def test_two_gaps_in_one_profile_are_reported_as_multiple_crossings():
    p = np.full(401, 200.0)
    p[90:110] = 40.0
    p[290:310] = 40.0
    a = analyze_profile(p)
    assert a.n_cross_left + a.n_cross_right >= 3


def test_sigma_noise_estimated_from_the_flat_ends():
    p = erf_profile(n=301, gap_px=20.0, noise=4.0, seed=11)
    a = analyze_profile(p)
    assert a.sigma_noise == pytest.approx(4.0, rel=0.35)


def test_rejects_profile_shorter_than_nine_samples():
    with pytest.raises(ValueError, match="프로파일"):
        analyze_profile(np.zeros(8))


def test_asymmetric_illumination_uses_separate_left_and_right_thresholds():
    """좌우 전극 밝기가 다르면 문턱도 좌우 따로 잡아야 폭이 안 틀어진다."""
    p = erf_profile(n=301, gap_px=20.0, sigma=2.0)
    p[:150] = 40.0 + (p[:150] - 40.0) * 1.4  # 왼쪽 전극만 40% 밝게
    a = analyze_profile(p)
    assert a.i_hi_left > a.i_hi_right
    assert a.width_px == pytest.approx(20.0, abs=0.3)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_edges.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.edges'`

- [ ] **Step 3: `ebl_gap/edges.py` 구현**

```python
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

    # 대비가 없으면 에지를 찾을 수 없다. 이 가드는 선택적 방어가 아니라 필수다:
    # 대비가 정확히 0이면 히스테리시스 밴드 폭(hysteresis * (i_hi - i_lo))도 0이 되어
    # 같은 값을 가진 모든 샘플이 armed 플래그를 뒤집고, _count_rising이 수십 번의
    # 가짜 교차를 센다. 그러면 Task 6이 평탄한(갭 없는) 라인을 multi_edge로 오판한다.
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
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_edges.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap/edges.py tests/test_edges.py
git commit -m "feat: add subpixel half-max edge detection with two-pass gap floor estimate"
```

---

### Task 4: ROI 회전 정렬 프로파일 추출

**Files:**
- Create: `ebl_gap/profile.py`
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: `Roi` (Task 1), `synth_gap_image`/`gap_center_x_at_row`/`half_max_centre` (Task 2, 테스트용)
- Produces: `extract_profiles(image, roi, angle_deg, *, along_average=1) -> np.ndarray` — `(roi.height, roi.width)` float64 배열. 각 **행**이 갭 축에 수직인 방향(측정 방향)의 밝기 프로파일이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_profile.py`:

```python
import numpy as np
import pytest

from ebl_gap.profile import extract_profiles
from ebl_gap.types import Roi
from tests.synth import gap_center_x_at_row, half_max_centre, synth_gap_image


def test_zero_angle_full_roi_is_the_identity():
    img = np.arange(40 * 60, dtype=float).reshape(40, 60)
    roi = Roi(0, 0, 59, 39)
    out = extract_profiles(img, roi, 0.0)
    assert out.shape == (40, 60)
    assert np.allclose(out, img, atol=1e-9)


def test_zero_angle_sub_roi_is_a_plain_crop():
    img = np.arange(40 * 60, dtype=float).reshape(40, 60)
    roi = Roi(10, 5, 39, 24)
    out = extract_profiles(img, roi, 0.0)
    assert out.shape == (20, 30)
    assert np.allclose(out, img[5:25, 10:40], atol=1e-9)


@pytest.mark.parametrize("angle_deg", [2.0, 5.0, 10.0])
def test_correct_angle_makes_every_row_share_one_gap_column(angle_deg):
    """정렬이 제대로 되면 기울어진 갭이 모든 행에서 같은 열에 온다.

    갭 중심은 argmin이 아니라 50% 문턱 중점으로 잡는다. 회전 정렬된 프로파일은
    행마다 샘플링 위상이 달라 평탄한 갭 바닥의 argmin이 4픽셀까지 흔들리는데,
    그것은 정렬 오차가 아니라 argmin의 동점 처리 방식일 뿐이다.
    """
    img = synth_gap_image(width=512, height=512, gap_nm=30.0, nm_per_px=1.0,
                          angle_deg=angle_deg)
    roi = Roi(106, 106, 405, 405)
    centres = np.array([half_max_centre(row)
                        for row in extract_profiles(img, roi, angle_deg)])
    assert np.ptp(centres) < 0.1


def test_wrong_angle_leaves_the_gap_drifting_across_columns():
    img = synth_gap_image(width=512, height=512, gap_nm=30.0, nm_per_px=1.0,
                          angle_deg=10.0)
    roi = Roi(106, 106, 405, 405)
    out = extract_profiles(img, roi, 0.0)
    minima = np.argmin(out, axis=1)
    assert minima.max() - minima.min() > 40


def test_along_axis_averaging_reduces_noise():
    rng = np.random.default_rng(0)
    img = np.full((200, 200), 100.0) + rng.normal(0.0, 10.0, (200, 200))
    roi = Roi(20, 20, 179, 179)
    plain = extract_profiles(img, roi, 0.0, along_average=1)
    smoothed = extract_profiles(img, roi, 0.0, along_average=9)
    assert smoothed.std() < plain.std() * 0.6


def test_empty_image_is_rejected_with_a_korean_error():
    """빈 이미지는 조용히 쓰레기 값을 돌려주는 대신 명확히 거부해야 한다.

    ndim만 검사하면 (0, 0) 배열이 통과하고 map_coordinates가 초기화되지 않은
    메모리를 담은 배열을 돌려준다. 예외보다 나쁜 실패 방식이다.
    """
    with pytest.raises(ValueError, match="이미지"):
        extract_profiles(np.empty((0, 0)), Roi(0, 0, 8, 4), 0.0)


def test_sampling_outside_the_image_clamps_instead_of_raising():
    img = np.full((50, 50), 7.0)
    roi = Roi(0, 0, 49, 49)
    out = extract_profiles(img, roi, 20.0)
    assert np.isfinite(out).all()
    assert out.min() == pytest.approx(7.0)


def test_angle_sign_matches_the_project_convention():
    """양의 각도는 행이 내려갈수록 갭이 오른쪽으로 가는 경우다."""
    img = synth_gap_image(width=512, height=512, gap_nm=30.0, nm_per_px=1.0,
                          angle_deg=8.0)
    assert gap_center_x_at_row(400, width=512, height=512, angle_deg=8.0) > \
           gap_center_x_at_row(100, width=512, height=512, angle_deg=8.0)
    roi = Roi(106, 106, 405, 405)
    centres = np.array([half_max_centre(row)
                        for row in extract_profiles(img, roi, 8.0)])
    assert np.ptp(centres) < 0.1
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.profile'`

- [ ] **Step 3: `ebl_gap/profile.py` 구현**

```python
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
    w, h = roi.width, roi.height

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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_profile.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap/profile.py tests/test_profile.py
git commit -m "feat: add rotation-aligned scanline profile extraction"
```

---

### Task 5: 갭 축 각도 추정

**Files:**
- Create: `ebl_gap/orientation.py`
- Test: `tests/test_orientation.py`

**Interfaces:**
- Consumes: `extract_profiles` (Task 4), `analyze_profile` (Task 3), `Roi` (Task 1)
- Produces: `InsufficientEdgesError(Exception)`, `estimate_angle_deg(image, roi, *, min_rows=5, **profile_kwargs) -> tuple[float, int]` — (각도, 각도 추정에 실제로 쓰인 행 수)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_orientation.py`:

```python
import numpy as np
import pytest

from ebl_gap.orientation import InsufficientEdgesError, estimate_angle_deg
from ebl_gap.types import Roi
from tests.synth import synth_gap_image


@pytest.mark.parametrize("angle_deg", [0.0, 2.0, 5.0, 10.0, -6.0])
def test_recovers_the_true_angle(angle_deg):
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0,
                          angle_deg=angle_deg)
    roi = Roi(106, 106, 405, 405)
    estimated, n_rows = estimate_angle_deg(img, roi)
    assert estimated == pytest.approx(angle_deg, abs=0.2)
    assert n_rows == 300


def test_survives_noise_that_breaks_some_rows():
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0,
                          angle_deg=7.0, noise_sigma=8.0, seed=5)
    roi = Roi(106, 106, 405, 405)
    estimated, _ = estimate_angle_deg(img, roi)
    assert estimated == pytest.approx(7.0, abs=0.5)


def test_theil_sen_is_not_dragged_by_a_few_corrupted_rows():
    """최소자승이라면 끌려갈 상황에서 Theil-Sen은 버텨야 한다.

    손상된 행이 analyze_profile 단계에서 걸러지면 회귀에 도달하지 못하고, 그러면
    최소자승으로 바꿔도 이 테스트가 통과한다 — 아무것도 검증하지 못한다는 뜻이다.
    그래서 대비를 유지한 채 갭 위치만 60픽셀 옮긴다. 에지는 정상적으로 검출되고
    회귀에 진짜 이상치로 들어간다. ROI 맨 위 20행을 옮기는 것은 지렛대 효과를
    키우기 위해서다 — 최소자승이라면 기울기가 3도에서 약 -1.3도까지 끌려간다.
    """
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0,
                          angle_deg=3.0)
    img[106:126] = np.roll(img[106:126], 60, axis=1)
    roi = Roi(106, 106, 405, 405)
    estimated, n_rows = estimate_angle_deg(img, roi)
    # 손상 행이 실제로 회귀에 들어갔는지 확인한다. 걸러졌다면 이 테스트는 무의미하다.
    assert n_rows >= 290
    assert estimated == pytest.approx(3.0, abs=0.5)


def test_raises_when_there_are_too_few_usable_rows():
    img = np.full((256, 256), 180.0)
    roi = Roi(28, 28, 227, 227)
    with pytest.raises(InsufficientEdgesError, match="에지"):
        estimate_angle_deg(img, roi)


def test_reports_how_many_rows_were_usable():
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0)
    img[106:156, :] = 200.0  # ROI 상단 50행을 못 쓰게 만든다
    roi = Roi(106, 106, 405, 405)
    _, n_rows = estimate_angle_deg(img, roi)
    assert 240 <= n_rows <= 260
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_orientation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.orientation'`

- [ ] **Step 3: `ebl_gap/orientation.py` 구현**

```python
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


def estimate_angle_deg(image, roi: Roi, *, min_rows: int = 5,
                       **profile_kwargs) -> tuple[float, int]:
    """갭 축 각도를 Theil-Sen 추정으로 구한다.

    각도 0으로 뽑은 프로파일에서 행마다 좌/우 에지 위치를 구하면, 에지 위치는 행
    번호의 선형 함수다. 그 기울기가 곧 tan(theta)이므로 한 번의 패스로 충분하다.

    최소자승 대신 Theil-Sen을 쓰는 이유는 에지 검출이 몇 줄 실패하거나 엉뚱한 값을
    내놓아도 추정치가 끌려가지 않기 때문이다.

    Returns:
        (각도(도), 추정에 쓰인 행 수)
    """
    profiles = extract_profiles(image, roi, 0.0)

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
    angle_deg = float(np.degrees(np.arctan(0.5 * (slope_left + slope_right))))
    return angle_deg, len(rows)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_orientation.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap/orientation.py tests/test_orientation.py
git commit -m "feat: add Theil-Sen gap axis angle estimation"
```

---

### Task 6: 라인 상태 판정

**Files:**
- Create: `ebl_gap/classify.py`
- Test: `tests/test_classify.py`

**Interfaces:**
- Consumes: `ProfileAnalysis` (Task 3), `LineResult`, `UNCERTAIN_STATUSES` (Task 1)
- Produces: `classify_line(analysis, *, contrast_k=5.0, min_width_px=3.0, low_confidence_px=10.0) -> tuple[str, frozenset[str], str]` — (status, flags, reason)

판정 순서는 스펙 5.1절을 그대로 따른다: `short` → `no_edge` → `multi_edge` → `sub_resolution` → `valid`. `outlier`는 전체 라인이 모여야 판정할 수 있으므로 여기가 아니라 Task 7의 `stats.py`에서 붙인다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_classify.py`:

```python
import pytest

from ebl_gap.classify import classify_line
from ebl_gap.edges import ProfileAnalysis


def make_analysis(**overrides) -> ProfileAnalysis:
    """판정 테스트용 ProfileAnalysis를 직접 만든다.

    이미지를 거치지 않고 레벨 값을 직접 지정해야 경계값을 정확히 찌를 수 있다.
    """
    fields = dict(i_hi_left=200.0, i_hi_right=200.0, i_lo=40.0, sigma_noise=4.0,
                  min_index=100, left_px=90.0, right_px=110.0,
                  n_cross_left=1, n_cross_right=1)
    fields.update(overrides)
    return ProfileAnalysis(**fields)


def test_healthy_line_is_valid_with_no_flags():
    status, flags, reason = classify_line(make_analysis())
    assert status == "valid"
    assert flags == frozenset()
    assert reason == ""


def test_low_contrast_is_short_even_when_edges_were_found():
    """대비가 노이즈 수준이면 찾았다는 에지는 잡음이다. short가 먼저다."""
    analysis = make_analysis(i_hi_left=50.0, i_hi_right=50.0, i_lo=40.0,
                             sigma_noise=4.0)
    status, _, reason = classify_line(analysis)
    assert status == "short"
    assert "대비" in reason


def test_zero_contrast_with_zero_noise_is_still_short():
    """노이즈 없는 합성 이미지에서 0 <= 0 이 short로 잡혀야 한다."""
    analysis = make_analysis(i_hi_left=100.0, i_hi_right=100.0, i_lo=100.0,
                             sigma_noise=0.0)
    assert classify_line(analysis)[0] == "short"


def test_missing_edge_is_no_edge():
    assert classify_line(make_analysis(left_px=None))[0] == "no_edge"
    assert classify_line(make_analysis(right_px=None))[0] == "no_edge"


def test_extra_crossings_are_multi_edge():
    assert classify_line(make_analysis(n_cross_left=2))[0] == "multi_edge"
    assert classify_line(make_analysis(n_cross_right=3))[0] == "multi_edge"


def test_width_below_three_pixels_is_sub_resolution():
    analysis = make_analysis(left_px=100.0, right_px=102.9)
    status, flags, reason = classify_line(analysis)
    assert status == "sub_resolution"
    assert flags == frozenset()
    assert "3" in reason


def test_width_exactly_three_pixels_is_valid_not_sub_resolution():
    """경계는 포함이다: min_width_px 이상이면 valid."""
    analysis = make_analysis(left_px=100.0, right_px=103.0)
    assert classify_line(analysis)[0] == "valid"


def test_width_between_three_and_ten_pixels_gets_low_confidence_flag():
    status, flags, _ = classify_line(make_analysis(left_px=100.0, right_px=105.0))
    assert status == "valid"
    assert flags == frozenset({"low_confidence"})


def test_width_exactly_ten_pixels_is_not_low_confidence():
    """경계는 배제다: low_confidence_px 이상이면 플래그가 붙지 않는다."""
    _, flags, _ = classify_line(make_analysis(left_px=100.0, right_px=110.0))
    assert flags == frozenset()


def test_thresholds_are_tunable():
    analysis = make_analysis(left_px=100.0, right_px=105.0)
    assert classify_line(analysis, min_width_px=6.0)[0] == "sub_resolution"
    assert classify_line(analysis, low_confidence_px=4.0)[1] == frozenset()


def test_short_check_runs_before_multi_edge():
    analysis = make_analysis(i_hi_left=50.0, i_hi_right=50.0, i_lo=40.0,
                             sigma_noise=4.0, n_cross_left=5)
    assert classify_line(analysis)[0] == "short"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_classify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.classify'`

- [ ] **Step 3: `ebl_gap/classify.py` 구현**

```python
"""스캔라인 한 줄의 상태를 판정한다.

판정은 status와 flags로 나뉜다. 이 둘을 섞지 않는 것이 중요하다. status는 라인이
통계에 들어가는지를 혼자서 결정하고, flags는 통계 포함 여부를 바꾸지 않는 부가
정보다.
"""

from __future__ import annotations

from ebl_gap.edges import ProfileAnalysis

_EPS = 1e-9


def classify_line(
    analysis: ProfileAnalysis,
    *,
    contrast_k: float = 5.0,
    min_width_px: float = 3.0,
    low_confidence_px: float = 10.0,
) -> tuple[str, frozenset[str], str]:
    """(status, flags, reason)을 돌려준다.

    판정 순서는 스펙 5.1절을 따른다. 대비 검사를 가장 먼저 하는 이유는, 갭이 닫힌
    라인에서도 잡음 때문에 에지가 "찾아지는" 일이 있기 때문이다. 그 값을 정상으로
    보고하면 short를 놓친다.
    """
    contrast = analysis.contrast
    noise_floor = contrast_k * max(analysis.sigma_noise, _EPS)
    if contrast <= noise_floor:
        return (
            "short",
            frozenset(),
            f"대비 {contrast:.1f} <= 노이즈 {analysis.sigma_noise:.1f} x {contrast_k:g}",
        )

    if analysis.left_px is None or analysis.right_px is None:
        missing = "왼쪽" if analysis.left_px is None else "오른쪽"
        return "no_edge", frozenset(), f"{missing} 문턱 상향 교차를 찾지 못함"

    if analysis.n_cross_left > 1 or analysis.n_cross_right > 1:
        return (
            "multi_edge",
            frozenset(),
            f"교차 횟수 좌 {analysis.n_cross_left} 우 {analysis.n_cross_right} "
            "(ROI에 다른 패턴이 포함된 것으로 보임)",
        )

    width_px = analysis.width_px
    assert width_px is not None  # 위에서 None을 걸렀다
    if width_px < min_width_px:
        return (
            "sub_resolution",
            frozenset(),
            f"폭 {width_px:.2f}px < {min_width_px:g}px (해상도 한계 이하)",
        )

    flags = frozenset({"low_confidence"}) if width_px < low_confidence_px else frozenset()
    return "valid", flags, ""
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_classify.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap/classify.py tests/test_classify.py
git commit -m "feat: add line status classification with separate confidence flags"
```

---

### Task 7: 통계 집계와 경고 생성

**Files:**
- Create: `ebl_gap/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `LineResult`, `RoiResult`, `ScaleInfo`, `UNCERTAIN_STATUSES` (Task 1)
- Produces: `mark_outliers(lines, *, mad_k=3.5) -> list[LineResult]`, `summarize(lines, *, scale, angle_deg, extra_warnings=()) -> RoiResult`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stats.py`:

```python
import pytest

from ebl_gap.stats import mark_outliers, summarize
from ebl_gap.types import LineResult, ScaleInfo

SCALE = ScaleInfo(nm_per_px=1.0, source="fei_metadata")


def line(row, width_nm, status="valid", flags=frozenset()):
    return LineResult(row=row, left_px=100.0,
                      right_px=None if width_nm is None else 100.0 + width_nm,
                      width_px=width_nm, width_nm=width_nm, status=status,
                      flags=flags, reason="")


def valid_lines(widths):
    return [line(i, w) for i, w in enumerate(widths)]


def test_mean_and_std_use_valid_lines_only():
    lines = valid_lines([50.0, 52.0, 48.0]) + [
        line(3, None, status="short"),
        line(4, 999.0, status="no_edge"),
    ]
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert result.mean_nm == pytest.approx(50.0)
    assert result.std_nm == pytest.approx(2.0)
    assert result.n_valid == 3
    assert result.n_short == 1
    assert result.n_uncertain == 1


def test_low_confidence_lines_are_included_in_the_mean():
    """플래그는 통계 포함 여부를 바꾸지 않는다."""
    lines = valid_lines([50.0, 50.0]) + [
        line(2, 50.0, flags=frozenset({"low_confidence"}))
    ]
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert result.n_valid == 3
    assert result.n_low_confidence == 1
    assert result.mean_nm == pytest.approx(50.0)


def test_mean_is_none_when_no_line_is_valid():
    result = summarize([line(0, None, status="short")], scale=SCALE, angle_deg=0.0)
    assert result.mean_nm is None
    assert result.std_nm is None


def test_std_is_none_with_a_single_valid_line():
    result = summarize(valid_lines([50.0]), scale=SCALE, angle_deg=0.0)
    assert result.mean_nm == pytest.approx(50.0)
    assert result.std_nm is None


def test_mark_outliers_relabels_far_lines_and_leaves_the_rest():
    """값이 전부 같아 MAD가 0이어도, 분해능이 척도가 되어 판정이 살아 있어야 한다.

    여기서 판정을 포기하면 이상치 제거가 가장 필요한 상황에서 기능이 꺼진다.
    """
    lines = valid_lines([50.0] * 20 + [200.0])
    marked = mark_outliers(lines, resolution_nm=1.0)
    assert marked[-1].status == "outlier"
    assert "척도" in marked[-1].reason
    assert all(m.status == "valid" for m in marked[:-1])


def test_mark_outliers_does_nothing_without_any_scale():
    """MAD도 0이고 분해능도 모르면 판정 기준 자체가 없다. 전부 남겨야 한다."""
    marked = mark_outliers(valid_lines([50.0] * 10))
    assert all(m.status == "valid" for m in marked)


def test_mark_outliers_does_nothing_when_most_values_tie():
    """MAD가 0이면 편차 판정이 불가능하다. 근소한 차이를 이상치로 만들면 안 된다.

    값이 전부 같은 경우만 테스트하면 가드를 지워도 통과한다 — 모든 편차가 정확히
    0이라 limit=0과 비교해도 걸리지 않기 때문이다. 일부만 같은 경우가 진짜 함정이고,
    픽셀 양자화 때문에 여러 라인이 같은 nm 값으로 떨어지는 것은 실측에서 흔하다.
    """
    marked = mark_outliers(valid_lines([50.0] * 9 + [51.0]), resolution_nm=1.0)
    assert all(m.status == "valid" for m in marked)


def test_mark_outliers_ignores_non_valid_lines():
    lines = valid_lines([50.0] * 10) + [line(10, 999.0, status="no_edge")]
    marked = mark_outliers(lines)
    assert marked[-1].status == "no_edge"


def test_short_ratio_at_or_above_five_percent_warns():
    lines = valid_lines([50.0] * 19) + [line(19, None, status="short")]
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert any("short" in w for w in result.warnings)


def test_uncertain_ratio_at_or_above_twenty_percent_warns():
    lines = valid_lines([50.0] * 8) + [line(8 + i, None, status="no_edge")
                                       for i in range(2)]
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert any("ROI 재설정" in w for w in result.warnings)


def test_fewer_than_ten_valid_lines_warns():
    result = summarize(valid_lines([50.0] * 9), scale=SCALE, angle_deg=0.0)
    assert any("유효 라인 부족" in w for w in result.warnings)


def test_high_relative_spread_warns():
    lines = valid_lines([30.0, 50.0, 70.0] * 5)
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert any("편차" in w for w in result.warnings)


def test_non_metadata_scale_source_warns():
    manual = ScaleInfo(nm_per_px=1.0, source="manual")
    result = summarize(valid_lines([50.0] * 20), scale=manual, angle_deg=0.0)
    assert any("스케일" in w for w in result.warnings)


def test_low_confidence_lines_produce_an_actionable_warning():
    lines = [line(i, 6.0, flags=frozenset({"low_confidence"})) for i in range(20)]
    result = summarize(lines, scale=SCALE, angle_deg=0.0)
    assert any("배율" in w for w in result.warnings)


def test_extra_warnings_are_appended():
    result = summarize(valid_lines([50.0] * 20), scale=SCALE, angle_deg=0.0,
                       extra_warnings=("각도 추정 실패",))
    assert "각도 추정 실패" in result.warnings


def test_clean_measurement_produces_no_warnings():
    result = summarize(valid_lines([50.0, 50.5, 49.5] * 10), scale=SCALE,
                       angle_deg=1.2)
    assert result.warnings == ()
    assert result.angle_deg == pytest.approx(1.2)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.stats'`

- [ ] **Step 3: `ebl_gap/stats.py` 구현**

```python
"""라인 결과를 모아 ROI 하나의 통계와 경고를 만든다."""

from __future__ import annotations

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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_stats.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap/stats.py tests/test_stats.py
git commit -m "feat: add ROI statistics with MAD outlier rejection and user warnings"
```

---

### Task 8: 측정 오케스트레이션과 정확도 회귀 테스트

이 태스크의 정확도 테스트가 프로젝트 전체의 품질 게이트다. 여기가 통과하지 않으면 나머지가 아무리 잘 만들어져도 측정값을 믿을 수 없다.

**Files:**
- Create: `ebl_gap/measure.py`
- Test: `tests/test_measure.py`
- Test: `tests/test_accuracy.py`

**Interfaces:**
- Consumes: Task 1~7 전부
- Produces: `MeasureParams` (dataclass), `measure_roi(image, roi, scale, *, angle_deg=None, params=MeasureParams()) -> RoiResult`

- [ ] **Step 1: 실패하는 통합 테스트 작성**

`tests/test_measure.py`:

```python
import numpy as np
import pytest

from ebl_gap.measure import MeasureParams, measure_roi
from ebl_gap.types import Roi, ScaleInfo
from tests.synth import synth_gap_image

SCALE = ScaleInfo(nm_per_px=1.0, source="fei_metadata")
ROI = Roi(106, 106, 405, 405)


def test_measures_every_row_of_the_roi():
    """이미지 최대 해상도를 그대로 쓴다: ROI 세로 픽셀 수만큼 라인이 나온다."""
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    result = measure_roi(img, ROI, SCALE)
    assert len(result.lines) == ROI.height == 300
    assert [ln.row for ln in result.lines] == list(range(300))


def test_estimates_the_angle_when_not_given():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0, angle_deg=6.0)
    result = measure_roi(img, ROI, SCALE)
    assert result.angle_deg == pytest.approx(6.0, abs=0.2)


def test_uses_the_supplied_angle_without_estimating():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0, angle_deg=6.0)
    result = measure_roi(img, ROI, SCALE, angle_deg=0.0)
    assert result.angle_deg == pytest.approx(0.0)
    # 각도를 무시하면 폭이 1/cos(6도) = 0.55% 과대평가된다.
    assert result.mean_nm > 40.0


def test_falls_back_to_zero_angle_with_a_warning_when_estimation_fails():
    img = np.full((512, 512), 180.0)
    result = measure_roi(img, ROI, SCALE)
    assert result.angle_deg == pytest.approx(0.0)
    assert any("각도" in w for w in result.warnings)


def test_a_closed_gap_is_reported_as_short_not_as_a_measurement():
    img = np.full((512, 512), 200.0)
    rng = np.random.default_rng(0)
    img = img + rng.normal(0.0, 3.0, img.shape)
    result = measure_roi(img, ROI, SCALE)
    assert result.mean_nm is None
    assert result.n_short == 300
    assert any("short" in w for w in result.warnings)


def test_widths_are_converted_to_nanometres_with_the_given_scale():
    img = synth_gap_image(gap_nm=60.0, nm_per_px=1.0)
    coarse = ScaleInfo(nm_per_px=2.0, source="manual")
    result = measure_roi(img, ROI, coarse)
    # 이미지의 갭은 60픽셀이고 스케일이 2 nm/px이므로 120 nm로 읽혀야 한다.
    assert result.mean_nm == pytest.approx(120.0, abs=2.0)


def test_params_are_threaded_through_to_the_edge_detector():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0, edge_sigma_px=3.0)
    narrow = measure_roi(img, ROI, SCALE,
                         params=MeasureParams(threshold_fraction=0.3)).mean_nm
    wide = measure_roi(img, ROI, SCALE,
                       params=MeasureParams(threshold_fraction=0.7)).mean_nm
    assert narrow < 40.0 < wide


def test_locally_filled_rows_do_not_drag_the_mean():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    img[200:205, 240:280] = 200.0  # 5개 행의 갭을 금속으로 메운다
    result = measure_roi(img, ROI, SCALE)
    assert result.mean_nm == pytest.approx(40.0, abs=1.0)
    # 메워진 행은 에지가 검출되긴 하지만 폭이 크게 어긋나고, mark_outliers의 MAD
    # 검사에서 outlier로 재분류된다. outlier는 UNCERTAIN_STATUSES에 속한다.
    assert result.n_uncertain == 5
    assert result.n_valid + result.n_short + result.n_uncertain == 300


def test_a_widened_row_is_rejected_as_an_outlier():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    img[200:205, 200:312] = 40.0  # 5개 행의 갭만 세 배 가까이 넓힌다
    result = measure_roi(img, ROI, SCALE)
    assert result.mean_nm == pytest.approx(40.0, abs=1.0)
    assert any(ln.status == "outlier" for ln in result.lines)


def test_result_carries_the_scale_it_was_measured_with():
    img = synth_gap_image(gap_nm=40.0, nm_per_px=1.0)
    result = measure_roi(img, ROI, SCALE)
    assert result.scale is SCALE
```

- [ ] **Step 2: 실패하는 정확도 회귀 테스트 작성**

`tests/test_accuracy.py`:

```python
"""측정 엔진의 정확도를 숫자로 고정하는 회귀 테스트.

스펙 8절의 기준을 그대로 구현한다: 갭 20/30/50/80/100 nm x 각도 0/2/5/10도 x
SNR 3수준에서 측정값이 참값의 +-1 px 이내여야 한다.

nm_per_px=1.0을 쓰는 이유는 정확도와 해상도를 분리해서 보기 위해서다. 픽셀이 굵어서
생기는 한계는 아래 test_coarse_pixels_* 에서 따로 다룬다.
"""

import pytest

from ebl_gap.measure import measure_roi
from ebl_gap.types import Roi, ScaleInfo
from tests.synth import synth_gap_image

ROI = Roi(106, 106, 405, 405)
NM_PER_PX = 1.0
SCALE = ScaleInfo(nm_per_px=NM_PER_PX, source="fei_metadata")
TOLERANCE_NM = 1.0 * NM_PER_PX


@pytest.mark.parametrize("gap_nm", [20.0, 30.0, 50.0, 80.0, 100.0])
@pytest.mark.parametrize("angle_deg", [0.0, 2.0, 5.0, 10.0])
@pytest.mark.parametrize("noise_sigma", [0.0, 3.0, 8.0])
def test_measured_gap_is_within_one_pixel_of_truth(gap_nm, angle_deg, noise_sigma):
    img = synth_gap_image(
        width=512, height=512, gap_nm=gap_nm, nm_per_px=NM_PER_PX,
        angle_deg=angle_deg, edge_sigma_px=1.5, noise_sigma=noise_sigma,
        edge_bright=15.0, seed=int(gap_nm * 100 + angle_deg * 10 + noise_sigma),
    )
    result = measure_roi(img, ROI, SCALE)
    assert result.mean_nm is not None, f"측정 실패: {result.warnings}"
    assert result.mean_nm == pytest.approx(gap_nm, abs=TOLERANCE_NM)


@pytest.mark.parametrize("gap_nm", [30.0, 50.0, 100.0])
def test_coarse_pixels_still_land_within_one_pixel(gap_nm):
    """100k배 촬영에 해당하는 3 nm/px에서도 +-1픽셀(=3nm) 안에 들어와야 한다."""
    nm_per_px = 3.0
    img = synth_gap_image(
        width=512, height=512, gap_nm=gap_nm, nm_per_px=nm_per_px,
        angle_deg=3.0, edge_sigma_px=1.2, noise_sigma=4.0, edge_bright=15.0,
        seed=int(gap_nm),
    )
    result = measure_roi(img, ROI, ScaleInfo(nm_per_px, "fei_metadata"))
    assert result.mean_nm == pytest.approx(gap_nm, abs=1.0 * nm_per_px)


def test_coarse_pixels_flag_low_confidence_for_a_thirty_nanometre_gap():
    """3 nm/px에서 30 nm 갭은 10픽셀이다. 값은 내되 정밀도 주의를 띄워야 한다."""
    nm_per_px = 3.0
    img = synth_gap_image(width=512, height=512, gap_nm=27.0,
                          nm_per_px=nm_per_px, edge_sigma_px=1.2, seed=1)
    result = measure_roi(img, ROI, ScaleInfo(nm_per_px, "fei_metadata"))
    assert result.n_low_confidence > 0
    assert any("배율" in w for w in result.warnings)


def test_edge_brightening_does_not_systematically_narrow_the_gap():
    """평탄부에서 문턱을 잡는 설계가 실제로 효과가 있는지 확인한다."""
    common = dict(width=512, height=512, gap_nm=50.0, nm_per_px=1.0,
                  edge_sigma_px=2.0, seed=2)
    plain = measure_roi(synth_gap_image(edge_bright=0.0, **common), ROI, SCALE)
    bright = measure_roi(synth_gap_image(edge_bright=40.0, **common), ROI, SCALE)
    assert abs(bright.mean_nm - plain.mean_nm) < 1.0
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python -m pytest tests/test_measure.py tests/test_accuracy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.measure'`

- [ ] **Step 4: `ebl_gap/measure.py` 구현**

```python
"""ROI 하나를 측정하는 단일 진입점.

GUI, CLI, 노트북 어디서든 이 함수 하나만 부르면 된다. 여기 아래로는 Qt가 전혀
등장하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from ebl_gap.classify import classify_line
from ebl_gap.edges import analyze_profile
from ebl_gap.orientation import InsufficientEdgesError, estimate_angle_deg
from ebl_gap.profile import extract_profiles
from ebl_gap.stats import mark_outliers, summarize
from ebl_gap.types import LineResult, Roi, RoiResult, ScaleInfo


@dataclass(frozen=True)
class MeasureParams:
    """측정에 쓰이는 모든 조절값. GUI가 그대로 노출한다."""

    threshold_fraction: float = 0.5
    flat_fraction: float = 0.2
    center_fraction: float = 0.6
    hysteresis: float = 0.1
    along_average: int = 1
    contrast_k: float = 5.0
    min_width_px: float = 3.0
    low_confidence_px: float = 10.0
    mad_k: float = 3.5

    @property
    def edge_kwargs(self) -> dict:
        return dict(
            threshold_fraction=self.threshold_fraction,
            flat_fraction=self.flat_fraction,
            center_fraction=self.center_fraction,
            hysteresis=self.hysteresis,
        )

    @property
    def classify_kwargs(self) -> dict:
        return dict(
            contrast_k=self.contrast_k,
            min_width_px=self.min_width_px,
            low_confidence_px=self.low_confidence_px,
        )


def measure_roi(
    image,
    roi: Roi,
    scale: ScaleInfo,
    *,
    angle_deg: float | None = None,
    params: MeasureParams = MeasureParams(),
) -> RoiResult:
    """ROI 안의 갭 폭을 모든 스캔라인에서 측정한다.

    Args:
        image: 2차원 밝기 배열.
        roi: 측정할 사각 영역.
        scale: 픽셀 크기와 그 출처.
        angle_deg: None이면 자동 추정한다. 값을 주면 그대로 쓴다.
        params: 측정 조절값.
    """
    extra_warnings: list[str] = []

    if angle_deg is None:
        try:
            angle_deg, _ = estimate_angle_deg(image, roi, **params.edge_kwargs)
        except InsufficientEdgesError as exc:
            angle_deg = 0.0
            extra_warnings.append(f"각도 자동 추정 실패, 0도로 측정함 ({exc})")

    profiles = extract_profiles(image, roi, angle_deg,
                                along_average=params.along_average)

    lines: list[LineResult] = []
    for row, profile in enumerate(profiles):
        try:
            analysis = analyze_profile(profile, **params.edge_kwargs)
        except ValueError as exc:
            lines.append(LineResult(row=row, left_px=None, right_px=None,
                                    width_px=None, width_nm=None,
                                    status="no_edge", flags=frozenset(),
                                    reason=str(exc)))
            continue

        status, flags, reason = classify_line(analysis, **params.classify_kwargs)
        width_px = analysis.width_px
        width_nm = None if width_px is None else width_px * scale.nm_per_px
        lines.append(LineResult(
            row=row,
            left_px=analysis.left_px,
            right_px=analysis.right_px,
            width_px=width_px,
            width_nm=width_nm,
            status=status,
            flags=flags,
            reason=reason,
        ))

    lines = mark_outliers(lines, mad_k=params.mad_k,
                          resolution_nm=scale.nm_per_px)
    return summarize(lines, scale=scale, angle_deg=angle_deg,
                     extra_warnings=tuple(extra_warnings))
```

- [ ] **Step 5: 통합 테스트 통과 확인**

Run: `python -m pytest tests/test_measure.py -v`
Expected: PASS (10 passed)

- [ ] **Step 6: 정확도 게이트 통과 확인**

Run: `python -m pytest tests/test_accuracy.py -v`
Expected: PASS (65 passed — 60개 파라미터 조합 + 5개)

정확도 테스트가 하나라도 실패하면 **넘어가지 말고 원인을 찾는다.** 실패하는 조합의 프로파일을 한 줄 뽑아 `analyze_profile`이 잡은 `i_hi_left`, `i_hi_right`, `i_lo`를 참값과 비교하는 것이 가장 빠른 진단이다. 허용 오차를 늘려서 통과시키는 것은 이 태스크의 목적을 무효화한다.

- [ ] **Step 7: 전체 테스트 실행**

Run: `python -m pytest -q`
Expected: 모두 통과

- [ ] **Step 8: 커밋**

```bash
git add ebl_gap/measure.py tests/test_measure.py tests/test_accuracy.py
git commit -m "feat: add measure_roi orchestration with accuracy regression gate"
```

---

### Task 9: FEI TIFF 메타데이터에서 스케일 읽기

**Files:**
- Create: `ebl_gap/metadata.py`
- Test: `tests/test_metadata.py`

**Interfaces:**
- Consumes: `ScaleInfo` (Task 1)
- Produces: `MetadataNotFoundError(Exception)`, `parse_ini(text) -> dict[str, dict[str, str]]`, `read_fei_metadata(path) -> dict[str, dict[str, str]]`, `scale_from_metadata(meta, image_width) -> tuple[ScaleInfo, list[str]]`, `databar_top_row(meta, image_height) -> int | None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_metadata.py`:

```python
import numpy as np
import pytest
import tifffile

from ebl_gap.metadata import (
    MetadataNotFoundError,
    databar_top_row,
    parse_ini,
    read_fei_metadata,
    scale_from_metadata,
)

FEI_INI = """[User]
Date=09/15/2026
UserText=SD gap dose test
[System]
Type=SEM
SystemType=Inspect F
[Beam]
HV=20000
Spot=3.0
[EBeam]
HV=20000
HorFieldsize=3.125e-006
[Scan]
InternalScan=true
Dwelltime=1e-005
PixelWidth=3.0517578125e-009
PixelHeight=3.0517578125e-009
HorFieldsize=3.125e-006
[Image]
ResolutionX=1024
ResolutionY=884
DigitalContrast=1.0
"""


def write_fei_tiff(path, ini=FEI_INI, height=943, width=1024):
    data = np.zeros((height, width), dtype=np.uint8)
    # 34682 = FEI_HELIOS. tifffile이 이 태그에서 INI 블록을 읽는다.
    tifffile.imwrite(path, data,
                     extratags=[(34682, 's', 0, ini, True)])
    return path


def test_parse_ini_splits_sections_and_keys():
    meta = parse_ini(FEI_INI)
    assert meta["Scan"]["PixelWidth"] == "3.0517578125e-009"
    assert meta["System"]["SystemType"] == "Inspect F"
    assert meta["Image"]["ResolutionY"] == "884"


def test_parse_ini_ignores_junk_lines_instead_of_raising():
    meta = parse_ini("garbage line\n[Scan]\nPixelWidth=1e-9\n\n\n")
    assert meta["Scan"]["PixelWidth"] == "1e-9"


def test_scale_from_metadata_converts_metres_to_nanometres():
    scale, warnings = scale_from_metadata(parse_ini(FEI_INI), image_width=1024)
    assert scale.nm_per_px == pytest.approx(3.0517578125)
    assert scale.source == "fei_metadata"
    assert warnings == []


def test_scale_from_metadata_warns_when_pixel_width_disagrees_with_field_width():
    """이미지가 크롭되거나 리사이즈된 경우를 잡아내는 검증 장치."""
    meta = parse_ini(FEI_INI)
    meta["Scan"]["HorFieldsize"] = "6.0e-006"  # 실제의 약 2배
    _, warnings = scale_from_metadata(meta, image_width=1024)
    assert any("HFW" in w for w in warnings)


def test_scale_from_metadata_raises_without_pixel_width():
    with pytest.raises(MetadataNotFoundError, match="PixelWidth"):
        scale_from_metadata({"Scan": {"Dwelltime": "1e-005"}}, image_width=1024)


def test_databar_top_row_is_the_image_height_from_metadata():
    assert databar_top_row(parse_ini(FEI_INI), image_height=943) == 884


def test_databar_top_row_is_none_when_resolution_is_absent():
    assert databar_top_row({"Scan": {}}, image_height=943) is None


def test_databar_top_row_is_none_when_there_is_no_databar():
    meta = parse_ini(FEI_INI)
    assert databar_top_row(meta, image_height=884) is None


def test_read_fei_metadata_round_trips_through_a_real_tiff(tmp_path):
    path = write_fei_tiff(tmp_path / "sd_gap_320uC.tif")
    meta = read_fei_metadata(path)
    assert meta["Scan"]["PixelWidth"] == "3.0517578125e-009"


def test_parse_ini_strips_null_byte_padding():
    """실제 FEI 파일은 INI 블록 끝을 널 바이트로 채운다.

    값에 붙은 널을 떼지 않으면 float() 변환이 실패하고, PixelWidth가 파일에
    분명히 있는데도 MetadataNotFoundError가 난다. 픽스처만으로는 안 드러나는
    실제 하드웨어 출력의 차이다.
    """
    meta = parse_ini("[Scan]\nPixelWidth=3.0e-009" + "\x00" * 20)
    assert meta["Scan"]["PixelWidth"] == "3.0e-009"
    assert float(meta["Scan"]["PixelWidth"]) == pytest.approx(3.0e-9)


def test_tag_fallback_works_when_fei_metadata_is_empty(tmp_path, monkeypatch):
    """tifffile이 fei_metadata를 비워 줄 때 태그 직접 읽기가 실제로 동작해야 한다.

    이 경로는 실제 Inspect F 파일이 픽스처와 다를 때를 위한 안전망인데,
    강제로 발동시키는 테스트가 없으면 죽은 코드인지 알 수 없다.
    """
    path = write_fei_tiff(tmp_path / "fallback_300uC.tif")
    monkeypatch.setattr(tifffile.TiffFile, "fei_metadata",
                        property(lambda self: None))
    meta = read_fei_metadata(path)
    assert float(meta["Scan"]["PixelWidth"]) == pytest.approx(3.0517578125e-9)


def test_read_fei_metadata_raises_on_a_plain_tiff(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(MetadataNotFoundError, match="FEI"):
        read_fei_metadata(path)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_metadata.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.metadata'`

- [ ] **Step 3: `ebl_gap/metadata.py` 구현**

```python
"""FEI TIFF에 내장된 INI 메타데이터를 읽어 스케일을 확정한다.

FEI/Thermo 장비는 TIFF 태그 34680(SFEG) 또는 34682(Helios)에 INI 형식 텍스트를
넣는다. [Scan] PixelWidth가 미터 단위 픽셀 크기이므로, 데이터바의 스케일바를 픽셀로
재서 추정하는 것보다 정확하다.
"""

from __future__ import annotations

from pathlib import Path

import tifffile

from ebl_gap.types import ScaleInfo

FEI_TAG_CODES = (34680, 34682, 34683)

#: PixelWidth * 이미지 가로폭과 HFW가 이 비율 이상 어긋나면 경고한다.
HFW_MISMATCH_TOLERANCE = 0.05


class MetadataNotFoundError(RuntimeError):
    """필요한 메타데이터가 없을 때."""


def parse_ini(text: str) -> dict[str, dict[str, str]]:
    """FEI의 INI 블록을 섹션 딕셔너리로 파싱한다.

    configparser를 쓰지 않는 이유는 FEI 블록에 중복 키와 빈 섹션이 섞여 있어
    strict 모드에서 예외가 나기 때문이다. 알아볼 수 없는 줄은 조용히 버린다.
    """
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw in text.replace("\r\n", "\n").split("\n"):
        # 실제 FEI 파일은 INI 블록 끝을 널 바이트로 채운다. 줄 앞뒤 어디에 붙든
        # 제거해야 한다 — 값에 붙은 널을 남기면 float() 변환이 실패하고,
        # PixelWidth가 파일에 분명히 있는데도 MetadataNotFoundError가 난다.
        line = raw.replace("\x00", "").strip()
        if not line or line.startswith((";", "#")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
            continue
        if current is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        current[key.strip()] = value.strip()
    return sections


def read_fei_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    """TIFF에서 FEI 메타데이터 섹션을 읽는다."""
    with tifffile.TiffFile(str(path)) as tif:
        meta = getattr(tif, "fei_metadata", None)
        if meta:
            return {str(k): {str(kk): str(vv) for kk, vv in v.items()}
                    for k, v in meta.items() if isinstance(v, dict)}
        for page in tif.pages:
            for code in FEI_TAG_CODES:
                tag = page.tags.get(code)
                if tag is None:
                    continue
                value = tag.value
                if isinstance(value, bytes):
                    value = value.decode("latin-1", errors="replace")
                if isinstance(value, dict):
                    # tifffile 버전에 따라 태그를 이미 섹션 dict로 파싱해서 준다.
                    # 이 경우를 처리하지 않으면 폴백이 통째로 죽은 코드가 된다.
                    parsed = {
                        str(k): {str(kk): str(vv) for kk, vv in v.items()}
                        for k, v in value.items()
                        if isinstance(v, dict)
                    }
                    if parsed:
                        return parsed
                    continue
                if isinstance(value, str) and "[" in value:
                    parsed = parse_ini(value)
                    if parsed:
                        return parsed
    raise MetadataNotFoundError(
        f"FEI 메타데이터를 찾지 못했다: {path}. "
        "스케일바 자동 검출 또는 수동 캘리브레이션을 쓴다."
    )


def _first_float(meta: dict[str, dict[str, str]], candidates) -> float | None:
    for section, key in candidates:
        raw = meta.get(section, {}).get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def scale_from_metadata(meta: dict[str, dict[str, str]],
                        image_width: int) -> tuple[ScaleInfo, list[str]]:
    """PixelWidth에서 ScaleInfo를 만들고 HFW와 교차 검증한다."""
    pixel_width_m = _first_float(meta, [("Scan", "PixelWidth"),
                                        ("EScan", "PixelWidth")])
    if pixel_width_m is None or pixel_width_m <= 0:
        raise MetadataNotFoundError(
            "메타데이터에 PixelWidth가 없거나 값이 올바르지 않다"
        )

    nm_per_px = pixel_width_m * 1e9
    warnings: list[str] = []

    hfw_m = _first_float(meta, [("Scan", "HorFieldsize"),
                                ("EScan", "HorFieldsize"),
                                ("EBeam", "HorFieldsize")])
    if hfw_m and hfw_m > 0:
        expected_nm = hfw_m * 1e9
        actual_nm = nm_per_px * image_width
        if abs(actual_nm - expected_nm) / expected_nm > HFW_MISMATCH_TOLERANCE:
            warnings.append(
                f"PixelWidth x 가로폭({actual_nm:.1f} nm)이 HFW"
                f"({expected_nm:.1f} nm)와 어긋난다. 이미지가 크롭 또는 "
                "리사이즈되었을 수 있으니 수동 캘리브레이션을 권장한다"
            )

    return ScaleInfo(nm_per_px=nm_per_px, source="fei_metadata"), warnings


def databar_top_row(meta: dict[str, dict[str, str]],
                    image_height: int) -> int | None:
    """데이터바가 시작되는 행 번호. 데이터바가 없으면 None.

    FEI는 [Image] ResolutionY에 데이터바를 뺀 실제 스캔 높이를 적는다. TIFF 전체
    높이가 그보다 크면 차이만큼이 데이터바다.
    """
    raw = meta.get("Image", {}).get("ResolutionY")
    if raw is None:
        return None
    try:
        scan_height = int(float(raw))
    except ValueError:
        return None
    if 0 < scan_height < image_height:
        return scan_height
    return None
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_metadata.py -v`
Expected: PASS (12 passed)

`test_read_fei_metadata_round_trips_through_a_real_tiff`가 실패하면, 설치된 tifffile 버전이 `fei_metadata`를 어떻게 노출하는지 확인한다: `python -c "import tifffile; print(tifffile.__version__)"`. 태그 직접 읽기 폴백이 있으므로 `fei_metadata`가 None이어도 통과해야 한다.

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap/metadata.py tests/test_metadata.py
git commit -m "feat: read pixel size from FEI TIFF metadata with HFW cross-check"
```

---

### Task 10: 스케일바 자동 검출과 수동 캘리브레이션

메타데이터가 없는 이미지(PNG 변환본, 잘라낸 이미지)를 위한 폴백이다. 실제 Inspect F 파일을 확보하기 전까지는 이 경로가 유일하게 확실히 동작하는 스케일 확정 수단이므로 반드시 만들어 둔다.

**Files:**
- Create: `ebl_gap/scalebar.py`
- Test: `tests/test_scalebar.py`

**Interfaces:**
- Consumes: `ScaleInfo` (Task 1)
- Produces: `ScalebarHit` (필드 `row`, `x0`, `x1`; 속성 `length_px`), `detect_scalebar(image, *, databar_top=None, min_length_px=20) -> ScalebarHit | None`, `scale_from_scalebar(length_px, length_nm) -> ScaleInfo`, `scale_from_two_points(p0, p1, length_nm) -> ScaleInfo`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_scalebar.py`:

```python
import numpy as np
import pytest

from ebl_gap.scalebar import (
    detect_scalebar,
    scale_from_scalebar,
    scale_from_two_points,
)


def image_with_databar(bar_length=100, bar_row=920, bar_x0=60):
    """스캔 영역 아래에 어두운 데이터바를 두고 밝은 스케일바 막대를 그린다."""
    img = np.full((943, 1024), 120.0)
    img[884:, :] = 10.0
    img[bar_row - 2:bar_row + 3, bar_x0:bar_x0 + bar_length] = 250.0
    return img


def test_detects_the_bar_and_its_pixel_length():
    hit = detect_scalebar(image_with_databar(), databar_top=884)
    assert hit is not None
    assert hit.length_px == pytest.approx(100, abs=2)
    assert 918 <= hit.row <= 922


def test_returns_none_when_the_databar_has_no_bar():
    img = np.full((943, 1024), 120.0)
    img[884:, :] = 10.0
    assert detect_scalebar(img, databar_top=884) is None


def test_returns_none_when_the_bright_run_is_too_short():
    assert detect_scalebar(image_with_databar(bar_length=8),
                           databar_top=884) is None


def test_searches_the_bottom_fifth_when_databar_top_is_unknown():
    hit = detect_scalebar(image_with_databar())
    assert hit is not None
    assert hit.length_px == pytest.approx(100, abs=2)


def test_ignores_a_run_spanning_almost_the_whole_width():
    """데이터바 자체가 밝게 반전된 이미지를 스케일바로 오인하면 안 된다."""
    img = np.full((943, 1024), 10.0)
    img[884:, :] = 250.0
    assert detect_scalebar(img, databar_top=884) is None


def test_scale_from_scalebar_divides_length_by_pixels():
    scale = scale_from_scalebar(length_px=100.0, length_nm=1000.0)
    assert scale.nm_per_px == pytest.approx(10.0)
    assert scale.source == "scalebar_auto"


def test_scale_from_two_points_uses_euclidean_distance():
    scale = scale_from_two_points((10.0, 20.0), (40.0, 60.0), length_nm=500.0)
    assert scale.nm_per_px == pytest.approx(10.0)  # 거리 50px
    assert scale.source == "manual"


def test_scale_helpers_reject_nonsense_input():
    with pytest.raises(ValueError, match="길이"):
        scale_from_scalebar(length_px=0.0, length_nm=1000.0)
    with pytest.raises(ValueError, match="길이"):
        scale_from_two_points((10.0, 10.0), (10.0, 10.0), length_nm=500.0)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_scalebar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.scalebar'`

- [ ] **Step 3: `ebl_gap/scalebar.py` 구현**

```python
"""메타데이터가 없는 이미지를 위한 스케일 확정 폴백.

데이터바에서 스케일바 막대를 찾아 픽셀 길이를 재거나, 사용자가 직접 두 점을 찍게
한다. 막대가 몇 나노미터인지는 OCR로 읽지 않고 사용자에게 묻는다 — 잘못 읽은 숫자로
모든 측정값이 조용히 틀어지는 것보다 한 번 묻는 편이 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ebl_gap.types import ScaleInfo

#: 전체 가로폭의 이 비율을 넘는 밝은 구간은 스케일바가 아니라 배경으로 본다.
MAX_BAR_WIDTH_RATIO = 0.9


@dataclass(frozen=True)
class ScalebarHit:
    """검출된 스케일바 막대의 위치."""

    row: int
    x0: int
    x1: int

    @property
    def length_px(self) -> int:
        return self.x1 - self.x0 + 1


def _longest_bright_run(mask: np.ndarray) -> tuple[int, int, int]:
    """불리언 행에서 가장 긴 True 구간의 (길이, 시작, 끝)을 구한다."""
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    if changes.size == 0:
        return 0, 0, 0
    starts, ends = changes[0::2], changes[1::2]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(lengths[best]), int(starts[best]), int(ends[best] - 1)


def detect_scalebar(image, *, databar_top: int | None = None,
                    min_length_px: int = 20) -> ScalebarHit | None:
    """데이터바 영역에서 가장 긴 밝은 수평 막대를 찾는다."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"이미지는 2차원이어야 한다 (받은 차원: {img.ndim})")

    top = databar_top if databar_top is not None else int(img.shape[0] * 0.8)
    top = max(0, min(top, img.shape[0] - 1))
    region = img[top:, :]
    if region.size == 0:
        return None

    span = float(region.max() - region.min())
    if span <= 0:
        return None
    bright = region > (region.min() + 0.5 * span)

    max_width = int(img.shape[1] * MAX_BAR_WIDTH_RATIO)
    best: ScalebarHit | None = None
    best_length = 0
    for offset, row_mask in enumerate(bright):
        length, x0, x1 = _longest_bright_run(row_mask)
        if length < min_length_px or length > max_width:
            continue
        if length > best_length:
            best_length = length
            best = ScalebarHit(row=top + offset, x0=x0, x1=x1)
    return best


def scale_from_scalebar(length_px: float, length_nm: float) -> ScaleInfo:
    """검출된 막대 길이와 사용자가 입력한 실제 길이로 스케일을 만든다."""
    if length_px <= 0 or length_nm <= 0:
        raise ValueError(
            f"길이는 양수여야 한다: {length_px} px, {length_nm} nm"
        )
    return ScaleInfo(nm_per_px=length_nm / length_px, source="scalebar_auto")


def scale_from_two_points(p0, p1, length_nm: float) -> ScaleInfo:
    """사용자가 찍은 두 점 사이 거리와 실제 길이로 스케일을 만든다."""
    distance_px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    if distance_px <= 0 or length_nm <= 0:
        raise ValueError(
            f"길이는 양수여야 한다: {distance_px:.3f} px, {length_nm} nm"
        )
    return ScaleInfo(nm_per_px=length_nm / distance_px, source="manual")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_scalebar.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap/scalebar.py tests/test_scalebar.py
git commit -m "feat: add scalebar detection and manual calibration fallbacks"
```

---

### Task 11: 세션 관리와 dose 집계

**Files:**
- Create: `ebl_gap/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `ImageRecord`, `RoiResult`, `ScaleInfo` (Task 1)
- Produces: `DEFAULT_DOSE_PATTERN`, `parse_dose(filename, pattern=DEFAULT_DOSE_PATTERN) -> float | None`, `DosePoint` (필드 `dose`, `mean_nm`, `std_nm`, `n_valid`, `n_short`, `path`), `Session` (메서드 `add(record)`, `dose_curve()`, `scale_warnings()`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_dataset.py`:

```python
from pathlib import Path

import pytest

from ebl_gap.dataset import DEFAULT_DOSE_PATTERN, Session, parse_dose
from ebl_gap.types import ImageRecord, RoiResult, ScaleInfo

SCALE = ScaleInfo(nm_per_px=3.0, source="fei_metadata")


def roi_result(mean_nm, n_valid=100, n_short=0, scale=SCALE):
    return RoiResult(mean_nm=mean_nm, std_nm=1.0, n_valid=n_valid,
                     n_short=n_short, n_uncertain=0, n_low_confidence=0,
                     angle_deg=0.0, lines=(), warnings=(), scale=scale)


def record(name, dose, results, scale=SCALE):
    return ImageRecord(path=Path(name), scale=scale, dose=dose,
                       roi_results=list(results))


@pytest.mark.parametrize("name,expected", [
    ("pattern_320uC_01.tif", 320.0),
    ("SD_gap_280uc.tif", 280.0),
    ("dose-test-412.5uC-run2.tif", 412.5),
    ("300 uC scan.tif", 300.0),
    ("no_dose_here.tif", None),
    ("run_01.tif", None),
])
def test_parse_dose_reads_the_microcoulomb_figure(name, expected):
    assert parse_dose(name) == expected


def test_parse_dose_accepts_a_custom_pattern():
    assert parse_dose("d0450_scan.tif", pattern=r"d(\d+)") == 450.0


def test_parse_dose_returns_none_on_an_unmatched_custom_pattern():
    assert parse_dose("pattern_320uC.tif", pattern=r"z(\d+)") is None


def test_default_pattern_is_case_insensitive():
    assert "(?i)" in DEFAULT_DOSE_PATTERN


def test_dose_curve_is_sorted_by_dose():
    session = Session()
    session.add(record("c.tif", 400.0, [roi_result(30.0)]))
    session.add(record("a.tif", 200.0, [roi_result(80.0)]))
    session.add(record("b.tif", 300.0, [roi_result(55.0)]))
    assert [p.dose for p in session.dose_curve()] == [200.0, 300.0, 400.0]
    assert [p.mean_nm for p in session.dose_curve()] == [80.0, 55.0, 30.0]


def test_dose_curve_skips_records_without_a_dose():
    session = Session()
    session.add(record("a.tif", None, [roi_result(80.0)]))
    session.add(record("b.tif", 300.0, [roi_result(55.0)]))
    assert [p.dose for p in session.dose_curve()] == [300.0]


def test_dose_curve_skips_records_with_no_measurement():
    session = Session()
    session.add(record("a.tif", 300.0, []))
    session.add(record("b.tif", 400.0, [roi_result(None, n_valid=0, n_short=300)]))
    assert session.dose_curve() == []


def test_multiple_rois_are_averaged_weighted_by_valid_line_count():
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(40.0, n_valid=100),
                                        roi_result(60.0, n_valid=300)]))
    point = session.dose_curve()[0]
    assert point.mean_nm == pytest.approx(55.0)  # (40*100 + 60*300) / 400
    assert point.n_valid == 400


def test_short_counts_are_summed_across_rois():
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(40.0, n_short=5),
                                        roi_result(60.0, n_short=7)]))
    assert session.dose_curve()[0].n_short == 12


def test_mixed_pixel_sizes_produce_a_warning():
    coarse = ScaleInfo(nm_per_px=6.0, source="fei_metadata")
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(40.0)], scale=SCALE))
    session.add(record("b.tif", 400.0, [roi_result(60.0, scale=coarse)],
                       scale=coarse))
    assert any("배율" in w for w in session.scale_warnings())


def test_consistent_pixel_sizes_produce_no_warning():
    session = Session()
    session.add(record("a.tif", 300.0, [roi_result(40.0)]))
    session.add(record("b.tif", 400.0, [roi_result(60.0)]))
    assert session.scale_warnings() == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.dataset'`

- [ ] **Step 3: `ebl_gap/dataset.py` 구현**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap/dataset.py tests/test_dataset.py
git commit -m "feat: add session management with dose parsing and dose-gap aggregation"
```

---

### Task 12: CSV, 오버레이 PNG, 요약 리포트 내보내기

**Files:**
- Create: `ebl_gap/export.py`
- Modify: `ebl_gap/profile.py` (좌표 역변환 헬퍼 추가)
- Test: `tests/test_export.py`
- Test: `tests/test_profile.py` (역변환 테스트 추가)

**Interfaces:**
- Consumes: Task 1, 4, 11
- Produces: `aligned_to_image(roi, angle_deg, u_px, v_px) -> tuple[float, float]` (profile.py), `write_summary_csv(path, session)`, `write_lines_csv(path, record, roi_index)`, `render_overlay(image, roi, result) -> np.ndarray`, `write_overlay_png(path, image, roi, result)`, `format_report(session) -> str`

- [ ] **Step 1: `profile.py`에 좌표 역변환 테스트 추가**

`tests/test_profile.py` 끝에 덧붙인다:

```python
def test_aligned_to_image_inverts_the_sampling_transform():
    from ebl_gap.profile import aligned_to_image

    roi = Roi(100, 50, 299, 249)
    for angle in (0.0, 7.0, -12.0):
        prof = extract_profiles(np.arange(300 * 400, dtype=float).reshape(300, 400),
                                roi, angle)
        # 정렬 좌표 (u, v)에서 뽑은 값과 역변환한 이미지 좌표에서 읽은 값이 같아야 한다.
        u_px, v_px = 37, 91
        x, y = aligned_to_image(roi, angle, u_px, v_px)
        img = np.arange(300 * 400, dtype=float).reshape(300, 400)
        from scipy.ndimage import map_coordinates
        direct = map_coordinates(img, [[y], [x]], order=1, mode="nearest")[0]
        assert prof[v_px, u_px] == pytest.approx(direct, rel=1e-9)


def test_aligned_to_image_centre_maps_to_roi_centre():
    from ebl_gap.profile import aligned_to_image

    roi = Roi(100, 50, 299, 249)
    x, y = aligned_to_image(roi, 15.0, (roi.width - 1) / 2.0,
                            (roi.height - 1) / 2.0)
    assert (x, y) == pytest.approx((roi.cx, roi.cy))
```

- [ ] **Step 2: `profile.py`에 `aligned_to_image` 추가**

`ebl_gap/profile.py` 끝에 덧붙인다:

```python
def aligned_to_image(roi: Roi, angle_deg: float, u_px: float,
                     v_px: float) -> tuple[float, float]:
    """정렬 좌표계의 (열, 행)을 원본 이미지 좌표 (x, y)로 되돌린다.

    extract_profiles가 쓰는 변환과 반드시 같은 식이어야 한다. 오버레이에 에지를
    그리려면 이 역변환이 필요하다.
    """
    a_rad = np.radians(float(angle_deg))
    u = float(u_px) - (roi.width - 1) / 2.0
    v = float(v_px) - (roi.height - 1) / 2.0
    x = roi.cx + u * np.cos(a_rad) + v * np.sin(a_rad)
    y = roi.cy - u * np.sin(a_rad) + v * np.cos(a_rad)
    return float(x), float(y)
```

- [ ] **Step 3: 역변환 테스트 통과 확인**

Run: `python -m pytest tests/test_profile.py -v`
Expected: PASS (12 passed)

- [ ] **Step 4: 실패하는 내보내기 테스트 작성**

`tests/test_export.py`:

```python
import csv
from pathlib import Path

import numpy as np
import pytest

from ebl_gap.dataset import Session
from ebl_gap.export import (
    format_report,
    render_overlay,
    write_lines_csv,
    write_overlay_png,
    write_summary_csv,
)
from ebl_gap.measure import measure_roi
from ebl_gap.types import ImageRecord, Roi, ScaleInfo
from tests.synth import synth_gap_image

ROI = Roi(106, 106, 405, 405)
SCALE = ScaleInfo(nm_per_px=3.0, source="fei_metadata")


def measured_session(tmp_path):
    img = synth_gap_image(gap_nm=120.0, nm_per_px=3.0, angle_deg=4.0)
    result = measure_roi(img, ROI, SCALE)
    session = Session()
    session.add(ImageRecord(path=tmp_path / "pattern_320uC.tif", scale=SCALE,
                            dose=320.0, roi_results=[result]))
    return session, img, result


def test_summary_csv_has_one_row_per_roi_with_scale_provenance(tmp_path):
    session, _, result = measured_session(tmp_path)
    out = tmp_path / "summary.csv"
    write_summary_csv(out, session)
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert len(rows) == 1
    row = rows[0]
    assert row["file"] == "pattern_320uC.tif"
    assert float(row["dose_uC"]) == pytest.approx(320.0)
    assert float(row["nm_per_px"]) == pytest.approx(3.0)
    assert row["scale_source"] == "fei_metadata"
    # CSV는 mean_nm을 소수 3자리로 쓴다. 허용오차는 그 반올림 단위(5e-4)여야
    # 한다. 더 조이면 구조적으로 실패하고, CSV 자릿수를 늘려 맞추는 것은
    # 측정이 갖지 않은 정밀도를 보고하는 셈이 된다 — 1~3 nm/px 해상도에서
    # 0.001 nm는 이미 물리적 의미보다 두 자릿수 이상 세밀하다.
    assert float(row["mean_nm"]) == pytest.approx(result.mean_nm, abs=5e-4)
    assert int(row["n_valid"]) == result.n_valid


def test_summary_csv_writes_empty_cells_for_unmeasured_images(tmp_path):
    session = Session()
    session.add(ImageRecord(path=Path("blank.tif"), scale=None, dose=None,
                            roi_results=[], error="FEI 메타데이터 없음"))
    out = tmp_path / "summary.csv"
    write_summary_csv(out, session)
    row = list(csv.DictReader(out.open(encoding="utf-8-sig")))[0]
    assert row["mean_nm"] == ""
    assert row["dose_uC"] == ""
    assert "메타데이터" in row["warnings"]


def test_lines_csv_has_one_row_per_scanline(tmp_path):
    session, _, result = measured_session(tmp_path)
    out = tmp_path / "lines.csv"
    write_lines_csv(out, session.records[0], roi_index=0)
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert len(rows) == ROI.height
    assert rows[0]["status"] in {"valid", "short", "no_edge", "multi_edge",
                                 "sub_resolution", "outlier"}
    assert float(rows[0]["width_nm"]) > 0


def test_overlay_is_rgb_and_marks_the_roi(tmp_path):
    _, img, result = measured_session(tmp_path)
    overlay = render_overlay(img, ROI, result)
    assert overlay.shape == (img.shape[0], img.shape[1], 3)
    assert overlay.dtype == np.uint8
    # ROI 테두리는 노란색으로 그린다.
    assert tuple(overlay[ROI.y0, ROI.x0]) == (255, 255, 0)


def test_overlay_draws_detected_edges_in_green(tmp_path):
    _, img, result = measured_session(tmp_path)
    overlay = render_overlay(img, ROI, result)
    green = np.all(overlay == np.array([0, 255, 0], dtype=np.uint8), axis=-1)
    # 유효 라인마다 좌우 에지 두 점씩 찍힌다.
    assert green.sum() >= result.n_valid


def test_overlay_marks_short_lines_in_red():
    img = np.full((512, 512), 200.0)
    img[:, 250:262] = 40.0
    img[300:400, 250:262] = 200.0  # 100행의 갭을 메워 short를 만든다
    result = measure_roi(img, ROI, SCALE)
    overlay = render_overlay(img, ROI, result)
    red = np.all(overlay == np.array([255, 0, 0], dtype=np.uint8), axis=-1)
    assert result.n_short > 0
    assert red.sum() > 0


def test_overlay_survives_an_roi_that_extends_past_the_image():
    """측정이 되는 ROI는 오버레이도 그려져야 한다.

    extract_profiles는 경계를 벗어난 ROI를 mode="nearest"로 허용한다. 같은 ROI로
    measure_roi가 성공했는데 오버레이만 IndexError로 터지면, CSV에는 값이 남고
    그림만 안 나오는 상태가 된다.
    """
    img = synth_gap_image(width=100, height=100, gap_nm=20.0, nm_per_px=1.0)
    roi = Roi(50, 50, 120, 120)
    result = measure_roi(img, roi, SCALE)
    overlay = render_overlay(img, roi, result)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8


def test_overlay_png_is_written_and_readable(tmp_path):
    from PIL import Image

    _, img, result = measured_session(tmp_path)
    out = tmp_path / "overlay.png"
    write_overlay_png(out, img, ROI, result)
    assert out.exists()
    with Image.open(out) as handle:
        assert handle.mode == "RGB"
        assert handle.size == (img.shape[1], img.shape[0])


def test_report_states_the_scale_source_and_the_counts(tmp_path):
    session, _, result = measured_session(tmp_path)
    text = format_report(session)
    assert "fei_metadata" in text
    assert "pattern_320uC.tif" in text
    assert f"{result.n_valid}" in text
    assert "320" in text


def test_report_surfaces_warnings(tmp_path):
    img = np.full((512, 512), 200.0)
    result = measure_roi(img, ROI, SCALE)
    session = Session()
    session.add(ImageRecord(path=Path("closed.tif"), scale=SCALE, dose=500.0,
                            roi_results=[result]))
    text = format_report(session)
    # "short"는 모든 ROI 줄에 라벨로 항상 찍히므로 그것만 보면 구현이 틀려도
    # 통과한다. 실제로 short 라인이 잡혔는지와, stats.py가 만든 경고 문구가
    # 리포트에 올라왔는지를 본다.
    assert result.n_short > 0
    assert "short 발생 구간 있음" in text
```

- [ ] **Step 5: 테스트 실패 확인**

Run: `python -m pytest tests/test_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.export'`

- [ ] **Step 6: `ebl_gap/export.py` 구현**

```python
"""측정 결과를 CSV, 오버레이 PNG, 요약 텍스트로 내보낸다.

CSV는 utf-8-sig로 쓴다. 엑셀에서 한글이 깨지지 않게 하기 위해서다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from ebl_gap.dataset import Session
from ebl_gap.profile import aligned_to_image
from ebl_gap.types import UNCERTAIN_STATUSES, ImageRecord, Roi, RoiResult

SUMMARY_COLUMNS = [
    "file", "dose_uC", "nm_per_px", "scale_source", "roi_index", "angle_deg",
    "mean_nm", "std_nm", "n_valid", "n_short", "n_uncertain",
    "n_low_confidence", "warnings",
]

LINE_COLUMNS = [
    "row", "left_px", "right_px", "width_px", "width_nm", "status", "flags",
    "reason",
]

STATUS_COLORS = {
    "valid": (0, 255, 0),
    "short": (255, 0, 0),
}
UNCERTAIN_COLOR = (255, 140, 0)
ROI_COLOR = (255, 255, 0)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_summary_csv(path: str | Path, session: Session) -> None:
    """이미지별, ROI별 요약을 한 행씩 쓴다."""
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for record in session.records:
            if not record.roi_results:
                writer.writerow({
                    "file": record.path.name,
                    "dose_uC": _fmt(record.dose, 2),
                    "nm_per_px": _fmt(record.scale.nm_per_px if record.scale else None),
                    "scale_source": record.scale.source if record.scale else "",
                    "roi_index": "",
                    "angle_deg": "", "mean_nm": "", "std_nm": "",
                    "n_valid": "", "n_short": "", "n_uncertain": "",
                    "n_low_confidence": "",
                    "warnings": record.error or "",
                })
                continue
            for index, result in enumerate(record.roi_results):
                writer.writerow({
                    "file": record.path.name,
                    "dose_uC": _fmt(record.dose, 2),
                    "nm_per_px": _fmt(result.scale.nm_per_px),
                    "scale_source": result.scale.source,
                    "roi_index": index,
                    "angle_deg": _fmt(result.angle_deg, 3),
                    "mean_nm": _fmt(result.mean_nm, 3),
                    "std_nm": _fmt(result.std_nm, 3),
                    "n_valid": result.n_valid,
                    "n_short": result.n_short,
                    "n_uncertain": result.n_uncertain,
                    "n_low_confidence": result.n_low_confidence,
                    "warnings": " | ".join(result.warnings),
                })


def write_lines_csv(path: str | Path, record: ImageRecord,
                    roi_index: int) -> None:
    """한 ROI의 스캔라인별 원시 측정값을 쓴다."""
    result = record.roi_results[roi_index]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LINE_COLUMNS)
        writer.writeheader()
        for line in result.lines:
            writer.writerow({
                "row": line.row,
                "left_px": _fmt(line.left_px, 3),
                "right_px": _fmt(line.right_px, 3),
                "width_px": _fmt(line.width_px, 3),
                "width_nm": _fmt(line.width_nm, 3),
                "status": line.status,
                "flags": " ".join(sorted(line.flags)),
                "reason": line.reason,
            })


def _to_rgb(image) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    lo, hi = float(img.min()), float(img.max())
    span = hi - lo if hi > lo else 1.0
    gray = np.clip((img - lo) / span * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _put(canvas: np.ndarray, x: float, y: float, color) -> None:
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < canvas.shape[0] and 0 <= xi < canvas.shape[1]:
        canvas[yi, xi] = color


def render_overlay(image, roi: Roi, result: RoiResult) -> np.ndarray:
    """원본 이미지 위에 ROI와 검출된 에지를 그린 RGB 배열을 만든다."""
    canvas = _to_rgb(image)

    # 테두리 좌표를 이미지 안으로 자른다. extract_profiles는 경계를 살짝 벗어난
    # ROI를 mode="nearest"로 허용하므로 같은 ROI로 measure_roi가 성공한다.
    # 여기서 자르지 않으면 측정은 되는데 오버레이만 IndexError로 죽어서,
    # CSV에는 값이 남고 그림만 안 나오는 상태가 된다.
    y0 = max(0, min(roi.y0, canvas.shape[0] - 1))
    y1 = max(0, min(roi.y1, canvas.shape[0] - 1))
    x0 = max(0, min(roi.x0, canvas.shape[1] - 1))
    x1 = max(0, min(roi.x1, canvas.shape[1] - 1))
    canvas[y0, x0 : x1 + 1] = ROI_COLOR
    canvas[y1, x0 : x1 + 1] = ROI_COLOR
    canvas[y0 : y1 + 1, x0] = ROI_COLOR
    canvas[y0 : y1 + 1, x1] = ROI_COLOR

    for line in result.lines:
        if line.status in STATUS_COLORS:
            color = STATUS_COLORS[line.status]
        elif line.status in UNCERTAIN_STATUSES:
            color = UNCERTAIN_COLOR
        else:
            continue

        if line.left_px is None or line.right_px is None:
            # 에지가 없는 라인은 정렬 좌표계의 중앙에 한 점만 찍는다.
            x, y = aligned_to_image(roi, result.angle_deg,
                                    (roi.width - 1) / 2.0, line.row)
            _put(canvas, x, y, color)
            continue

        for u in (line.left_px, line.right_px):
            x, y = aligned_to_image(roi, result.angle_deg, u, line.row)
            _put(canvas, x, y, color)
    return canvas


def write_overlay_png(path: str | Path, image, roi: Roi,
                      result: RoiResult) -> None:
    Image.fromarray(render_overlay(image, roi, result), mode="RGB").save(str(path))


def format_report(session: Session) -> str:
    """사람이 읽는 요약 텍스트."""
    lines: list[str] = ["EBL dose test 갭 측정 요약", "=" * 40, ""]

    for warning in session.scale_warnings():
        lines.append(f"[세션 경고] {warning}")
    if session.scale_warnings():
        lines.append("")

    for record in session.records:
        dose = "미상" if record.dose is None else f"{record.dose:g} uC"
        lines.append(f"- {record.path.name} (dose {dose})")
        if record.error:
            lines.append(f"    오류: {record.error}")
        if not record.roi_results:
            lines.append("    측정 결과 없음")
            lines.append("")
            continue
        for index, result in enumerate(record.roi_results):
            if result.mean_nm is None:
                head = "갭 측정 불가"
            else:
                spread = "" if result.std_nm is None else f" +- {result.std_nm:.2f}"
                head = f"갭 {result.mean_nm:.2f}{spread} nm"
            lines.append(
                f"    ROI {index}: {head} "
                f"(유효 {result.n_valid} / short {result.n_short} / "
                f"판정보류 {result.n_uncertain} 라인, 각도 {result.angle_deg:.2f}도, "
                f"{result.scale.nm_per_px:.4f} nm/px [{result.scale.source}])"
            )
            for warning in result.warnings:
                lines.append(f"        ! {warning}")
        lines.append("")

    curve = session.dose_curve()
    if curve:
        lines.append("dose - 갭 관계")
        lines.append("-" * 40)
        for point in curve:
            spread = "" if point.std_nm is None else f" +- {point.std_nm:.2f}"
            lines.append(
                f"  {point.dose:>8.1f} uC : {point.mean_nm:7.2f}{spread} nm "
                f"(유효 {point.n_valid}, short {point.n_short})"
            )
    return "\n".join(lines)
```

- [ ] **Step 7: 테스트 통과 확인**

Run: `python -m pytest tests/test_export.py -v`
Expected: PASS (10 passed)

- [ ] **Step 8: 엔진 전체 테스트 실행**

Run: `python -m pytest -q`
Expected: 모두 통과

- [ ] **Step 9: 커밋**

```bash
git add ebl_gap/export.py ebl_gap/profile.py tests/test_export.py tests/test_profile.py
git commit -m "feat: add CSV, overlay PNG and text report export"
```

---

### Task 13: 이미지 로딩 파이프라인

GUI가 파일을 열 때 필요한 모든 것 — 픽셀, 스케일, dose, 데이터바 위치 — 을 한 번에 돌려주는 엔진 함수다. GUI에 이 로직이 들어가면 테스트할 수 없게 된다.

**Files:**
- Create: `ebl_gap/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: Task 1, 9, 11
- Produces: `LoadedImage` (필드 `record: ImageRecord`, `pixels: np.ndarray`, `databar_top: int | None`), `load_image(path, *, dose_pattern=DEFAULT_DOSE_PATTERN) -> LoadedImage`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_loader.py`:

```python
import numpy as np
import pytest
import tifffile
from PIL import Image

from ebl_gap.loader import load_image
from tests.test_metadata import FEI_INI


def write_fei_tiff(path, height=943, width=1024, value=90):
    data = np.full((height, width), value, dtype=np.uint8)
    tifffile.imwrite(path, data, extratags=[(34682, 's', 0, FEI_INI, True)])
    return path


def test_loads_pixels_scale_and_dose_from_an_fei_tiff(tmp_path):
    path = write_fei_tiff(tmp_path / "pattern_320uC_01.tif")
    loaded = load_image(path)
    assert loaded.pixels.shape == (943, 1024)
    assert loaded.pixels.dtype == np.float64
    assert loaded.record.scale.nm_per_px == pytest.approx(3.0517578125)
    assert loaded.record.scale.source == "fei_metadata"
    assert loaded.record.dose == pytest.approx(320.0)
    assert loaded.databar_top == 884
    assert loaded.record.error is None


def test_records_an_error_instead_of_raising_when_metadata_is_missing(tmp_path):
    path = tmp_path / "plain_280uC.tif"
    tifffile.imwrite(path, np.zeros((64, 64), dtype=np.uint8))
    loaded = load_image(path)
    assert loaded.record.scale is None
    assert "메타데이터" in loaded.record.error
    assert loaded.record.dose == pytest.approx(280.0)  # dose는 여전히 읽힌다
    assert loaded.pixels.shape == (64, 64)


def test_reads_a_png_through_pillow(tmp_path):
    path = tmp_path / "crop_300uC.png"
    Image.fromarray(np.full((32, 48), 200, dtype=np.uint8)).save(path)
    loaded = load_image(path)
    assert loaded.pixels.shape == (32, 48)
    assert loaded.record.scale is None
    assert loaded.databar_top is None


def test_converts_rgb_input_to_greyscale(tmp_path):
    path = tmp_path / "rgb.png"
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    Image.fromarray(rgb, mode="RGB").save(path)
    loaded = load_image(path)
    assert loaded.pixels.ndim == 2
    assert loaded.pixels.max() > 0


def test_hfw_mismatch_warning_reaches_the_record(tmp_path):
    bad_ini = FEI_INI.replace("HorFieldsize=3.125e-006", "HorFieldsize=9.0e-006")
    path = tmp_path / "mismatch_300uC.tif"
    tifffile.imwrite(path, np.zeros((943, 1024), dtype=np.uint8),
                     extratags=[(34682, 's', 0, bad_ini, True)])
    loaded = load_image(path)
    assert loaded.record.scale is not None
    assert "HFW" in loaded.record.error


def test_custom_dose_pattern_is_honoured(tmp_path):
    path = write_fei_tiff(tmp_path / "d0450_run.tif")
    loaded = load_image(path, dose_pattern=r"d(\d+)")
    assert loaded.record.dose == pytest.approx(450.0)


def test_infinite_resolution_does_not_raise(tmp_path):
    """TIFF도 읽히고 FEI 태그도 파싱되는데 필드 값만 이상한 경우.

    int(float("inf"))는 ValueError가 아니라 OverflowError를 던지므로
    databar_top_row 안의 좁은 except를 빠져나간다. 한 장 때문에 폴더 전체 스캔이
    멈추면 안 된다.
    """
    ini = FEI_INI.replace("ResolutionY=884", "ResolutionY=inf")
    path = tmp_path / "inf_300uC.tif"
    tifffile.imwrite(path, np.zeros((943, 1024), dtype=np.uint8),
                     extratags=[(34682, 's', 0, ini, True)])
    loaded = load_image(path)
    assert loaded.databar_top is None
    assert "데이터바" in loaded.record.error
    # 스케일은 정상이므로 측정은 계속할 수 있어야 한다.
    assert loaded.record.scale is not None


def test_nan_pixel_width_does_not_raise(tmp_path):
    """PixelWidth=nan은 float()을 통과하고 <= 0 검사도 통과한다.

    NaN 비교는 항상 거짓이므로 가드를 지나쳐 ScaleInfo가 ValueError를 던지는데,
    그것은 MetadataNotFoundError가 아니라 좁은 except를 빠져나간다.
    """
    ini = FEI_INI.replace("PixelWidth=3.0517578125e-009", "PixelWidth=nan")
    path = tmp_path / "nan_300uC.tif"
    tifffile.imwrite(path, np.zeros((943, 1024), dtype=np.uint8),
                     extratags=[(34682, 's', 0, ini, True)])
    loaded = load_image(path)
    assert loaded.record.scale is None
    assert "스케일" in loaded.record.error


def test_unreadable_file_produces_a_record_with_an_error(tmp_path):
    path = tmp_path / "broken.tif"
    path.write_bytes(b"not a tiff at all")
    loaded = load_image(path)
    assert loaded.record.error is not None
    assert loaded.pixels.size == 0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap.loader'`

- [ ] **Step 3: `ebl_gap/loader.py` 구현**

```python
"""이미지 한 장을 열어 픽셀, 스케일, dose, 데이터바 위치를 한 번에 돌려준다.

파일을 못 읽거나 메타데이터가 없어도 예외를 던지지 않고 ImageRecord.error에 사유를
적는다. 폴더를 통째로 여는 도중 한 장 때문에 전체가 멈추면 안 되기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ebl_gap.dataset import DEFAULT_DOSE_PATTERN, parse_dose
from ebl_gap.metadata import (
    MetadataNotFoundError,
    databar_top_row,
    read_fei_metadata,
    scale_from_metadata,
)
from ebl_gap.types import ImageRecord

TIFF_SUFFIXES = {".tif", ".tiff"}


def _read_pixels(path: Path) -> np.ndarray:
    if path.suffix.lower() in TIFF_SUFFIXES:
        import tifffile

        data = tifffile.imread(str(path))
    else:
        from PIL import Image

        with Image.open(path) as handle:
            data = np.asarray(handle)

    array = np.asarray(data)
    if array.ndim == 3:
        # RGB/RGBA는 ITU-R 601 가중치로 회색조로 바꾼다.
        rgb = array[..., :3].astype(np.float64)
        array = rgb @ np.array([0.299, 0.587, 0.114])
    if array.ndim != 2:
        raise ValueError(f"2차원 이미지가 아니다 (차원 {array.ndim})")
    return np.ascontiguousarray(array, dtype=np.float64)


@dataclass(frozen=True)
class LoadedImage:
    """파일 한 장에서 읽어낸 모든 것."""

    record: ImageRecord
    pixels: np.ndarray
    databar_top: int | None


def load_image(path: str | Path, *,
               dose_pattern: str = DEFAULT_DOSE_PATTERN) -> LoadedImage:
    """이미지를 읽고 가능한 만큼 메타데이터를 채운다."""
    path = Path(path)
    record = ImageRecord(path=path, dose=parse_dose(path.name, dose_pattern))

    try:
        pixels = _read_pixels(path)
    except Exception as exc:  # 손상된 파일, 지원하지 않는 포맷 등
        record.error = f"이미지를 읽지 못했다: {exc}"
        return LoadedImage(record=record, pixels=np.empty((0, 0)),
                           databar_top=None)

    databar_top: int | None = None
    try:
        meta = read_fei_metadata(path)
    except MetadataNotFoundError as exc:
        record.error = str(exc)
        return LoadedImage(record=record, pixels=pixels, databar_top=None)
    except Exception as exc:
        record.error = f"메타데이터를 읽지 못했다: {exc}"
        return LoadedImage(record=record, pixels=pixels, databar_top=None)

    # 아래 두 호출도 각각 감싼다. 둘 다 "TIFF는 읽히고 FEI 태그도 파싱되는데
    # 필드 값만 이상한" 경우에 터지고, 그 예외는 MetadataNotFoundError가 아니다:
    #   ResolutionY=inf  -> int(float("inf"))가 OverflowError
    #   PixelWidth=nan   -> float()을 통과하고 <= 0 검사도 통과(NaN 비교는 항상
    #                       거짓)한 뒤 ScaleInfo가 ValueError
    # 한 장 때문에 폴더 전체 스캔이 멈추면 안 된다는 것이 이 함수의 존재 이유다.
    notes: list[str] = []

    try:
        databar_top = databar_top_row(meta, pixels.shape[0])
    except Exception as exc:
        databar_top = None
        notes.append(f"데이터바 위치를 읽지 못했다: {exc}")

    try:
        scale, warnings = scale_from_metadata(meta, pixels.shape[1])
    except MetadataNotFoundError as exc:
        notes.append(str(exc))
        record.error = " | ".join(notes)
        return LoadedImage(record=record, pixels=pixels, databar_top=databar_top)
    except Exception as exc:
        notes.append(f"스케일을 계산하지 못했다: {exc}")
        record.error = " | ".join(notes)
        return LoadedImage(record=record, pixels=pixels, databar_top=databar_top)

    record.scale = scale
    notes.extend(warnings)
    if notes:
        record.error = " | ".join(notes)
    return LoadedImage(record=record, pixels=pixels, databar_top=databar_top)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_loader.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap/loader.py tests/test_loader.py
git commit -m "feat: add image loading pipeline with graceful metadata fallback"
```

---

### Task 14: GUI 뼈대와 드래그 ROI 이미지 뷰

여기서부터 PySide6가 등장한다. `ebl_gap/` 아래에는 여전히 Qt가 한 줄도 들어가지 않는다.

GUI도 테스트한다. `QT_QPA_PLATFORM=offscreen`을 쓰면 화면 없이 위젯을 만들고 조작할 수 있다. 그리기 결과는 검증하지 않지만, 위젯이 만들어지는지와 엔진 호출 배선이 맞는지는 확인할 수 있다.

**Files:**
- Create: `ebl_gap_gui/__init__.py`
- Create: `ebl_gap_gui/image_view.py`
- Create: `tests/conftest.py`
- Test: `tests/test_gui_image_view.py`

**Interfaces:**
- Consumes: `Roi` (Task 1), `render_overlay` (Task 12)
- Produces: `ImageView(QWidget)` — 메서드 `set_image(pixels)`, `current_roi() -> Roi | None`, `set_roi(roi)`, `show_overlay(rgb)`, `clear_overlay()`; 시그널 `roi_changed = Signal()`

- [ ] **Step 1: `tests/conftest.py` 작성**

```python
"""GUI 테스트를 화면 없이 돌리기 위한 설정.

QT_QPA_PLATFORM은 PySide6가 import되기 전에 설정되어야 한다. conftest.py는 테스트
수집 전에 실행되므로 여기가 맞는 자리다.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """세션 전체가 공유하는 QApplication. Qt는 프로세스당 하나만 허용한다."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_gui_image_view.py`:

```python
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from ebl_gap.types import Roi  # noqa: E402
from ebl_gap_gui.image_view import ImageView  # noqa: E402
from tests.synth import synth_gap_image  # noqa: E402


def test_view_starts_with_no_roi(qapp):
    view = ImageView()
    assert view.current_roi() is None


def test_setting_an_image_creates_a_default_centred_roi(qapp):
    view = ImageView()
    view.set_image(synth_gap_image(width=512, height=512))
    roi = view.current_roi()
    assert roi is not None
    assert 0 <= roi.x0 < roi.x1 < 512
    assert 0 <= roi.y0 < roi.y1 < 512


def test_set_roi_round_trips(qapp):
    view = ImageView()
    view.set_image(synth_gap_image(width=512, height=512))
    view.set_roi(Roi(100, 120, 300, 340))
    roi = view.current_roi()
    assert (roi.x0, roi.y0, roi.x1, roi.y1) == (100, 120, 300, 340)


def test_roi_is_clamped_to_the_image_bounds(qapp):
    view = ImageView()
    view.set_image(np.zeros((100, 100)))
    view.set_roi(Roi(-50, -50, 400, 400))
    roi = view.current_roi()
    assert roi.x0 >= 0 and roi.y0 >= 0
    assert roi.x1 <= 99 and roi.y1 <= 99


def test_setting_the_roi_programmatically_emits_roi_changed(qapp):
    view = ImageView()
    view.set_image(synth_gap_image(width=512, height=512))
    seen = []
    view.roi_changed.connect(lambda: seen.append(1))
    view.set_roi(Roi(10, 10, 200, 200))
    assert seen


def test_dragging_the_roi_emits_roi_changed(qapp):
    """마우스 드래그가 타는 경로를 직접 확인한다.

    set_roi()는 자기 본문에서 roi_changed를 명시적으로 발신하므로, 그것만
    테스트하면 pyqtgraph 배선이 완전히 깨져 있어도 통과한다. 실제 드래그는
    RectROI 내부의 setPos/setSize를 거쳐 sigRegionChanged로 나오므로 그 경로를
    직접 두드려야 한다.
    """
    view = ImageView()
    view.set_image(synth_gap_image(width=512, height=512))
    seen = []
    view.roi_changed.connect(lambda: seen.append(1))
    view._roi.setPos([120, 130])
    assert seen


def test_overlay_can_be_shown_and_cleared(qapp):
    view = ImageView()
    view.set_image(synth_gap_image(width=64, height=64))
    overlay = np.zeros((64, 64, 3), dtype=np.uint8)
    overlay[..., 1] = 255
    view.show_overlay(overlay)
    assert view.has_overlay() is True
    view.clear_overlay()
    assert view.has_overlay() is False


def test_changing_the_image_resets_the_overlay(qapp):
    view = ImageView()
    view.set_image(synth_gap_image(width=64, height=64))
    view.show_overlay(np.zeros((64, 64, 3), dtype=np.uint8))
    view.set_image(synth_gap_image(width=64, height=64))
    assert view.has_overlay() is False


def test_empty_image_is_rejected_without_crashing(qapp):
    view = ImageView()
    view.set_image(np.empty((0, 0)))
    assert view.current_roi() is None
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python -m pytest tests/test_gui_image_view.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap_gui'`
(PySide6가 설치되지 않은 환경이면 SKIP된다. 그 경우 먼저 `pip install -e ".[gui,dev]"` 를 실행한다.)

- [ ] **Step 4: `ebl_gap_gui/__init__.py`와 `image_view.py` 구현**

`ebl_gap_gui/__init__.py`:

```python
"""EBL 갭 측정 툴의 GUI 층.

이 패키지는 ebl_gap 엔진을 호출해 결과를 그리기만 한다. 계산 로직을 여기 두면
화면 없이 검증할 수 없게 되므로, 어떤 측정 계산도 여기에 넣지 않는다.
"""
```

`ebl_gap_gui/image_view.py`:

```python
"""SEM 이미지를 띄우고 드래그로 ROI를 지정하는 위젯."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ebl_gap.types import Roi

DEFAULT_ROI_FRACTION = 0.4


class ImageView(QWidget):
    """이미지 + 드래그/리사이즈 ROI + 오버레이."""

    roi_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._shape: tuple[int, int] | None = None

        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.invertY(True)  # 이미지 좌표계: y가 아래로 증가
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self._plot.addItem(self._image_item)

        self._overlay_item = pg.ImageItem(axisOrder="row-major")
        self._overlay_item.setZValue(5)
        self._overlay_item.setVisible(False)
        self._plot.addItem(self._overlay_item)

        self._roi = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("y", width=2))
        self._roi.addScaleHandle([1, 1], [0, 0])
        self._roi.addScaleHandle([0, 0], [1, 1])
        self._roi.setZValue(10)
        self._roi.setVisible(False)
        # sigRegionChanged는 ROI 객체를 인자로 넘기며 발신한다. 0-인자 Signal의
        # emit에 직접 연결하면 PySide6가 매 변경마다 TypeError를 던지고 리스너는
        # 호출되지 않는다 — 마우스 드래그는 전부 이 경로를 타므로, 직접 연결하면
        # 프로그램이 set_roi()로 바꿀 때만 신호가 살아 있는 상태가 된다.
        self._roi.sigRegionChanged.connect(lambda *_: self.roi_changed.emit())
        self._plot.addItem(self._roi)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

    def set_image(self, pixels) -> None:
        """새 이미지를 띄운다. 오버레이는 지우고 ROI는 기본 위치로 돌린다."""
        array = np.asarray(pixels, dtype=np.float64)
        self.clear_overlay()
        if array.ndim != 2 or array.size == 0:
            self._shape = None
            self._roi.setVisible(False)
            self._image_item.clear()
            return

        self._shape = (array.shape[0], array.shape[1])
        self._image_item.setImage(array, autoLevels=True)
        self._plot.autoRange()

        height, width = self._shape
        w = max(9, int(width * DEFAULT_ROI_FRACTION))
        h = max(5, int(height * DEFAULT_ROI_FRACTION))
        self.set_roi(Roi((width - w) // 2, (height - h) // 2,
                         (width - w) // 2 + w - 1, (height - h) // 2 + h - 1))

    def set_roi(self, roi: Roi) -> None:
        """ROI를 지정한다. 이미지 밖으로 나가면 경계 안쪽으로 잘라 넣는다."""
        if self._shape is None:
            return
        height, width = self._shape
        x0 = max(0, min(roi.x0, width - 9))
        y0 = max(0, min(roi.y0, height - 5))
        x1 = max(x0 + 8, min(roi.x1, width - 1))
        y1 = max(y0 + 4, min(roi.y1, height - 1))
        self._roi.setVisible(True)
        self._roi.setPos([x0, y0], finish=False)
        self._roi.setSize([x1 - x0 + 1, y1 - y0 + 1], finish=False)
        self.roi_changed.emit()

    def current_roi(self) -> Roi | None:
        if self._shape is None or not self._roi.isVisible():
            return None
        pos = self._roi.pos()
        size = self._roi.size()
        height, width = self._shape
        x0 = int(round(pos.x()))
        y0 = int(round(pos.y()))
        x1 = x0 + int(round(size.x())) - 1
        y1 = y0 + int(round(size.y())) - 1
        x0 = max(0, min(x0, width - 9))
        y0 = max(0, min(y0, height - 5))
        x1 = max(x0 + 8, min(x1, width - 1))
        y1 = max(y0 + 4, min(y1, height - 1))
        try:
            return Roi(x0, y0, x1, y1)
        except ValueError:
            return None

    def show_overlay(self, rgb) -> None:
        array = np.asarray(rgb, dtype=np.uint8)
        self._overlay_item.setImage(array, autoLevels=False)
        self._overlay_item.setVisible(True)

    def clear_overlay(self) -> None:
        self._overlay_item.setVisible(False)

    def has_overlay(self) -> bool:
        return bool(self._overlay_item.isVisible())
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_image_view.py -v`
Expected: PASS (9 passed)

- [ ] **Step 6: 커밋**

```bash
git add ebl_gap_gui/ tests/conftest.py tests/test_gui_image_view.py
git commit -m "feat: add pyqtgraph image view with draggable measurement ROI"
```

---

### Task 15: 파일 목록, 결과 패널, 결과 테이블

**Files:**
- Create: `ebl_gap_gui/panels.py`
- Test: `tests/test_gui_panels.py`

**Interfaces:**
- Consumes: `ImageRecord`, `RoiResult` (Task 1), `Session` (Task 11)
- Produces: `FilePanel(QWidget)` — `set_records(records)`, `current_index() -> int | None`, `set_dose(index, dose)`, `refresh_row(index)`, 시그널 `selection_changed = Signal(int)`, `dose_edited = Signal(int, object)`; `ResultPanel(QWidget)` — `show_result(result)`, `clear()`, `text() -> str`; `ResultTable(QWidget)` — `set_session(session)`, `row_count() -> int`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_gui_panels.py`:

```python
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from ebl_gap.dataset import Session  # noqa: E402
from ebl_gap.types import ImageRecord, RoiResult, ScaleInfo  # noqa: E402
from ebl_gap_gui.panels import FilePanel, ResultPanel, ResultTable  # noqa: E402

SCALE = ScaleInfo(nm_per_px=3.0, source="fei_metadata")


def make_result(mean_nm=42.0, warnings=()):
    return RoiResult(mean_nm=mean_nm, std_nm=1.5, n_valid=280, n_short=12,
                     n_uncertain=8, n_low_confidence=30, angle_deg=2.4,
                     lines=(), warnings=tuple(warnings), scale=SCALE)


def make_records():
    return [
        ImageRecord(path=Path("a_300uC.tif"), scale=SCALE, dose=300.0,
                    roi_results=[make_result(60.0)]),
        ImageRecord(path=Path("b_400uC.tif"), scale=SCALE, dose=400.0,
                    roi_results=[make_result(30.0)]),
    ]


def test_file_panel_lists_one_row_per_record(qapp):
    panel = FilePanel()
    panel.set_records(make_records())
    assert panel.row_count() == 2
    assert panel.current_index() == 0


def test_file_panel_emits_selection_changes(qapp):
    panel = FilePanel()
    panel.set_records(make_records())
    seen = []
    panel.selection_changed.connect(seen.append)
    panel.select(1)
    assert seen[-1] == 1
    assert panel.current_index() == 1


def test_file_panel_shows_the_parsed_dose(qapp):
    panel = FilePanel()
    panel.set_records(make_records())
    assert panel.dose_text(0) == "300"
    assert panel.dose_text(1) == "400"


def test_file_panel_shows_an_empty_dose_cell_when_unknown(qapp):
    panel = FilePanel()
    panel.set_records([ImageRecord(path=Path("x.tif"), scale=SCALE, dose=None)])
    assert panel.dose_text(0) == ""


def test_setting_a_dose_updates_the_record_and_the_cell(qapp):
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)
    panel.set_dose(0, 275.0)
    assert records[0].dose == pytest.approx(275.0)
    assert panel.dose_text(0) == "275"


def test_editing_the_dose_cell_updates_the_record_and_emits(qapp):
    """실제 사용자가 타는 경로: 셀 텍스트를 직접 고친다.

    set_dose()는 _loading으로 막힌 경로를 지나므로 _on_item_changed를 거치지
    않는다. 그것만 테스트하면 편집 처리가 통째로 깨져도 통과한다.
    """
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)
    seen = []
    panel.dose_edited.connect(lambda i, v: seen.append((i, v)))
    panel._table.item(0, 1).setText("275")
    assert records[0].dose == pytest.approx(275.0)
    assert seen == [(0, 275.0)]


def test_clearing_the_dose_cell_means_unknown(qapp):
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)
    seen = []
    panel.dose_edited.connect(lambda i, v: seen.append((i, v)))
    panel._table.item(0, 1).setText("")
    assert records[0].dose is None
    assert seen == [(0, None)]


def test_unparseable_dose_reverts_and_does_not_emit(qapp):
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)
    seen = []
    panel.dose_edited.connect(lambda i, v: seen.append((i, v)))
    panel._table.item(0, 1).setText("abc")
    assert records[0].dose == pytest.approx(300.0)
    assert panel.dose_text(0) == "300"
    assert seen == []


def test_negative_dose_reverts(qapp):
    """음수 dose는 물리적으로 불가능하다. 조용히 받으면 곡선 x축이 틀어진다."""
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)
    seen = []
    panel.dose_edited.connect(lambda i, v: seen.append((i, v)))
    panel._table.item(0, 1).setText("-50")
    assert records[0].dose == pytest.approx(300.0)
    assert seen == []


def test_reloading_onto_the_same_row_still_announces_the_selection(qapp):
    """0행이 선택된 채 다른 폴더를 열면 Qt는 신호를 보내지 않는다.

    그대로 두면 결과 패널이 이전 폴더의 결과를 계속 보여준다.
    """
    panel = FilePanel()
    panel.set_records(make_records())
    seen = []
    panel.selection_changed.connect(seen.append)
    panel.set_records(make_records())
    assert seen == [0]


def test_loading_flag_is_cleared_even_when_filling_a_row_raises(qapp):
    """예외로 _loading이 True로 남으면 이후 dose 편집이 전부 조용히 무시된다.

    이 상태는 눈에 보이지 않는다 — 사용자는 dose를 고쳤는데 아무 일도 일어나지
    않는 것만 본다. 그래서 플래그 불변식을 직접 확인한다.

    예외 뒤에 성공하는 set_records를 한 번 끼워 넣고 셀 편집으로 확인하려 하면
    안 된다. 그 호출이 try/finally 없이도 자기 끝에서 플래그를 내려버려서,
    버그가 있든 없든 통과하는 테스트가 된다.
    """
    records = make_records()
    panel = FilePanel()
    panel.set_records(records)

    broken = ImageRecord(path=Path("x.tif"), scale=SCALE, dose=1.0)
    broken.path = None  # _fill_row에서 .name 접근이 터진다
    with pytest.raises(AttributeError):
        panel.set_records([broken])

    assert panel._loading is False


def test_file_panel_marks_measured_and_unmeasured_rows(qapp):
    records = make_records()
    records[1].roi_results = []
    panel = FilePanel()
    panel.set_records(records)
    assert "측정" in panel.status_text(0)
    assert panel.status_text(1) == "미측정"


def test_file_panel_marks_load_errors(qapp):
    record = ImageRecord(path=Path("bad.tif"), error="메타데이터 없음")
    panel = FilePanel()
    panel.set_records([record])
    assert "오류" in panel.status_text(0)


def test_result_panel_reports_the_mean_counts_and_scale_source(qapp):
    panel = ResultPanel()
    panel.show_result(make_result(42.0))
    text = panel.text()
    assert "42.00" in text
    assert "280" in text  # 유효 라인
    assert "12" in text   # short 라인
    assert "fei_metadata" in text
    assert "2.4" in text  # 각도


def test_result_panel_lists_warnings(qapp):
    panel = ResultPanel()
    panel.show_result(make_result(warnings=("short 발생 구간 있음, 확인 필요",)))
    assert "short 발생" in panel.text()


def test_result_panel_says_so_when_there_is_no_measurement(qapp):
    panel = ResultPanel()
    panel.show_result(RoiResult(mean_nm=None, std_nm=None, n_valid=0,
                                n_short=300, n_uncertain=0, n_low_confidence=0,
                                angle_deg=0.0, lines=(), warnings=(),
                                scale=SCALE))
    assert "측정 불가" in panel.text()


def test_result_panel_clears(qapp):
    panel = ResultPanel()
    panel.show_result(make_result())
    panel.clear()
    assert panel.text().strip() == ""


def test_result_table_has_one_row_per_measured_roi(qapp):
    session = Session()
    for record in make_records():
        session.add(record)
    table = ResultTable()
    table.set_session(session)
    assert table.row_count() == 2
    assert table.cell(0, "file") == "a_300uC.tif"
    assert table.cell(0, "mean_nm") == "60.00"


def test_result_table_is_emptied_by_a_fresh_session(qapp):
    table = ResultTable()
    session = Session()
    session.add(make_records()[0])
    table.set_session(session)
    table.set_session(Session())
    assert table.row_count() == 0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_panels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap_gui.panels'`

- [ ] **Step 3: `ebl_gap_gui/panels.py` 구현**

```python
"""파일 목록, ROI 결과 요약, 세션 결과 테이블 위젯."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ebl_gap.dataset import Session
from ebl_gap.types import ImageRecord, RoiResult

FILE_COLUMNS = ("파일", "dose(uC)", "상태")

TABLE_COLUMNS = (
    ("file", "파일"),
    ("dose_uC", "dose(uC)"),
    ("mean_nm", "갭(nm)"),
    ("std_nm", "표준편차(nm)"),
    ("n_valid", "유효"),
    ("n_short", "short"),
    ("n_uncertain", "판정보류"),
    ("angle_deg", "각도(도)"),
    ("scale_source", "스케일 출처"),
)


def _fmt(value, digits=2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


class FilePanel(QWidget):
    """불러온 이미지 목록. dose는 셀을 눌러 직접 고칠 수 있다."""

    selection_changed = Signal(int)
    dose_edited = Signal(int, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[ImageRecord] = []
        self._loading = False

        self._table = QTableWidget(0, len(FILE_COLUMNS))
        self._table.setHorizontalHeaderLabels(FILE_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self._table.currentCellChanged.connect(self._on_current_cell_changed)
        self._table.itemChanged.connect(self._on_item_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)

    def set_records(self, records) -> None:
        self._loading = True
        try:
            self._records = list(records)
            self._table.setRowCount(len(self._records))
            for index in range(len(self._records)):
                self._fill_row(index)
            if self._records:
                self._table.setCurrentCell(0, 0)
        finally:
            # 예외가 나도 플래그를 반드시 내린다. True로 남으면 이후 사용자의 dose
            # 편집이 전부 조용히 무시되는데, 그것이 잘못 파싱된 dose를 바로잡는
            # 유일한 경로다. 실패가 눈에 보이지도 않는다.
            self._loading = False
        if self._records:
            # Qt의 currentCellChanged는 인덱스가 실제로 바뀔 때만 발신한다. 0행이
            # 선택된 채로 다른 폴더를 열면 setCurrentCell(0, 0)이 no-op이라 신호가
            # 나가지 않고, 결과 패널이 이전 폴더의 결과를 계속 보여준다. 위에서
            # _loading으로 암묵 발신을 막았으므로 여기서 정확히 한 번 발신된다.
            self.selection_changed.emit(0)

    def _fill_row(self, index: int) -> None:
        record = self._records[index]

        name = QTableWidgetItem(record.path.name)
        name.setFlags(name.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(index, 0, name)

        dose = QTableWidgetItem("" if record.dose is None else f"{record.dose:g}")
        self._table.setItem(index, 1, dose)

        status = QTableWidgetItem(self._status_for(record))
        status.setFlags(status.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(index, 2, status)

    @staticmethod
    def _status_for(record: ImageRecord) -> str:
        if record.error and record.scale is None:
            return f"오류: {record.error[:20]}"
        if record.roi_results:
            return f"측정 {len(record.roi_results)}건"
        return "미측정"

    def refresh_row(self, index: int) -> None:
        self._loading = True
        try:
            self._fill_row(index)
        finally:
            self._loading = False

    def _on_current_cell_changed(self, row: int, _col, _prow, _pcol) -> None:
        if not self._loading and 0 <= row < len(self._records):
            self.selection_changed.emit(row)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() != 1:
            return
        row = item.row()
        text = item.text().strip()
        try:
            dose = float(text) if text else None
        except ValueError:
            self.refresh_row(row)
            return
        if dose is not None and dose < 0:
            # 음수 dose는 물리적으로 불가능하다. 파싱 불가와 같이 되돌린다 —
            # 조용히 받아들이면 dose-gap 곡선의 x축이 틀어진 채로 해석된다.
            self.refresh_row(row)
            return
        self._records[row].dose = dose
        self.dose_edited.emit(row, dose)

    def set_dose(self, index: int, dose: float | None) -> None:
        self._records[index].dose = dose
        self.refresh_row(index)

    def select(self, index: int) -> None:
        self._table.setCurrentCell(index, 0)

    def current_index(self) -> int | None:
        row = self._table.currentRow()
        return row if 0 <= row < len(self._records) else None

    def row_count(self) -> int:
        return self._table.rowCount()

    def dose_text(self, index: int) -> str:
        item = self._table.item(index, 1)
        return "" if item is None else item.text()

    def status_text(self, index: int) -> str:
        item = self._table.item(index, 2)
        return "" if item is None else item.text()


class ResultPanel(QWidget):
    """선택된 ROI 하나의 결과 요약."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addStretch(1)

    def show_result(self, result: RoiResult) -> None:
        if result.mean_nm is None:
            head = "갭 측정 불가"
        else:
            spread = "" if result.std_nm is None else f" ± {result.std_nm:.2f}"
            head = f"갭 {result.mean_nm:.2f}{spread} nm"

        parts = [
            head,
            f"유효 {result.n_valid} / short {result.n_short} / "
            f"판정보류 {result.n_uncertain} 라인",
            f"정밀도 주의 {result.n_low_confidence} 라인",
            f"갭 각도 {result.angle_deg:.2f}도",
            f"{result.scale.nm_per_px:.4f} nm/px [{result.scale.source}]",
        ]
        if result.warnings:
            parts.append("")
            parts.extend(f"! {w}" for w in result.warnings)
        self._label.setText("\n".join(parts))

    def clear(self) -> None:
        self._label.setText("")

    def text(self) -> str:
        return self._label.text()


class ResultTable(QWidget):
    """세션 전체의 측정 결과 표."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._keys = [key for key, _ in TABLE_COLUMNS]
        self._table = QTableWidget(0, len(TABLE_COLUMNS))
        self._table.setHorizontalHeaderLabels([label for _, label in TABLE_COLUMNS])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)

    def set_session(self, session: Session) -> None:
        rows = [(record, index, result)
                for record in session.records
                for index, result in enumerate(record.roi_results)]
        self._table.setRowCount(len(rows))
        for row, (record, _index, result) in enumerate(rows):
            values = {
                "file": record.path.name,
                "dose_uC": "" if record.dose is None else f"{record.dose:g}",
                "mean_nm": _fmt(result.mean_nm),
                "std_nm": _fmt(result.std_nm),
                "n_valid": str(result.n_valid),
                "n_short": str(result.n_short),
                "n_uncertain": str(result.n_uncertain),
                "angle_deg": f"{result.angle_deg:.2f}",
                "scale_source": result.scale.source,
            }
            for column, key in enumerate(self._keys):
                self._table.setItem(row, column, QTableWidgetItem(values[key]))

    def row_count(self) -> int:
        return self._table.rowCount()

    def cell(self, row: int, key: str) -> str:
        item = self._table.item(row, self._keys.index(key))
        return "" if item is None else item.text()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_panels.py -v`
Expected: PASS (19 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap_gui/panels.py tests/test_gui_panels.py
git commit -m "feat: add file list, result summary and session result table panels"
```

---

### Task 16: dose–gap 곡선

**Files:**
- Create: `ebl_gap_gui/dose_plot.py`
- Test: `tests/test_gui_dose_plot.py`

**Interfaces:**
- Consumes: `Session`, `DosePoint` (Task 11)
- Produces: `DosePlot(QWidget)` — `set_session(session)`, `point_count() -> int`, `warning_text() -> str`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_gui_dose_plot.py`:

```python
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from ebl_gap.dataset import Session  # noqa: E402
from ebl_gap.types import ImageRecord, RoiResult, ScaleInfo  # noqa: E402
from ebl_gap_gui.dose_plot import DosePlot  # noqa: E402

SCALE = ScaleInfo(nm_per_px=3.0, source="fei_metadata")


def result(mean_nm, std_nm=1.0, n_valid=200, n_short=0, scale=SCALE):
    return RoiResult(mean_nm=mean_nm, std_nm=std_nm, n_valid=n_valid,
                     n_short=n_short, n_uncertain=0, n_low_confidence=0,
                     angle_deg=0.0, lines=(), warnings=(), scale=scale)


def session_with(*pairs, scale=SCALE):
    session = Session()
    for dose, mean_nm in pairs:
        session.add(ImageRecord(path=Path(f"{dose:g}uC.tif"), scale=scale,
                                dose=dose, roi_results=[result(mean_nm)]))
    return session


def test_plots_one_point_per_measured_dose(qapp):
    plot = DosePlot()
    plot.set_session(session_with((300.0, 80.0), (400.0, 50.0), (500.0, 20.0)))
    assert plot.point_count() == 3


def test_ignores_images_without_a_dose(qapp):
    session = session_with((300.0, 80.0))
    session.add(ImageRecord(path=Path("nodose.tif"), scale=SCALE, dose=None,
                            roi_results=[result(40.0)]))
    plot = DosePlot()
    plot.set_session(session)
    assert plot.point_count() == 1


def test_replacing_the_session_clears_the_previous_curve(qapp):
    plot = DosePlot()
    plot.set_session(session_with((300.0, 80.0), (400.0, 50.0)))
    plot.set_session(Session())
    assert plot.point_count() == 0


def test_shows_a_warning_when_magnification_is_mixed(qapp):
    session = session_with((300.0, 80.0))
    coarse = ScaleInfo(nm_per_px=9.0, source="fei_metadata")
    session.add(ImageRecord(path=Path("b.tif"), scale=coarse, dose=400.0,
                            roi_results=[result(50.0, scale=coarse)]))
    plot = DosePlot()
    plot.set_session(session)
    assert "배율" in plot.warning_text()


def test_no_warning_for_a_consistent_session(qapp):
    plot = DosePlot()
    plot.set_session(session_with((300.0, 80.0), (400.0, 50.0)))
    assert plot.warning_text() == ""


def test_short_ratio_uses_valid_plus_short_as_the_denominator(qapp):
    """분모가 n_valid + n_short인지 경계에서 확인한다.

    52/(1000+52) = 0.0494 -> 임계 5% 미만이라 표시 안 됨.
    분모를 n_valid로 잘못 쓰면 52/1000 = 0.052로 임계를 넘어 표시된다.
    값이 극단적인 케이스만 테스트하면 분모를 틀려도 통과하는데, 이 분모가
    dose 점에 "short 발생" 표시를 붙일지 결정하고 사용자는 그걸 보고 dose를 고른다.
    """
    session = Session()
    session.add(ImageRecord(path=Path("a.tif"), scale=SCALE, dose=300.0,
                            roi_results=[result(80.0, n_valid=1000, n_short=52)]))
    plot = DosePlot()
    plot.set_session(session)
    assert plot.point_count() == 1
    assert plot.shorted_point_count() == 0


def test_shorted_doses_are_marked_separately(qapp):
    session = Session()
    session.add(ImageRecord(path=Path("a.tif"), scale=SCALE, dose=300.0,
                            roi_results=[result(80.0)]))
    session.add(ImageRecord(path=Path("b.tif"), scale=SCALE, dose=500.0,
                            roi_results=[result(15.0, n_valid=50, n_short=250)]))
    plot = DosePlot()
    plot.set_session(session)
    assert plot.point_count() == 2
    assert plot.shorted_point_count() == 1
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_dose_plot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap_gui.dose_plot'`

- [ ] **Step 3: `ebl_gap_gui/dose_plot.py` 구현**

```python
"""dose에 따른 갭 폭 변화를 보여주는 그래프."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ebl_gap.dataset import Session

#: short 라인 비율이 이 값을 넘으면 그 dose 점을 빨갛게 표시한다.
SHORT_RATIO_MARK = 0.05


class DosePlot(QWidget):
    """dose-gap 곡선. short가 섞인 점은 눈에 띄게 표시한다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_points = 0
        self._n_shorted = 0

        self._warning = QLabel("")
        self._warning.setWordWrap(True)
        self._warning.setStyleSheet("color: #b35c00;")

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "dose", units="uC/cm2")
        self._plot.setLabel("left", "갭 폭", units="nm")
        self._plot.showGrid(x=True, y=True, alpha=0.3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._warning)
        layout.addWidget(self._plot)

    def set_session(self, session: Session) -> None:
        self._plot.clear()
        points = session.dose_curve()
        self._n_points = len(points)
        self._n_shorted = 0

        warnings = session.scale_warnings()
        self._warning.setText(" | ".join(warnings))

        if not points:
            return

        doses = np.array([p.dose for p in points], dtype=float)
        means = np.array([p.mean_nm for p in points], dtype=float)
        spreads = np.array(
            [0.0 if p.std_nm is None else p.std_nm for p in points], dtype=float
        )

        self._plot.plot(doses, means, pen=pg.mkPen("#1f77b4", width=2),
                        symbol="o", symbolSize=8, symbolBrush="#1f77b4")
        self._plot.addItem(pg.ErrorBarItem(x=doses, y=means, height=2 * spreads,
                                           pen=pg.mkPen("#1f77b4")))

        shorted = [p for p in points
                   if p.n_valid + p.n_short > 0
                   and p.n_short / (p.n_valid + p.n_short) > SHORT_RATIO_MARK]
        self._n_shorted = len(shorted)
        if shorted:
            self._plot.plot(
                np.array([p.dose for p in shorted], dtype=float),
                np.array([p.mean_nm for p in shorted], dtype=float),
                pen=None, symbol="x", symbolSize=16,
                symbolPen=pg.mkPen("#d62728", width=3),
            )

    def point_count(self) -> int:
        return self._n_points

    def shorted_point_count(self) -> int:
        return self._n_shorted

    def warning_text(self) -> str:
        return self._warning.text()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_dose_plot.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap_gui/dose_plot.py tests/test_gui_dose_plot.py
git commit -m "feat: add dose-gap curve plot with short-dose markers"
```

---

### Task 17: 스케일 캘리브레이션 다이얼로그

FEI 메타데이터가 없는 이미지에서 스케일을 확정하는 창이다. 실제 Inspect F 파일을 확보하기 전에는 이 경로가 실사용의 주 경로가 될 수 있으므로 제대로 만든다.

**자동 검출은 조용히 틀릴 수 있다.** `detect_scalebar`는 데이터바에서 가장 긴 밝은 수평 런을 고르는데, 실제 SEM 데이터바에는 HV·WD·배율·파일명이 밝은 텍스트로 들어간다. 그 텍스트 블록이 막대보다 긴 런을 만들면 검출기는 아무 신호 없이 텍스트 좌표를 돌려준다(60px 막대와 150px 텍스트 블록으로 실증됨). 이것은 스펙이 OCR을 거부한 이유와 **정확히 같은 실패 방식**이다. 그래서 이 창은 검출 결과를 그냥 쓰지 않는다: 어디서 찾았는지 좌표로 보여주고, 사용자가 명시적으로 확인해야만 자동 경로를 쓴다. 사용자가 직접 잰 픽셀 거리를 넣은 경우는 사람이 이미 본 것이므로 확인이 필요 없다.

**Files:**
- Create: `ebl_gap_gui/calibration.py`
- Test: `tests/test_gui_calibration.py`

**Interfaces:**
- Consumes: `detect_scalebar`, `scale_from_scalebar`, `scale_from_two_points` (Task 10), `ScaleInfo` (Task 1)
- Produces: `CalibrationDialog(QDialog)` — 생성자 `(pixels, databar_top=None, parent=None)`; 메서드 `detected_length_px() -> int | None`, `summary_text() -> str`, `set_length(value, unit)`, `set_manual_pixels(distance_px)`, `confirm_detection(checked)`, `scale_info() -> ScaleInfo | None`; 상수 `UNIT_FACTORS`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_gui_calibration.py`:

```python
import numpy as np
import pytest

pytest.importorskip("PySide6")

from ebl_gap_gui.calibration import UNIT_FACTORS, CalibrationDialog  # noqa: E402


def databar_image(bar_length=100):
    img = np.full((943, 1024), 120.0)
    img[884:, :] = 10.0
    img[918:923, 60:60 + bar_length] = 250.0
    return img


def test_unit_factors_convert_to_nanometres():
    assert UNIT_FACTORS["nm"] == pytest.approx(1.0)
    assert UNIT_FACTORS["µm"] == pytest.approx(1000.0)


def test_dialog_detects_the_bar_length_on_open(qapp):
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    assert dialog.detected_length_px() == pytest.approx(100, abs=2)


def test_dialog_reports_no_detection_on_a_plain_image(qapp):
    dialog = CalibrationDialog(np.full((200, 200), 100.0))
    assert dialog.detected_length_px() is None


def test_scale_from_detected_bar_and_entered_length(qapp):
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    dialog.set_length(1.0, "µm")
    dialog.confirm_detection(True)
    scale = dialog.scale_info()
    assert scale.nm_per_px == pytest.approx(10.0, rel=0.05)
    assert scale.source == "scalebar_auto"


def test_nanometre_unit_is_used_as_entered(qapp):
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    dialog.set_length(500.0, "nm")
    dialog.confirm_detection(True)
    assert dialog.scale_info().nm_per_px == pytest.approx(5.0, rel=0.05)


def test_detected_bar_is_not_used_until_the_user_confirms_it(qapp):
    """자동 검출은 데이터바의 밝은 텍스트를 막대로 오인할 수 있다.

    스펙이 OCR을 거부한 이유와 같은 실패 방식이므로, 확인 없이는 쓰지 않는다.
    """
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    dialog.set_length(1.0, "µm")
    assert dialog.scale_info() is None
    dialog.confirm_detection(True)
    assert dialog.scale_info().source == "scalebar_auto"


def test_summary_tells_the_user_where_the_bar_was_found(qapp):
    """길이만 보여주면 잘못 잡았는지 알 수 없다. 위치를 보여줘야 확인이 가능하다."""
    text = CalibrationDialog(databar_image(), databar_top=884).summary_text()
    assert "행" in text
    assert "60" in text  # 막대 시작 x 좌표


def test_typing_into_the_length_field_works_without_the_helper(qapp):
    """사용자는 set_length()를 부르지 않는다. 스핀박스에 직접 입력한다.

    헬퍼를 통해서만 테스트하면 실제 폼이 동작하지 않아도 전부 통과한다.
    이 프로젝트에서 같은 패턴의 버그가 이미 네 번 나왔다.
    """
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    dialog._length.setValue(1.0)
    dialog._unit.setCurrentText("µm")
    dialog._confirm.setChecked(True)
    scale = dialog.scale_info()
    assert scale is not None
    assert scale.nm_per_px == pytest.approx(10.0, rel=0.05)
    assert scale.source == "scalebar_auto"


def test_manual_pixel_distance_needs_no_confirmation(qapp):
    """사람이 직접 잰 거리는 이미 눈으로 확인한 값이다."""
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    dialog.set_length(1.0, "µm")
    dialog.set_manual_pixels(200.0)
    scale = dialog.scale_info()
    assert scale.nm_per_px == pytest.approx(5.0)
    assert scale.source == "manual"


def test_scale_is_none_before_a_length_is_entered(qapp):
    dialog = CalibrationDialog(databar_image(), databar_top=884)
    assert dialog.scale_info() is None


def test_scale_is_none_when_nothing_was_detected_and_nothing_entered(qapp):
    dialog = CalibrationDialog(np.full((200, 200), 100.0))
    dialog.set_length(1.0, "µm")
    assert dialog.scale_info() is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap_gui.calibration'`

- [ ] **Step 3: `ebl_gap_gui/calibration.py` 구현**

```python
"""메타데이터가 없는 이미지의 스케일을 사람이 확정하는 창.

스케일바 라벨을 OCR로 읽지 않고 사용자에게 묻는다. 잘못 읽은 숫자 하나로 모든
측정값이 조용히 틀어지는 것보다 한 번 묻는 편이 낫다.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from ebl_gap.scalebar import (
    detect_scalebar,
    scale_from_scalebar,
    scale_from_two_points,
)
from ebl_gap.types import ScaleInfo

UNIT_FACTORS = {"nm": 1.0, "µm": 1000.0}


class CalibrationDialog(QDialog):
    """검출된 스케일바 길이 또는 사용자가 잰 픽셀 거리로 스케일을 만든다."""

    def __init__(self, pixels, databar_top: int | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("스케일 캘리브레이션")

        array = np.asarray(pixels, dtype=np.float64)
        hit = detect_scalebar(array, databar_top=databar_top) if array.size else None
        self._detected_px: int | None = hit.length_px if hit else None

        if hit is None:
            summary = ("스케일바를 자동으로 찾지 못했습니다. "
                       "이미지에서 두 점을 찍어 픽셀 거리를 입력하세요.")
        else:
            # 어디서 찾았는지 반드시 보여준다. 데이터바의 밝은 텍스트가 막대보다 긴
            # 런을 만들면 검출기가 아무 신호 없이 텍스트 좌표를 돌려주기 때문이다.
            summary = (
                f"스케일바 막대를 {hit.length_px} px로 검출했습니다 "
                f"(행 {hit.row}, x {hit.x0}~{hit.x1}). "
                "데이터바의 밝은 텍스트를 막대로 잘못 잡을 수 있으니 "
                "위치가 맞는지 확인한 뒤 아래를 체크하세요."
            )
        self._summary = QLabel(summary)
        self._summary.setWordWrap(True)

        self._confirm = QCheckBox("검출된 막대가 맞습니다")
        self._confirm.setEnabled(hit is not None)

        self._length = QDoubleSpinBox()
        self._length.setDecimals(4)
        self._length.setRange(0.0, 1e6)
        self._length.setValue(0.0)

        self._unit = QComboBox()
        self._unit.addItems(list(UNIT_FACTORS))
        self._unit.setCurrentText("µm")

        self._manual = QDoubleSpinBox()
        self._manual.setDecimals(2)
        self._manual.setRange(0.0, 1e6)
        self._manual.setValue(0.0)

        form = QFormLayout()
        form.addRow("실제 길이", self._length)
        form.addRow("단위", self._unit)
        form.addRow("직접 잰 픽셀 거리 (선택)", self._manual)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._summary)
        layout.addWidget(self._confirm)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def detected_length_px(self) -> int | None:
        return self._detected_px

    def summary_text(self) -> str:
        return self._summary.text()

    def confirm_detection(self, checked: bool) -> None:
        """검출된 막대가 맞다고 사용자가 확인한다."""
        self._confirm.setChecked(bool(checked))

    def set_length(self, value: float, unit: str) -> None:
        self._length.setValue(float(value))
        self._unit.setCurrentText(unit)

    def set_manual_pixels(self, distance_px: float) -> None:
        self._manual.setValue(float(distance_px))

    def length_nm(self) -> float:
        return self._length.value() * UNIT_FACTORS[self._unit.currentText()]

    def scale_info(self) -> ScaleInfo | None:
        """입력이 충분하면 ScaleInfo를, 아니면 None을 돌려준다.

        자동 검출 경로는 사용자가 확인 체크를 해야만 쓴다. 직접 잰 픽셀 거리는
        사람이 이미 이미지를 보고 잰 값이므로 별도 확인이 필요 없다.
        """
        # 위젯 값만 읽는다. 별도 "입력했음" 플래그를 두면 안 된다 — 헬퍼
        # 메서드에서만 세워지고 실제 스핀박스 입력에는 반응하지 않아서,
        # 사용자가 칸에 직접 타이핑하면 그 값이 통째로 무시된다.
        # 스핀박스 기본값과 하한이 둘 다 0이므로 length_nm <= 0이 "미입력"과
        # 같은 뜻이고, 플래그는 불필요할 뿐 아니라 해롭다.
        length_nm = self.length_nm()
        if length_nm <= 0:
            return None

        manual_px = self._manual.value()
        if manual_px > 0:
            return scale_from_two_points((0.0, 0.0), (manual_px, 0.0), length_nm)

        if self._detected_px and self._confirm.isChecked():
            return scale_from_scalebar(float(self._detected_px), length_nm)
        return None
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_calibration.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: 커밋**

```bash
git add ebl_gap_gui/calibration.py tests/test_gui_calibration.py
git commit -m "feat: add scale calibration dialog for images without FEI metadata"
```

---

### Task 18: 메인 윈도우 배선, 실행 스크립트, README

**Files:**
- Create: `ebl_gap_gui/app.py`
- Create: `README.md` (기존 파일을 덮어쓴다)
- Test: `tests/test_gui_app.py`

**Interfaces:**
- Consumes: Task 12~17 전부
- Produces: `MainWindow(QMainWindow)` — `open_folder(path)`, `measure_current()`, `export_summary_csv(path)`, `export_lines_csv(path)`, `export_overlay(path)`, `export_report(path)`, 속성 `session`; `main() -> int`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_gui_app.py`:

```python
import csv

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from ebl_gap_gui.app import MainWindow  # noqa: E402
from tests.synth import synth_gap_image  # noqa: E402
from tests.test_metadata import FEI_INI  # noqa: E402


def write_sample(path, gap_nm, dose):
    """3 nm/px 메타데이터가 붙은 512x512 합성 SEM TIFF를 만든다."""
    # 합성 이미지가 3.0 nm/px이므로 메타데이터도 정확히 3.0 nm/px로 맞춘다.
    # 어긋나면 측정값에 계통 오차가 생겨 테스트가 무엇을 재는지 흐려진다.
    ini = (FEI_INI
           .replace("ResolutionX=1024", "ResolutionX=512")
           .replace("ResolutionY=884", "ResolutionY=512")
           .replace("PixelWidth=3.0517578125e-009", "PixelWidth=3.0e-009")
           .replace("PixelHeight=3.0517578125e-009", "PixelHeight=3.0e-009")
           .replace("HorFieldsize=3.125e-006", "HorFieldsize=1.536e-006"))
    img = synth_gap_image(width=512, height=512, gap_nm=gap_nm, nm_per_px=3.0,
                          angle_deg=2.0, edge_sigma_px=1.2, noise_sigma=3.0,
                          seed=int(dose))
    data = np.clip(img, 0, 255).astype(np.uint8)
    out = path / f"pattern_{dose:g}uC.tif"
    tifffile.imwrite(out, data, extratags=[(34682, 's', 0, ini, True)])
    return out


@pytest.fixture()
def folder(tmp_path):
    write_sample(tmp_path, gap_nm=90.0, dose=300)
    write_sample(tmp_path, gap_nm=60.0, dose=400)
    return tmp_path


def test_open_folder_loads_every_image(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    assert len(window.session.records) == 2
    assert all(r.scale is not None for r in window.session.records)


def test_open_folder_reads_dose_from_filenames(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    assert sorted(r.dose for r in window.session.records) == [300.0, 400.0]


def test_measure_current_produces_a_result_near_the_true_gap(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    record = window.session.records[0]
    assert len(record.roi_results) == 1
    assert record.roi_results[0].mean_nm == pytest.approx(90.0, abs=3.0)


def test_measuring_twice_replaces_rather_than_appends(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    window.measure_current()
    assert len(window.session.records[0].roi_results) == 1


def test_measurement_updates_the_result_panel_and_table(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    assert "nm" in window.result_panel.text()
    assert window.result_table.row_count() == 1


def test_measuring_both_images_fills_the_dose_curve(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    for index in (0, 1):
        window.select_image(index)
        window.measure_current()
    assert window.dose_plot.point_count() == 2


def test_dragging_the_roi_clears_the_stale_overlay(qapp, folder):
    """측정 후 ROI를 옮기면 이전 위치의 에지 오버레이가 남아 있으면 안 된다."""
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    assert window.image_view.has_overlay() is True
    window.image_view._roi.setPos([120, 130])
    assert window.image_view.has_overlay() is False


def test_export_summary_csv_has_a_row_per_measurement(qapp, folder, tmp_path):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    out = tmp_path / "summary.csv"
    window.export_summary_csv(out)
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert len(rows) == 2  # 측정 1건 + 미측정 1건
    assert any(row["mean_nm"] for row in rows)


def test_export_overlay_writes_a_png(qapp, folder, tmp_path):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    out = tmp_path / "overlay.png"
    window.export_overlay(out)
    assert out.exists() and out.stat().st_size > 0


def test_export_report_mentions_the_scale_source(qapp, folder, tmp_path):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    out = tmp_path / "report.txt"
    window.export_report(out)
    assert "fei_metadata" in out.read_text(encoding="utf-8")


def test_measuring_without_a_scale_is_refused_with_a_message(qapp, tmp_path):
    plain = tmp_path / "plain_300uC.tif"
    tifffile.imwrite(plain, np.zeros((256, 256), dtype=np.uint8))
    window = MainWindow()
    window.open_folder(tmp_path)
    window.select_image(0)
    window.measure_current()
    assert window.session.records[0].roi_results == []
    assert "스케일" in window.status_text()


def test_empty_folder_is_handled_without_crashing(qapp, tmp_path):
    window = MainWindow()
    window.open_folder(tmp_path)
    assert window.session.records == []
    window.measure_current()  # 예외 없이 지나가야 한다
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap_gui.app'`

- [ ] **Step 3: `ebl_gap_gui/app.py` 구현**

```python
"""메인 윈도우. 엔진 호출과 위젯 갱신을 배선하기만 한다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QWidget,
)
from PySide6.QtCore import Qt

from ebl_gap.dataset import Session
from ebl_gap.export import (
    format_report,
    render_overlay,
    write_lines_csv,
    write_overlay_png,
    write_summary_csv,
)
from ebl_gap.loader import load_image
from ebl_gap.measure import MeasureParams, measure_roi
from ebl_gap_gui.calibration import CalibrationDialog
from ebl_gap_gui.dose_plot import DosePlot
from ebl_gap_gui.image_view import ImageView
from ebl_gap_gui.panels import FilePanel, ResultPanel, ResultTable

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class MainWindow(QMainWindow):
    """폴더 열기 -> ROI 드래그 -> 측정 -> 내보내기."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("EBL dose test 갭 분석")
        self.session = Session()
        self.params = MeasureParams()

        self._pixels: dict[int, np.ndarray] = {}
        self._databar_tops: dict[int, int | None] = {}
        self._current: int | None = None
        self._status = ""

        self.file_panel = FilePanel()
        self.image_view = ImageView()
        self.result_panel = ResultPanel()
        self.result_table = ResultTable()
        self.dose_plot = DosePlot()

        self.file_panel.selection_changed.connect(self.select_image)
        self.file_panel.dose_edited.connect(lambda *_: self._refresh_session_views())
        # ROI를 옮기면 낡은 오버레이를 지운다. 측정 결과는 그 ROI에 묶여 있으므로,
        # 새 위치에 이전 위치의 에지가 그려진 채로 남으면 사용자가 틀린 그림을
        # 보고 판단하게 된다. 다시 측정할 때 새 오버레이가 그려진다.
        self.image_view.roi_changed.connect(self.image_view.clear_overlay)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.file_panel)
        splitter.addWidget(self.image_view)
        splitter.addWidget(self.result_panel)
        splitter.setSizes([260, 700, 300])
        self.setCentralWidget(splitter)

        tabs = QTabWidget()
        tabs.addTab(self.result_table, "결과 테이블")
        tabs.addTab(self.dose_plot, "dose-gap 곡선")
        dock = QDockWidget("세션", self)
        dock.setWidget(tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

        self._build_toolbar()
        self.statusBar().showMessage("폴더를 열어 시작하세요")

    # ------------------------------------------------------------------ 배선

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("주요 동작")
        bar.addAction("폴더 열기", self._choose_folder)
        bar.addAction("측정", self.measure_current)
        bar.addAction("스케일 캘리브레이션", self.calibrate_current)
        bar.addSeparator()
        bar.addAction("요약 CSV", lambda: self._save_as(self.export_summary_csv,
                                                        "summary.csv"))
        bar.addAction("라인 CSV", lambda: self._save_as(self.export_lines_csv,
                                                        "lines.csv"))
        bar.addAction("오버레이 PNG", lambda: self._save_as(self.export_overlay,
                                                            "overlay.png"))
        bar.addAction("요약 리포트", lambda: self._save_as(self.export_report,
                                                           "report.txt"))

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "SEM 이미지 폴더 선택")
        if folder:
            self.open_folder(folder)

    def _save_as(self, handler, default_name: str) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "저장", default_name)
        if path:
            handler(path)

    def _set_status(self, message: str) -> None:
        self._status = message
        self.statusBar().showMessage(message)

    def status_text(self) -> str:
        return self._status

    # ------------------------------------------------------------------ 동작

    def open_folder(self, folder: str | Path) -> None:
        """폴더 안의 이미지를 전부 읽어 세션을 새로 만든다."""
        folder = Path(folder)
        paths = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in IMAGE_SUFFIXES)

        self.session = Session()
        self._pixels.clear()
        self._databar_tops.clear()
        self._current = None

        for index, path in enumerate(paths):
            loaded = load_image(path)
            self.session.add(loaded.record)
            self._pixels[index] = loaded.pixels
            self._databar_tops[index] = loaded.databar_top

        self.file_panel.set_records(self.session.records)
        self._refresh_session_views()
        if self.session.records:
            self.select_image(0)
            self._set_status(f"{len(self.session.records)}장 불러옴")
        else:
            self.image_view.set_image(np.empty((0, 0)))
            self.result_panel.clear()
            self._set_status("폴더에 이미지가 없습니다")

    def select_image(self, index: int) -> None:
        if not (0 <= index < len(self.session.records)):
            return
        self._current = index
        self.image_view.set_image(self._pixels[index])

        record = self.session.records[index]
        if record.roi_results:
            self.result_panel.show_result(record.roi_results[0])
        else:
            self.result_panel.clear()
        if record.error:
            self._set_status(f"{record.path.name}: {record.error}")
        else:
            self._set_status(record.path.name)

    def measure_current(self) -> None:
        """현재 이미지의 ROI를 측정하고 결과를 화면 전체에 반영한다."""
        if self._current is None:
            self._set_status("측정할 이미지가 없습니다")
            return
        record = self.session.records[self._current]
        if record.scale is None:
            self._set_status(
                "스케일이 확정되지 않았습니다. 스케일 캘리브레이션을 먼저 하세요"
            )
            return
        roi = self.image_view.current_roi()
        if roi is None:
            self._set_status("ROI를 드래그해서 지정하세요")
            return

        pixels = self._pixels[self._current]
        databar_top = self._databar_tops.get(self._current)
        if databar_top is not None and roi.y1 >= databar_top:
            self._set_status(
                f"ROI가 데이터바 영역({databar_top}행 이하)을 침범했습니다"
            )
            return

        result = measure_roi(pixels, roi, record.scale, params=self.params)
        record.roi_results = [result]  # ROI 하나만 유지한다

        self.result_panel.show_result(result)
        self.image_view.show_overlay(render_overlay(pixels, roi, result))
        self.file_panel.refresh_row(self._current)
        self._refresh_session_views()

        if result.mean_nm is None:
            self._set_status("갭 측정 불가 — 결과 패널의 경고를 확인하세요")
        else:
            self._set_status(f"갭 {result.mean_nm:.2f} nm "
                             f"(유효 {result.n_valid} 라인)")

    def calibrate_current(self) -> None:
        """메타데이터가 없는 이미지의 스케일을 사용자가 정한다."""
        if self._current is None:
            return
        pixels = self._pixels[self._current]
        dialog = CalibrationDialog(pixels,
                                   databar_top=self._databar_tops.get(self._current),
                                   parent=self)
        if dialog.exec() != CalibrationDialog.Accepted:
            return
        scale = dialog.scale_info()
        if scale is None:
            QMessageBox.warning(self, "스케일 확정 실패",
                                "실제 길이와 픽셀 거리를 모두 입력해야 합니다.")
            return
        self.session.records[self._current].scale = scale
        self._set_status(f"스케일 {scale.nm_per_px:.4f} nm/px [{scale.source}]")

    def _refresh_session_views(self) -> None:
        self.result_table.set_session(self.session)
        self.dose_plot.set_session(self.session)

    # -------------------------------------------------------------- 내보내기

    def export_summary_csv(self, path: str | Path) -> None:
        write_summary_csv(path, self.session)
        self._set_status(f"요약 CSV 저장: {path}")

    def export_lines_csv(self, path: str | Path) -> None:
        if self._current is None or not self.session.records[self._current].roi_results:
            self._set_status("먼저 측정하세요")
            return
        write_lines_csv(path, self.session.records[self._current], roi_index=0)
        self._set_status(f"라인 CSV 저장: {path}")

    def export_overlay(self, path: str | Path) -> None:
        if self._current is None:
            return
        record = self.session.records[self._current]
        roi = self.image_view.current_roi()
        if not record.roi_results or roi is None:
            self._set_status("먼저 측정하세요")
            return
        write_overlay_png(path, self._pixels[self._current], roi,
                          record.roi_results[0])
        self._set_status(f"오버레이 저장: {path}")

    def export_report(self, path: str | Path) -> None:
        Path(path).write_text(format_report(self.session), encoding="utf-8")
        self._set_status(f"리포트 저장: {path}")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: `README.md` 작성 (기존 내용을 덮어쓴다)**

````markdown
# movingking — EBL dose test 갭 분석 툴

EBL dose test로 만든 S/D 전극 패턴의 갭 폭을 SEM 이미지에서 자동으로 측정한다.
이미지 위에서 측정할 구간을 드래그하면 그 구간의 **모든 스캔라인**에서 갭 폭을 재고,
평균과 표준편차를 내며, 갭이 닫혔거나(short) 판정이 어려운 구간을 분리해 보고한다.

## 설치

```bash
pip install -e ".[gui,dev]"
```

## 실행

```bash
ebl-gap
```

## 사용 순서

1. **폴더 열기** — SEM 이미지가 든 폴더를 고른다. FEI TIFF면 픽셀 크기를
   메타데이터에서 자동으로 읽고, 파일명에서 dose 값(`pattern_320uC.tif` → 320)도
   읽는다.

   > dose 파싱은 파일명에서 `숫자 + uC` 패턴을 **앞에서부터 처음 나오는 것**으로
   > 잡는다. `sample_1000uC_and_320uC.tif`처럼 후보가 둘이면 앞의 1000을 쓴다.
   > 날짜(`20260915_320uC.tif`)는 `uC`가 붙지 않아 무시된다. 값이 틀렸으면 파일
   > 목록의 dose 칸을 직접 고치면 되고, 규칙 자체는 설정에서 바꿀 수 있다.
2. 스케일을 자동으로 못 읽었으면 **스케일 캘리브레이션**으로 확정한다. 확정 전에는
   측정이 거부된다.
3. 이미지 위에서 갭을 가로지르도록 **ROI를 드래그**한다. ROI 양 끝이 전극 평탄부에
   충분히 걸치도록 넉넉하게 잡는 것이 좋다 — 문턱을 평탄부 밝기로 잡기 때문이다.
4. **측정**을 누른다. 갭 각도를 자동으로 추정해 정렬한 뒤 ROI의 모든 행에서 폭을
   잰다. 검출된 에지가 오버레이로 그려진다.
5. 이미지마다 3~4를 반복하면 **dose-gap 곡선** 탭에 관계가 그려진다.
6. **요약 CSV / 라인 CSV / 오버레이 PNG / 요약 리포트**로 내보낸다.

## 결과 읽는 법

평균과 표준편차는 `valid` 라인만으로 계산한다. 나머지 라인은 다음과 같이 분류된다.

| 상태 | 뜻 |
|---|---|
| `short` | 갭이 닫혔다. 대비가 노이즈 수준이다 |
| `no_edge` | 문턱을 넘는 지점을 못 찾았다 |
| `multi_edge` | ROI에 패턴이 여러 개 들어왔을 수 있다 |
| `sub_resolution` | 갭이 3픽셀 미만이다. 이 배율로는 측정할 수 없다 |
| `outlier` | 다른 라인들과 크게 어긋난다 |

`low_confidence`는 상태가 아니라 플래그다. 갭이 10픽셀 미만이라는 뜻이고, 값은
평균에 정상적으로 들어간다. **이 플래그가 많이 뜨면 배율을 올려 다시 촬영하는 것이
정밀도를 올리는 가장 확실한 방법이다.** 100 nm 이하를 재고 있다면 자주 보게 된다.

`nm/px`와 그 출처(`fei_metadata` / `scalebar_auto` / `manual`)는 화면과 CSV에 항상
따라붙는다. 스케일 출처를 모르는 계측값은 나중에 재현할 수 없기 때문이다.

## 구조

- `ebl_gap/` — Qt에 의존하지 않는 측정 엔진. 화면 없이 단독으로 테스트된다
- `ebl_gap_gui/` — PySide6 + pyqtgraph GUI. 계산 로직이 없다
- `tests/synth.py` — 정답 갭 폭을 아는 합성 SEM 이미지 생성기
- `docs/superpowers/specs/` — 설계 문서

측정 엔진은 노트북에서도 바로 쓸 수 있다.

```python
from ebl_gap.loader import load_image
from ebl_gap.measure import measure_roi
from ebl_gap.types import Roi

loaded = load_image("pattern_320uC.tif")
result = measure_roi(loaded.pixels, Roi(400, 300, 700, 600), loaded.record.scale)
print(result.mean_nm, result.n_valid, result.warnings)
```

## 테스트

```bash
pytest
```

`tests/test_accuracy.py`가 품질 게이트다. 갭 20~100 nm, 기울기 0~10도, 노이즈 3수준의
모든 조합에서 측정값이 참값의 ±1픽셀 안에 들어오는지 확인한다. **허용 오차를 늘려서
통과시키면 이 테스트의 의미가 사라진다.**
````

- [ ] **Step 6: 전체 테스트 실행**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest -q`
Expected: 모두 통과

- [ ] **Step 7: 엔진이 Qt에 의존하지 않는지 확인**

이 프로젝트의 핵심 제약이므로 명시적으로 검증한다.

Run:
```bash
grep -rE "PySide6|pyqtgraph|ebl_gap_gui" ebl_gap/ && echo "제약 위반" || echo "OK: 엔진은 Qt에 의존하지 않음"
```
Expected: `OK: 엔진은 Qt에 의존하지 않음`

- [ ] **Step 8: 커밋**

```bash
git add ebl_gap_gui/app.py README.md tests/test_gui_app.py
git commit -m "feat: wire main window, add entry point and usage documentation"
```

---

### Task 18 수정 라운드: 중복 선택 호출과 README의 허위 안내

리뷰가 실행으로 확인한 두 가지를 고친다.

**1) `open_folder`가 `select_image(0)`을 두 번 부른다.**

`file_panel.set_records()`는 Task 15 수정에서 "행 0으로 다시 불러와도 결과 패널이
갱신되도록" 항상 `selection_changed(0)`을 **정확히 한 번** 동기 발신하게 만들어
두었다. 그래서 `set_records()` 호출만으로 이미 `select_image(0)`이 실행된다.
그 뒤 줄에서 다시 명시적으로 부르는 것은 폴더를 열 때마다 전체 이미지 렌더,
`autoRange()`, ROI 리셋, `roi_changed` -> `clear_overlay` 왕복을 한 번씩 더
시킨다. 리뷰어가 호출 횟수를 직접 세어 `[0, 0]` 2회를 확인했다.

경로는 **신호 하나만** 남긴다. 명시 호출 쪽을 지우는 이유는, 다시 불러오기
경로(같은 행 0으로 재로딩)에서는 신호만이 유일한 갱신 수단이고 명시 호출은
`open_folder`에만 있기 때문이다. 둘 중 하나를 지운다면 모든 경로를 덮는 쪽을
남겨야 한다.

- [x] **Step 1: 중복 호출을 잡는 테스트를 먼저 쓴다 (RED)**

`tests/test_gui_app.py`에 추가한다. 헬퍼가 아니라 `open_folder`라는 실제 사용자
경로를 통해서 센다.

```python
def test_open_folder_selects_first_image_exactly_once(qapp, tmp_path,
                                                      monkeypatch):
    """폴더 열기가 첫 장을 정확히 한 번만 선택한다.

    set_records가 selection_changed(0)을 동기 발신하므로 명시 호출을 더하면
    렌더와 ROI 리셋이 두 배로 돈다. 상태가 깨지지는 않지만 낭비이고, 무엇보다
    '정확히 한 번'이라는 set_records의 불변식을 무너뜨린다.
    """
    _write_synth_tif(tmp_path / "a_320uC.tif")
    _write_synth_tif(tmp_path / "b_340uC.tif")
    window = MainWindow()

    calls: list[int] = []
    original = window.select_image
    monkeypatch.setattr(window, "select_image",
                        lambda index: (calls.append(index), original(index))[1])

    window.open_folder(tmp_path)

    assert calls == [0]
    assert window._current == 0
```

`_write_synth_tif`는 이미 `tests/test_gui_app.py`에 있는 헬퍼를 쓴다. 이름이
다르면 그 파일의 기존 헬퍼를 그대로 쓰고, 없으면 `tests/synth.py`의
`synth_gap_image`로 만들어 `tifffile.imwrite`로 저장한다.

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py -q -k exactly_once`
Expected: 실패한다. `calls == [0, 0]`

- [x] **Step 2: 명시 호출을 지운다 (GREEN)**

`ebl_gap_gui/app.py`의 `open_folder`에서 `self.select_image(0)` 한 줄만 지운다.
상태 메시지와 빈 폴더 분기는 그대로 둔다.

```python
        self.file_panel.set_records(self.session.records)
        for index, pixels in self._pixels.items():
            self.file_panel.set_thumbnail(index, pixels)
        self._refresh_session_views()
        if self.session.records:
            # 첫 장 선택은 set_records가 발신하는 selection_changed(0)이 한다.
            # 여기서 또 부르면 렌더와 ROI 리셋이 두 번 돈다.
            self._set_status(f"{len(self.session.records)}장 불러옴")
        else:
            ...
```

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py -q`
Expected: 모두 통과. 특히 기존의 "폴더를 열면 첫 장이 보인다" 계열 테스트가
그대로 통과해야 한다 — 신호 경로만으로 선택이 실제로 일어난다는 증거다.

**2) README가 없는 설정 화면을 가리킨다.**

README 28행의 "규칙 자체는 설정에서 바꿀 수 있다"는 거짓이다. GUI에는 설정 화면이
없고 `dose_pattern`은 `load_image()`의 키워드 인자로만 닿는다. 리뷰어가
`ebl_gap_gui/*.py` 전체에서 `설정`과 `dose_pattern`을 grep해 0건을 확인했다.

- [x] **Step 3: README를 실제 경로로 고친다**

```
   > dose 파싱은 파일명에서 `숫자 + uC` 패턴을 **앞에서부터 처음 나오는 것**으로
   > 잡는다. `sample_1000uC_and_320uC.tif`처럼 후보가 둘이면 앞의 1000을 쓴다.
   > 날짜(`20260915_320uC.tif`)는 `uC`가 붙지 않아 무시된다. 값이 틀렸으면 파일
   > 목록의 dose 칸을 직접 고치면 된다. 규칙 자체를 바꾸려면 파이썬에서
   > `load_image(path, dose_pattern=r"d(\d+)")`처럼 직접 넘긴다 — GUI에는 이
   > 설정이 없다.
```

Run: `grep -n "설정에서" README.md`
Expected: 출력 없음

- [x] **Step 4: 전체 테스트**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest -q`
Expected: 모두 통과

- [x] **Step 5: 커밋**

```bash
git add ebl_gap_gui/app.py README.md tests/test_gui_app.py
git commit -m "Remove the duplicate first-image selection and fix the README dose note"
```

---

### Task 19: 라인 프로파일 미니 플롯과 파일 목록 썸네일

스펙 7절이 요구하는 UI 요소 중 남은 두 가지다. 프로파일 미니 플롯은 장식이 아니라
진단 도구다 — 측정값이 이상할 때 문턱이 어디에 섰는지 눈으로 확인할 수 있는 유일한
수단이다.

**Files:**
- Create: `ebl_gap_gui/profile_plot.py`
- Modify: `ebl_gap_gui/panels.py` (FilePanel에 썸네일 추가)
- Modify: `ebl_gap_gui/app.py` (미니 플롯 배선, 썸네일 채우기)
- Test: `tests/test_gui_profile_plot.py`
- Test: `tests/test_gui_panels.py` (썸네일 테스트 추가)
- Test: `tests/test_gui_app.py` (배선 테스트 추가)

**Interfaces:**
- Consumes: `LineResult` (Task 1), `extract_profiles` (Task 4), `analyze_profile` (Task 3)
- Produces: `ProfilePlot(QWidget)` — `show_line(profile, line, analysis)`, `clear()`, `has_curve() -> bool`, `title_text() -> str`; `to_thumbnail_icon(pixels, size=64) -> QIcon` (panels.py); `FilePanel.set_thumbnail(index, pixels)`, `FilePanel.has_thumbnail(index) -> bool`; `MainWindow.show_line(row)`, `MainWindow.profile_plot`

- [ ] **Step 1: 실패하는 프로파일 플롯 테스트 작성**

`tests/test_gui_profile_plot.py`:

```python
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from ebl_gap.edges import analyze_profile  # noqa: E402
from ebl_gap.types import LineResult  # noqa: E402
from ebl_gap_gui.profile_plot import ProfilePlot  # noqa: E402


def sample_profile(n=201, gap=20.0):
    x = np.arange(n, dtype=float)
    d = np.abs(x - (n - 1) / 2.0)
    return np.where(d < gap / 2.0, 40.0, 200.0)


def valid_line(analysis, row=7):
    return LineResult(row=row, left_px=analysis.left_px,
                      right_px=analysis.right_px, width_px=analysis.width_px,
                      width_nm=analysis.width_px * 3.0, status="valid",
                      flags=frozenset(), reason="")


def test_starts_empty(qapp):
    assert ProfilePlot().has_curve() is False


def test_showing_a_line_draws_the_profile(qapp):
    profile = sample_profile()
    analysis = analyze_profile(profile)
    plot = ProfilePlot()
    plot.show_line(profile, valid_line(analysis), analysis)
    assert plot.has_curve() is True


def test_title_reports_the_row_width_and_status(qapp):
    profile = sample_profile()
    analysis = analyze_profile(profile)
    plot = ProfilePlot()
    plot.show_line(profile, valid_line(analysis, row=42), analysis)
    title = plot.title_text()
    assert "42" in title
    assert "valid" in title
    assert "nm" in title


def test_title_says_so_for_an_unmeasured_line(qapp):
    profile = np.full(201, 180.0)
    analysis = analyze_profile(profile)
    line = LineResult(row=3, left_px=None, right_px=None, width_px=None,
                      width_nm=None, status="short", flags=frozenset(),
                      reason="대비 없음")
    plot = ProfilePlot()
    plot.show_line(profile, line, analysis)
    assert "short" in plot.title_text()
    assert plot.has_curve() is True


def test_clear_removes_the_curve(qapp):
    profile = sample_profile()
    analysis = analyze_profile(profile)
    plot = ProfilePlot()
    plot.show_line(profile, valid_line(analysis), analysis)
    plot.clear()
    assert plot.has_curve() is False
    assert plot.title_text() == ""
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_profile_plot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebl_gap_gui.profile_plot'`

- [ ] **Step 3: `ebl_gap_gui/profile_plot.py` 구현**

```python
"""스캔라인 한 줄의 밝기 프로파일과 문턱 위치를 보여주는 진단 플롯.

측정값이 이상할 때 문턱이 어디에 섰는지, 에지가 어디로 잡혔는지 눈으로 확인할 수
있는 유일한 수단이다.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ebl_gap.edges import ProfileAnalysis
from ebl_gap.types import LineResult


class ProfilePlot(QWidget):
    """프로파일 곡선 + 좌우 문턱 + 검출된 에지 위치."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._has_curve = False

        self._title = QLabel("")
        self._title.setWordWrap(True)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "측정 방향 위치", units="px")
        self._plot.setLabel("left", "밝기")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setMaximumHeight(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._plot)

    def show_line(self, profile, line: LineResult,
                  analysis: ProfileAnalysis) -> None:
        values = np.asarray(profile, dtype=np.float64)
        self._plot.clear()
        self._plot.plot(np.arange(values.size, dtype=float), values,
                        pen=pg.mkPen("#1f77b4", width=1))

        # 좌우 문턱을 따로 그린다. 조명이 기울면 두 선의 높이가 달라진다.
        for level, color in (
            (analysis.i_lo + 0.5 * (analysis.i_hi_left - analysis.i_lo), "#888888"),
            (analysis.i_lo + 0.5 * (analysis.i_hi_right - analysis.i_lo), "#bbbbbb"),
        ):
            self._plot.addItem(pg.InfiniteLine(pos=level, angle=0,
                                               pen=pg.mkPen(
                                                   color,
                                                   style=Qt.PenStyle.DashLine)))

        for edge in (line.left_px, line.right_px):
            if edge is not None:
                self._plot.addItem(pg.InfiniteLine(
                    pos=edge, angle=90, pen=pg.mkPen("#2ca02c", width=2)))

        if line.width_nm is None:
            detail = "폭 측정 없음"
        else:
            detail = f"폭 {line.width_px:.2f} px = {line.width_nm:.2f} nm"
        flags = f" [{' '.join(sorted(line.flags))}]" if line.flags else ""
        reason = f" — {line.reason}" if line.reason else ""
        self._title.setText(
            f"행 {line.row} · {line.status}{flags} · {detail}{reason}"
        )
        self._has_curve = True

    def clear(self) -> None:
        self._plot.clear()
        self._title.setText("")
        self._has_curve = False

    def has_curve(self) -> bool:
        return self._has_curve

    def title_text(self) -> str:
        return self._title.text()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_profile_plot.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: 썸네일 테스트를 `tests/test_gui_panels.py` 끝에 추가**

```python
def test_thumbnail_starts_absent(qapp):
    panel = FilePanel()
    panel.set_records(make_records())
    assert panel.has_thumbnail(0) is False


def test_setting_a_thumbnail_attaches_an_icon(qapp):
    import numpy as np

    panel = FilePanel()
    panel.set_records(make_records())
    panel.set_thumbnail(0, np.arange(64 * 64, dtype=float).reshape(64, 64))
    assert panel.has_thumbnail(0) is True
    assert panel.has_thumbnail(1) is False


def test_empty_pixels_do_not_attach_a_thumbnail(qapp):
    import numpy as np

    panel = FilePanel()
    panel.set_records(make_records())
    panel.set_thumbnail(0, np.empty((0, 0)))
    assert panel.has_thumbnail(0) is False


def test_thumbnail_survives_a_row_refresh(qapp):
    import numpy as np

    panel = FilePanel()
    panel.set_records(make_records())
    panel.set_thumbnail(0, np.full((32, 32), 120.0))
    panel.refresh_row(0)
    assert panel.has_thumbnail(0) is True
```

- [ ] **Step 6: `panels.py`에 썸네일 지원 추가**

`ebl_gap_gui/panels.py`의 import에 다음을 더한다:

```python
import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QImage, QPixmap
```

모듈 수준에 헬퍼를 추가한다:

```python
THUMBNAIL_SIZE = 56


def to_thumbnail_icon(pixels, size: int = THUMBNAIL_SIZE) -> QIcon:
    """2차원 밝기 배열을 목록에 넣을 회색조 아이콘으로 만든다."""
    array = np.asarray(pixels, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        return QIcon()

    lo, hi = float(array.min()), float(array.max())
    span = hi - lo if hi > lo else 1.0
    gray = np.clip((array - lo) / span * 255.0, 0, 255).astype(np.uint8)

    step = max(1, max(gray.shape) // size)
    gray = np.ascontiguousarray(gray[::step, ::step])
    height, width = gray.shape
    # .copy()로 numpy 버퍼에서 떼어낸다. 떼지 않으면 배열이 해제될 때 화면이 깨진다.
    image = QImage(gray.data, width, height, width,
                   QImage.Format_Grayscale8).copy()
    return QIcon(QPixmap.fromImage(image))
```

`FilePanel.__init__`의 `self._records = []` 다음 줄에 썸네일 보관소를 추가한다:

```python
        self._thumbnails: dict[int, QIcon] = {}
```

같은 `__init__`에서 아이콘이 보이도록 테이블을 설정한다 (`self._table` 생성 직후):

```python
        self._table.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
```

`set_records`의 `self._records = list(records)` 다음 줄에 보관소를 비운다:

```python
        self._thumbnails.clear()
```

`_fill_row`의 `name` 항목을 만든 직후, `setItem` 전에 아이콘을 붙인다:

```python
        icon = self._thumbnails.get(index)
        if icon is not None:
            name.setIcon(icon)
```

클래스 끝에 메서드 두 개를 추가한다:

```python
    def set_thumbnail(self, index: int, pixels) -> None:
        """이미지 미리보기를 목록 행에 붙인다."""
        icon = to_thumbnail_icon(pixels)
        if icon.isNull():
            self._thumbnails.pop(index, None)
        else:
            self._thumbnails[index] = icon
        self.refresh_row(index)

    def has_thumbnail(self, index: int) -> bool:
        return index in self._thumbnails
```

- [ ] **Step 7: 패널 테스트 통과 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_panels.py -v`
Expected: PASS (23 passed)

- [ ] **Step 8: 배선 테스트를 `tests/test_gui_app.py` 끝에 추가**

```python
def test_open_folder_attaches_thumbnails(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    assert window.file_panel.has_thumbnail(0) is True
    assert window.file_panel.has_thumbnail(1) is True


def test_measuring_shows_a_representative_profile(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    assert window.profile_plot.has_curve() is True
    assert "valid" in window.profile_plot.title_text()


def test_show_line_switches_to_the_requested_row(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    window.show_line(5)
    assert "행 5" in window.profile_plot.title_text()


def test_show_line_is_ignored_before_measuring(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.show_line(5)
    assert window.profile_plot.has_curve() is False


def test_selecting_another_image_clears_the_profile_plot(qapp, folder):
    window = MainWindow()
    window.open_folder(folder)
    window.select_image(0)
    window.measure_current()
    window.select_image(1)
    assert window.profile_plot.has_curve() is False
```

- [ ] **Step 9: `app.py`에 미니 플롯과 썸네일 배선 추가**

import에 다음을 더한다:

```python
from ebl_gap.edges import analyze_profile
from ebl_gap.profile import extract_profiles
from ebl_gap_gui.profile_plot import ProfilePlot
```

`__init__`에서 `self.result_panel = ResultPanel()` 다음에 위젯을 만들고,
`self._profiles: np.ndarray | None = None` 을 상태에 추가한다:

```python
        self.profile_plot = ProfilePlot()
        self._profiles: np.ndarray | None = None
```

결과 패널과 미니 플롯을 세로로 묶어 오른쪽 칸에 넣는다. `splitter.addWidget(self.result_panel)` 을 다음으로 바꾼다:

```python
        right = QSplitter(Qt.Vertical)
        right.addWidget(self.result_panel)
        right.addWidget(self.profile_plot)
        right.setSizes([500, 260])
        splitter.addWidget(right)
```

`open_folder`의 루프에서 썸네일을 붙인다. `self._databar_tops[index] = loaded.databar_top` 다음, 루프 밖의 `self.file_panel.set_records(...)` 호출 뒤에 다음을 넣는다:

```python
        for index, pixels in self._pixels.items():
            self.file_panel.set_thumbnail(index, pixels)
```

`select_image`에서 이미지가 바뀌면 미니 플롯을 지운다. `self.image_view.set_image(self._pixels[index])` 다음 줄에 추가한다:

```python
        self.profile_plot.clear()
        self._profiles = None
```

`measure_current`에서 프로파일을 보관하고 대표 라인을 띄운다.
**`record.roi_results = [result]` 다음에** 넣는다 — `show_line`이 결과를 읽으므로
결과가 기록된 뒤여야 한다:

```python
        self._profiles = extract_profiles(pixels, roi, result.angle_deg,
                                          along_average=self.params.along_average)
        self._show_representative_line(result)
```

클래스에 메서드 두 개를 추가한다:

```python
    def _show_representative_line(self, result) -> None:
        """대표 라인 하나를 미니 플롯에 띄운다.

        폭이 중앙값에 가장 가까운 valid 라인을 고른다. 평균이 어떤 프로파일에서
        나왔는지 보여주는 것이 목적이므로 첫 줄보다 이쪽이 낫다.
        """
        valid = [ln for ln in result.lines
                 if ln.status == "valid" and ln.width_nm is not None]
        if not valid:
            target = result.lines[0].row if result.lines else None
        else:
            widths = sorted(ln.width_nm for ln in valid)
            median = widths[len(widths) // 2]
            target = min(valid, key=lambda ln: abs(ln.width_nm - median)).row
        if target is not None:
            self.show_line(target)

    def show_line(self, row: int) -> None:
        """특정 스캔라인의 프로파일을 미니 플롯에 띄운다."""
        if self._profiles is None or self._current is None:
            return
        record = self.session.records[self._current]
        if not record.roi_results:
            return
        result = record.roi_results[0]
        if not (0 <= row < len(result.lines)) or row >= self._profiles.shape[0]:
            return
        profile = self._profiles[row]
        self.profile_plot.show_line(profile, result.lines[row],
                                    analyze_profile(profile,
                                                    **self.params.edge_kwargs))
```

- [ ] **Step 10: 배선 테스트 통과 확인**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py -v`
Expected: PASS (17 passed)

- [ ] **Step 11: 전체 테스트 실행**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest -q`
Expected: 모두 통과

- [ ] **Step 12: 엔진이 Qt에 의존하지 않는지 다시 확인**

Run:
```bash
grep -rE "PySide6|pyqtgraph|ebl_gap_gui" ebl_gap/ && echo "제약 위반" || echo "OK: 엔진은 Qt에 의존하지 않음"
```
Expected: `OK: 엔진은 Qt에 의존하지 않음`

- [ ] **Step 13: 커밋과 푸시**

```bash
git add ebl_gap_gui/ tests/
git commit -m "feat: add scanline profile diagnostic plot and file list thumbnails"
git push -u origin claude/intelligent-cray-f9eq7a
```

---

## 실제 Inspect F 이미지를 확보한 뒤 할 일

이 계획은 실제 촬영 파일 없이 작성되었다. FEI TIFF 포맷 사양에 맞춰 파서를 만들고
합성 픽스처로 검증하지만, 실제 파일에서 섹션 이름이나 키 이름이 다를 수 있다.
실제 이미지를 얻으면 다음 순서로 맞춘다.

1. 헤더를 눈으로 확인한다.
   ```bash
   python -c "
   from ebl_gap.metadata import read_fei_metadata
   import json, sys
   print(json.dumps(read_fei_metadata(sys.argv[1]), indent=2, ensure_ascii=False))
   " 실제파일.tif
   ```
2. `MetadataNotFoundError`가 나면 태그 번호를 확인한다.
   ```bash
   python -c "
   import tifffile, sys
   with tifffile.TiffFile(sys.argv[1]) as t:
       for tag in t.pages[0].tags:
           print(tag.code, tag.name, str(tag.value)[:120])
   " 실제파일.tif
   ```
   나온 태그 번호를 `ebl_gap/metadata.py`의 `FEI_TAG_CODES`에 추가하고,
   `tests/test_metadata.py`의 `FEI_INI` 픽스처를 실제 헤더 텍스트로 교체한다.
3. `PixelWidth`와 `HorFieldsize`의 실제 섹션 이름이 다르면
   `scale_from_metadata`의 후보 목록에 추가한다.
4. 실제 이미지에서 측정한 갭 값이 눈으로 본 것과 어긋나면, 스펙 5장의 임계값
   (`contrast_k=5.0`, `mad_k=3.5`, `min_width_px=3.0`)을 실제 데이터에 맞춰 조정하고
   그 근거를 스펙에 적는다.
