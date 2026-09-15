# EBL Dose Test 갭 분석 툴 — 설계 문서

작성일: 2026-09-15

## 1. 목적

전자빔 리소그래피(EBL) dose test로 제작한 Source/Drain 전극 패턴의 **갭 폭을 SEM
이미지에서 자동으로 측정**하는 데스크톱 툴을 만든다. 사용자가 이미지 위에서 측정할
구간을 드래그하면 그 구간의 모든 스캔라인에서 갭 폭을 재고, 통계를 내며, 갭이
형성되지 않았거나 short가 발생한 구간을 분리해 보고한다. 여러 dose 조건의 이미지를
한 세션에서 처리해 dose–gap 관계를 확인하는 것이 최종 목적이다.

## 2. 사용 환경과 제약

| 항목 | 내용 |
|---|---|
| SEM 장비 | FEI Inspect F |
| 이미지 형식 | FEI TIFF (INI 형식 메타데이터 블록 내장) |
| 목표 갭 크기 | 100 nm 이하 |
| 실행 형태 | Python 데스크톱 앱 (PySide6) |
| 사용자 | 연구실 개인 PC, 오프라인 사용 가능해야 함 |

목표 갭이 100 nm 이하라는 점이 설계 전반을 지배한다. Inspect F를 1024x884 해상도로
100k 배율 촬영하면 픽셀당 대략 3 nm이므로 30 nm 갭은 10픽셀 남짓이다. 이 영역에서는
결과를 눈으로 검증할 수 없으므로 **정답을 아는 합성 이미지로 알고리즘 정확도를
회귀 테스트에 고정하는 것이 필수**다.

## 3. 아키텍처

측정 엔진과 GUI를 완전히 분리한다. 엔진은 Qt를 import하지 않는 순수 파이썬 패키지이고,
GUI는 엔진이 반환한 데이터 타입을 그리기만 한다. 위젯 코드에 계산 로직을 두지 않는다.

이 분리가 주는 것은 **검증 가능성**이다. 화면이 없는 CI 환경에서 엔진 전체를 pytest로
돌릴 수 있고, 나중에 Jupyter에서 `from ebl_gap import measure_roi`로 재사용하거나
CLI 배치를 붙이기 쉽다.

```
ebl_gap/                    # Qt 무의존 측정 엔진 (numpy + scipy)
  types.py       ScaleInfo, LineResult, RoiResult, ImageRecord
  metadata.py    FEI TIFF INI 블록 파싱 -> PixelWidth(m/px), HV, HFW, 데이터바 경계
  scalebar.py    폴백: 데이터바 스케일바 자동 검출 / 사용자 2점 수동 캘리브레이션
  orientation.py ROI 내 에지점에 Theil-Sen 직선 피팅 -> 갭 축 각도 theta
  profile.py     theta에 수직인 방향으로 스캔라인 밝기 프로파일 추출
  edges.py       50% 문턱 + 선형보간 서브픽셀 에지 위치
  measure.py     오케스트레이션: ROI 하나 -> LineResult 리스트 -> RoiResult
  classify.py    라인 상태 판정 (valid / short / no_edge / ...)
  stats.py       정상 라인 통계, MAD 기반 이상치 제거
  dataset.py     세션 관리, 파일명 dose 파싱, dose-gap 집계
  export.py      CSV(요약/원시), 오버레이 PNG, 요약 텍스트

ebl_gap_gui/                # PySide6 + pyqtgraph, 계산 로직 없음
  app.py         메인 윈도우, 레이아웃
  file_panel.py  파일 목록, 썸네일, dose 편집
  image_view.py  이미지 표시, 드래그 ROI, 오버레이
  result_table.py 이미지별 결과 테이블
  dose_plot.py   dose-gap 곡선

tests/
  synth.py       정답 갭 폭을 아는 합성 SEM 이미지 생성기
  test_*.py
```

### 3.1 데이터 흐름

1. 폴더 열기 -> 각 TIFF에서 `metadata.py`가 `PixelWidth`를 읽어 nm/px 확정
2. 이미지 표시 -> 사용자가 ROI 드래그
3. `orientation.py`가 갭 축 각도 `theta` 산출
4. ROI를 `-theta` 회전 정렬
5. 정렬된 ROI의 **모든 행을 1픽셀 간격으로** 훑으며 `profile.py` + `edges.py`가
   각 행의 갭 폭을 px 단위로 측정
6. nm 환산 -> `classify.py` 라인별 상태 판정
7. `stats.py`가 정상 라인만으로 평균/표준편차 산출
8. 테이블, 오버레이, CSV, dose 곡선에 반영

### 3.2 데이터 타입

```python
ScaleInfo(nm_per_px: float, source: str)
# source in {"fei_metadata", "scalebar_auto", "manual"}

LineResult(row: int, left_px: float, right_px: float,
           width_px: float | None, width_nm: float | None,
           status: str, flags: set[str], reason: str)
# status in {"valid", "short", "no_edge", "multi_edge", "sub_resolution", "outlier"}
# flags  in {"low_confidence"}   -- status와 독립. 통계 포함 여부를 바꾸지 않는다.

RoiResult(mean_nm: float | None, std_nm: float | None,
          n_valid: int, n_short: int, n_uncertain: int,
          angle_deg: float, lines: list[LineResult], warnings: list[str])

ImageRecord(path: Path, scale: ScaleInfo, dose: float | None,
            roi_results: list[RoiResult])
```

`ScaleInfo.source`는 화면과 CSV에 **항상** 표시한다. 스케일 출처를 모르는 계측값은
나중에 재현할 수 없기 때문이다.

## 4. 측정 알고리즘

### 4.1 스케일 확정

FEI TIFF는 파일 끝에 INI 형식 텍스트 블록을 붙인다. `[Scan]` 섹션의 `PixelWidth`가
**미터 단위**로 들어있으므로 `nm_per_px = PixelWidth * 1e9`로 환산한다. 스케일바를
픽셀로 재서 추정하는 것보다 정확하다.

검증 장치: `PixelWidth * 이미지_가로폭 ~= HFW`(Horizontal Field Width)가 성립하는지
대조하고 5% 이상 어긋나면 경고한다. 이미지가 크롭되었거나 리사이즈된 경우를 잡아낸다.

메타데이터가 없으면(PNG 변환본, 잘라낸 이미지) 폴백 두 단계:

1. 데이터바 영역에서 스케일바 막대를 자동 검출해 픽셀 길이를 재고, 사용자에게
   실제 길이(예: "1 um")를 입력받는다.
2. 사용자가 이미지 위에서 직접 두 점을 클릭하고 그 거리를 입력한다.

어느 경로를 썼는지는 `ScaleInfo.source`로 결과에 따라다닌다.

ROI가 데이터바 영역을 침범하면 측정을 거부한다.

### 4.2 갭 각도 추정

ROI 안에서 행별로 거칠게 에지쌍을 찾아 좌/우 에지점 집합을 만들고, 각각에
**Theil-Sen 추정**(`scipy.stats.theilslopes`)으로 직선을 피팅한다. 최소자승 대신
Theil-Sen을 쓰는 이유는 에지 검출이 몇 줄 실패해도 각도가 끌려가지 않기 때문이다.
두 직선 각도의 평균이 `theta`이며, 화면에 표시하고 사용자가 수동으로 고정할 수 있다.

기울어진 갭을 수평 스캔라인으로 측정하면 폭이 `1/cos(theta)`만큼 과대평가된다.
1도에서 0.015%, 5도에서 0.4%, 10도에서 1.5%다.

### 4.3 스캔라인 프로파일

ROI를 `-theta` 회전시켜 정렬한 뒤 `scipy.ndimage.map_coordinates`(order=1, 이중선형)로
리샘플하고, **모든 행을 1픽셀 간격으로** 훑는다.

갭 축 방향 이동평균 옵션을 두되 **기본값은 1(평균 없음)**이다. 요청된 최대 해상도
측정을 기본으로 유지하고, 노이즈가 심한 이미지에서만 사용자가 3 정도로 올린다.

### 4.4 50% 문턱 서브픽셀 에지 검출

각 프로파일 `p[x]`에 대해:

1. 전극 평탄부 밝기를 프로파일 **양 끝 20% 구간의 중앙값**으로 좌/우 각각 구한다
   (`I_hi_L`, `I_hi_R`). 끝쪽을 쓰는 것이 edge-brightening 회피의 핵심이다. 에지에서
   튀어오른 밝기를 기준으로 삼으면 문턱이 위로 밀려 갭이 좁게 측정된다.
2. 갭 바닥 `I_lo`는 **2단계로** 구한다. 중앙 영역의 단순 분위수를 쓰면 안 된다 —
   폭 300픽셀 ROI에 20픽셀 갭이 있으면 중앙 영역의 하위 20%에도 금속 픽셀이 대부분
   들어와 `I_lo`가 금속 밝기로 잡히고 문턱이 통째로 틀어진다. 100 nm 이하를 재는 이
   툴에서는 이것이 정상 상황이므로 반드시 2단계로 구한다.
   - 1차: 중앙 영역(양 끝 flat 구간을 제외한 나머지)의 **최소값**을 `I_lo`로 두고
     아래 3~4단계를 수행해 갭 위치를 대략 잡는다.
   - 2차: 1차로 얻은 갭의 **중앙 절반 구간**(`left + 0.25w` ~ `right - 0.25w`)의
     중앙값을 `I_lo`로 다시 구하고 문턱을 보정한 뒤 3~4단계를 재실행한다.
   - 중앙 절반에 샘플이 1개 미만이면(매우 좁은 갭) 1차 결과를 그대로 쓴다.
3. 문턱을 좌우 따로 잡는다: `T_L = I_lo + 0.5 * (I_hi_L - I_lo)`, `T_R`도 동일.
   SEM 이미지는 좌우 밝기가 기울어지는 경우가 흔해 단일 문턱보다 안전하다.
4. 갭 최저점(프로파일 중앙 60% 구간의 최소값 위치)에서 좌/우로 진행하며 처음 문턱을 상향 교차하는 구간에서
   **선형보간으로 서브픽셀 위치**를 확정하고 `width_px = right_px - left_px`.
5. 회전 정렬을 이미 했으므로 `width_nm = width_px * nm_per_px`이고 cos 보정은 없다.

문턱 비율(기본 50%)은 UI에서 조절 가능하게 노출한다.

## 5. 이상 상태 판정

판정은 **상태(status)** 하나와 **플래그(flags)** 집합으로 나뉜다. 이 둘을 섞지 않는
것이 중요하다. 상태는 라인이 통계에 들어가는지를 혼자서 결정하고, 플래그는 통계
포함 여부를 바꾸지 않는 부가 정보다.

### 5.1 상태 (status) — 통계 포함 여부를 결정

위에서부터 순서대로 판정하고 처음 걸리는 것을 채택한다.

| status | 조건 | 통계 | 분류 |
|---|---|---|---|
| `short` | `I_hi - I_lo < 5 * sigma_noise` 이고 프로파일에 국소 최소 구조가 없음 | 제외 | short |
| `no_edge` | 좌 또는 우에서 문턱 상향 교차를 찾지 못함 | 제외 | 판정보류 |
| `multi_edge` | 히스테리시스 적용 후에도 한쪽 교차가 2회 이상 | 제외 | 판정보류 |
| `sub_resolution` | `width_px < 3` | 제외 | 판정보류 |
| `outlier` | 위를 통과한 라인 중 `abs(w - median) > 3.5 * MAD` | 제외 | 이상치 |
| `valid` | 위 어디에도 걸리지 않음 | 포함 | 정상 |

`sigma_noise`는 전극 평탄부(프로파일 양 끝 20% 구간)의 밝기 표준편차로 추정한다.

`multi_edge`의 교차 횟수를 셀 때는 잡음에 의한 중복 교차를 막기 위해 히스테리시스를
적용한다. 상향 교차로 인정하려면 `T + 0.1 * (I_hi - I_lo)` 위로 올라가야 하고, 다음
교차를 세려면 그전에 `T - 0.1 * (I_hi - I_lo)` 아래로 내려와야 한다.

`outlier` 판정의 median과 MAD는 `short`/`no_edge`/`multi_edge`/`sub_resolution`을
걸러낸 나머지 라인들로 계산한다.

### 5.2 플래그 (flags) — 통계에 영향 없음

| flag | 조건 | 의미 |
|---|---|---|
| `low_confidence` | `status == "valid"` 이고 `3 <= width_px < 10` | 값은 유효하나 정밀도가 픽셀 한계에 가까움 |

목표 갭이 100 nm 이하이므로 이 플래그는 자주 켜진다. 오류가 아니라 **"배율을 올리면
정밀도가 올라간다"는 행동 가능한 안내**로 표시한다. `low_confidence` 라인은 평균에
정상적으로 포함되며, ROI 요약에는 해당 라인 수를 따로 적는다.

### 5.3 통계 산출

평균과 표준편차는 `status == "valid"` 라인만으로 계산한다(`low_confidence` 플래그가
켜진 라인 포함). `n_valid`, `n_short`, `n_uncertain`(= `no_edge` + `multi_edge` +
`sub_resolution` + `outlier`)을 각각 보고하고, 비율도 함께 표시한다.

### 5.4 ROI 요약 경고

| 조건 | 메시지 |
|---|---|
| short 비율 >= 5% | short 발생 구간 있음, 확인 필요 |
| 판정보류 비율 >= 20% | 측정 신뢰도 낮음, ROI 재설정 권장 |
| 유효 라인 < 10 | 유효 라인 부족, 평균 신뢰 불가 |
| std / mean > 0.2 | 갭 폭 편차 큼, 패턴 불균일 의심 |
| `ScaleInfo.source != "fei_metadata"` | 스케일 출처가 자동 메타데이터가 아님 |

## 6. 에러 처리

| 상황 | 동작 |
|---|---|
| FEI 메타데이터 없음 | 스케일바 폴백 다이얼로그. 캘리브레이션을 끝내기 전까지는 px 단위로만 표시하고 CSV의 nm 열을 비운다 |
| `PixelWidth * width`와 HFW 불일치 | 경고 표시, 사용자가 수동 캘리브레이션 선택 가능 |
| ROI가 데이터바 침범 | 측정 거부, 사유 표시 |
| 세션 내 배율 혼재 | dose 곡선에 경고 표시 |
| 파일명 dose 파싱 실패 | 테이블 셀을 비우고 사용자 입력 요구. 곡선은 dose가 있는 이미지만 |
| TIFF 읽기 실패 | 해당 파일을 목록에 오류로 표시하고 나머지는 계속 처리 |

## 7. UI 구성

- **좌측**: 파일 목록 (썸네일, dose 값, 측정 상태 아이콘)
- **중앙**: pyqtgraph ImageView + 드래그/리사이즈/회전 ROI, 오버레이(검출된 좌우
  에지선, 각도, short 라인은 빨강)
- **우측**: ROI 결과 패널 (평균 +- 표준편차, 유효/short/보류 라인 수, theta, nm/px와
  출처) + 선택한 라인의 프로파일 미니 플롯
- **하단 탭**: 결과 테이블 / dose-gap 곡선
- **내보내기**: 이미지별 요약 CSV, 라인별 원시값 CSV, 오버레이 PNG, 요약 텍스트

dose 값은 파일명에서 자동 파싱하고 테이블에서 수동 수정할 수 있다. 기본 정규식은
`(?i)(\d+(?:\.\d+)?)\s*uc` 이며(예: `pattern_320uC_01.tif` -> 320) 설정에서 바꿀 수
있다. 매칭에 실패하면 dose를 비워 두고 사용자 입력을 기다린다.

## 8. 테스트 전략

`tests/synth.py`가 합성 SEM 이미지를 생성한다. 파라미터는 갭 폭(정답), 각도, 에지
확산(erf 프로파일의 sigma), 노이즈 레벨, edge-brightening 강도다.

**회귀 테스트 기준**: 갭 20/30/50/80/100 nm x 각도 0/2/5/10도 x SNR 3수준 조합에서
측정값이 정답의 **+-1 px 이내**여야 한다.

추가 검증:

- 갭 0(short) 이미지 -> `short` 상태 반환
- 저대비 이미지 -> `short` 또는 `no_edge` 상태 반환 (`valid` 반환은 실패로 간주)
- ROI에 패턴 2개 포함 -> `multi_edge` 검출
- `metadata.py` -> 실제 FEI 헤더 텍스트 픽스처로 파싱 검증
- `stats.py` -> MAD 이상치 제거, 유효 라인 부족 시 None 반환
- `classify.py` -> 각 status 판정 경계값과 `low_confidence` 플래그 단위 테스트
- `width_px`가 3, 10 경계에 정확히 놓인 경우의 판정 (경계 포함 규칙 고정)

GUI는 자동 테스트하지 않는다. 계산이 전부 엔진에 있으므로 GUI 버그는 표시 문제로
한정되고, 수동 확인으로 충분하다.

## 9. 의존성

`numpy`, `scipy`, `tifffile`, `PySide6`, `pyqtgraph`, `pytest`

## 10. 알려진 리스크

**실제 Inspect F 촬영 파일이 없다.** 메타데이터 파서는 공개된 FEI TIFF 포맷 사양에
맞춰 구현하고 합성 픽스처로만 검증한다. 실제 이미지 한 장 또는 헤더 텍스트를 확보하면
파서를 실제 파일 기준으로 맞춘다. 그 전까지는 폴백 경로(스케일바 자동 검출, 수동
2점 캘리브레이션)가 확실히 동작하도록 만들어 두어, 메타데이터 인식이 실패해도 툴을
쓸 수 있게 한다.

## 11. 범위에서 제외

- 갭 이외의 CD(선폭, 오버레이 정렬 오차) 측정
- 여러 장에 ROI를 자동 전파하는 기능 (장마다 스테이지 위치가 달라 신뢰할 수 없음)
- OCR 기반 데이터바 텍스트 자동 판독 (사용자 입력으로 대체)
- 3차원/단면 측정, 틸트 보정
