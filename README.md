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
   > 목록의 dose 칸을 직접 고치면 된다. 규칙 자체를 바꾸려면 파이썬에서
   > `load_image(path, dose_pattern=r"d(\d+)")`처럼 직접 넘긴다 — GUI에는 이
   > 설정이 없다.
2. 스케일을 자동으로 못 읽었으면 **스케일 캘리브레이션**으로 확정한다. 확정 전에는
   측정이 거부된다.
3. 이미지 위에서 갭을 가로지르도록 **ROI를 드래그**한다. ROI 양 끝이 전극 평탄부에
   충분히 걸치도록 넉넉하게 잡는 것이 좋다 — 문턱을 평탄부 밝기로 잡기 때문이다.
4. **측정**을 누른다. 갭 각도를 자동으로 추정해 정렬한 뒤 ROI의 모든 행에서 폭을
   잰다. 검출된 에지가 오버레이로 그려진다.
5. 값이 이상하면 오른쪽 아래 **라인 스핀박스**로 행을 골라 프로파일을 확인한다.
   **이상 ▶** 버튼은 valid가 아닌 다음 라인으로 건너뛴다 — `short`나 `no_edge`가
   왜 그렇게 판정됐는지 문턱 위치로 직접 확인할 수 있다.
6. 이미지마다 3~4를 반복하면 **dose-gap 곡선** 탭에 관계가 그려진다.
7. **요약 CSV / 라인 CSV / 오버레이 PNG / 요약 리포트**로 내보낸다.

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
