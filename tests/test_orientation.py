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
    estimated, n_rows, _ = estimate_angle_deg(img, roi)
    assert estimated == pytest.approx(angle_deg, abs=0.2)
    assert n_rows == 300


def test_survives_noise_that_breaks_some_rows():
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0,
                          angle_deg=7.0, noise_sigma=8.0, seed=5)
    roi = Roi(106, 106, 405, 405)
    estimated, _, _ = estimate_angle_deg(img, roi)
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
    estimated, n_rows, _ = estimate_angle_deg(img, roi)
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
    _, n_rows, _ = estimate_angle_deg(img, roi)
    assert 240 <= n_rows <= 260


# --- 방향 판별 (Task 28) ---------------------------------------------------
#
# 이 프로젝트의 합성 이미지는 전부 세로 갭이었다. 사용자의 실제 이미지는 갭이
# 가로로 누워 있었고, 세로 갭을 전제한 추정이 31.5도를 내놓아 참값 70 nm가
# 137.39 nm로 부풀었다. 가로 갭은 `np.rot90`으로 만든다 — 참값이 보존된다.


@pytest.mark.parametrize("gap_nm", [20.0, 50.0, 100.0])
@pytest.mark.parametrize("noise_sigma", [0.0, 3.0, 8.0])
def test_detects_a_vertical_gap_as_the_zero_degree_base(gap_nm, noise_sigma):
    from ebl_gap.orientation import detect_base_angle_deg

    img = synth_gap_image(width=512, height=512, gap_nm=gap_nm, nm_per_px=1.0,
                          noise_sigma=noise_sigma, edge_bright=15.0,
                          seed=int(gap_nm + noise_sigma))
    roi = Roi(106, 106, 405, 405)
    assert detect_base_angle_deg(img, roi) == 0.0


@pytest.mark.parametrize("gap_nm", [20.0, 50.0, 100.0])
@pytest.mark.parametrize("noise_sigma", [0.0, 3.0, 8.0])
def test_detects_a_horizontal_gap_as_the_ninety_degree_base(gap_nm, noise_sigma):
    from ebl_gap.orientation import detect_base_angle_deg

    img = np.rot90(synth_gap_image(
        width=512, height=512, gap_nm=gap_nm, nm_per_px=1.0,
        noise_sigma=noise_sigma, edge_bright=15.0,
        seed=int(gap_nm + noise_sigma)))
    roi = Roi(106, 106, 405, 405)
    assert detect_base_angle_deg(img, roi) == 90.0


def test_a_tie_picks_the_vertical_base():
    """어느 쪽도 맞지 않는 경우다 — 임의 선택이지만 정해져 있어야 한다.

    정사각 ROI에 45도 대각 갭을 놓고 전치 대칭으로 다듬은 뒤 정수로 반올림하면
    두 방향의 대비가 비트 단위로 같아진다. 반올림이 필요한 이유는 대칭 행렬이어도
    `mean(axis=0)`과 `mean(axis=1)`의 덧셈 순서가 달라 마지막 비트가 갈리기
    때문이다 — 정수 값이면 덧셈이 정확해져 순서가 결과를 바꾸지 못한다. 실제 SEM
    이미지도 8비트 정수다. 동률에서 방향이 흔들리면 같은 이미지를 두 번 재면서
    측정 방향이 바뀔 수 있다.
    """
    from ebl_gap.orientation import detect_base_angle_deg

    diagonal = synth_gap_image(width=256, height=256, gap_nm=40.0,
                               nm_per_px=1.0, angle_deg=45.0)
    img = np.round((diagonal + diagonal.T) / 2.0)  # 정확한 동률을 만든다
    roi = Roi(28, 28, 227, 227)
    assert detect_base_angle_deg(img, roi) == 0.0


@pytest.mark.parametrize("tilt_deg", [0.0, 3.0, -4.0])
def test_estimates_the_tilt_inside_the_horizontal_base(tilt_deg):
    """가로 갭도 기준(90도) 위에서 기울기를 얹어 돌려줘야 한다.

    세로 갭을 `np.rot90`으로 눕히면 각도 a인 갭이 90 + a가 된다. 기준을 못 고르면
    이 이미지에서 31.5도 같은 값이 나오고, 58도 틀린 각도는 1/cos만큼 폭을 부풀린다
    (사용자 제보: 참값 70 nm가 137.39 nm).
    """
    img = np.rot90(synth_gap_image(width=512, height=512, gap_nm=40.0,
                                   nm_per_px=1.0, angle_deg=tilt_deg))
    roi = Roi(106, 106, 405, 405)
    angle_deg, n_rows, base_deg = estimate_angle_deg(img, roi)
    assert base_deg == 90.0
    assert angle_deg == pytest.approx(90.0 + tilt_deg, abs=0.2)
    assert n_rows == 300


def test_the_vertical_base_is_reported_alongside_the_angle():
    img = synth_gap_image(width=512, height=512, gap_nm=40.0, nm_per_px=1.0,
                          angle_deg=5.0)
    roi = Roi(106, 106, 405, 405)
    angle_deg, _, base_deg = estimate_angle_deg(img, roi)
    assert base_deg == 0.0
    assert angle_deg == pytest.approx(5.0, abs=0.2)


def test_a_given_base_is_used_instead_of_detecting_one():
    """호출자가 기준을 정했으면 판별을 다시 하지 않는다.

    `measure_roi`는 타당성 경고에 쓸 기준을 이미 알고 있고, 같은 측정에서 기준이
    두 번 정해지면 경고가 가리키는 기준과 측정에 쓴 기준이 갈라질 수 있다.

    같은 이미지에서 기준을 0도로 박으면 수정 전과 같은 자리로 돌아간다 — 가로
    갭을 세로 기준으로 훑으니 행마다 에지가 없다(재현 실험: "가로 갭 70nm(자동)
    -> 측정 실패").
    """
    img = np.rot90(synth_gap_image(width=512, height=512, gap_nm=40.0,
                                   nm_per_px=1.0))
    roi = Roi(106, 106, 405, 405)
    _, _, base_deg = estimate_angle_deg(img, roi)
    assert base_deg == 90.0
    with pytest.raises(InsufficientEdgesError):
        estimate_angle_deg(img, roi, base_deg=0.0)
