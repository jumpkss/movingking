import numpy as np
import pytest
from scipy.special import erf

from ebl_gap.classify import classify_line
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


def test_zero_contrast_analysis_still_carries_the_fraction():
    """대비 0 조기 반환도 실제로 쓴 fraction을 실어 나른다.

    구성 지점이 둘이라 한쪽만 고치면 이 경로에서만 문턱선이 틀린 높이에 선다.
    """
    analysis = analyze_profile(np.full(201, 180.0), threshold_fraction=0.3)
    assert analysis.threshold_fraction == pytest.approx(0.3)
    assert analysis.left_px is None and analysis.right_px is None


def test_hysteresis_suppresses_noise_induced_extra_crossings():
    """문턱 근처에서 잡음이 여러 번 넘나들어도 교차는 한 번으로 센다."""
    p = erf_profile(n=301, gap_px=20.0, sigma=2.0, noise=6.0, seed=3)
    a = analyze_profile(p)
    assert a.n_cross_left == 1
    assert a.n_cross_right == 1


def haloed_profile(n=200, gap_px=20.0, sigma=2.0, halo=80.0):
    """에지 바깥에 넓은 edge-brightening 어깨가 붙은 프로파일.

    전극 밝기 200, 갭 바닥 40, 참 폭 20px. 어깨(+80)는 에지에서 2 sigma 바깥부터
    프로파일 양 끝 21픽셀 앞까지 덮는다. 그래서 끝에서 20% 창(k=40)은 깨끗한
    금속 21개와 어깨 19개를 담아 중앙값이 200으로 남고, 21%(k=42)부터는 어깨가
    과반이 되어 중앙값이 어깨 밝기로 끌려간다. 경계를 20% 바로 위에 올려 둔
    구성이다.
    """
    x = np.arange(n, dtype=float)
    d = np.abs(x - (n - 1) / 2.0)
    t = 0.5 * (erf((d - gap_px / 2.0) / (np.sqrt(2.0) * sigma)) + 1.0)
    p = 40.0 + 160.0 * t
    shoulder = (d >= gap_px / 2.0 + 2.0 * sigma) & (d <= 78.5)
    return p + halo * shoulder


def test_flat_window_stays_off_the_edge_brightening_shoulder():
    """평탄부 창은 전극 밝기만 담아야 한다 — 어깨가 섞이면 문턱이 통째로 틀어진다.

    창이 20%를 넘으면 중앙값이 어깨 밝기(280)로 끌려가 문턱이 120에서 160으로
    올라가고, 참 폭 20px가 22.7px로 나온다. 13% 계통 오차이고 방향이 일정해서
    반복 측정으로도 드러나지 않는다. edge-brightening은 SEM에서 늘 있는 현상이라
    이 오차는 실측에서 상시 켜진다.
    """
    a = analyze_profile(haloed_profile())
    assert a.i_hi_left == pytest.approx(200.0), (
        f"평탄부 창이 밝은 어깨를 물었다 (i_hi_left {a.i_hi_left:.1f}, 전극은 200)")
    assert a.width_px == pytest.approx(20.0, abs=0.2)


def test_flat_window_is_wide_enough_to_outvote_bright_specks():
    """창이 너무 좁으면 프로파일 끝의 밝은 점 몇 개가 전극 밝기를 대신한다.

    위 검사만 있으면 창을 얼마든지 좁혀도 통과한다. ROI 가장자리의 밝은 티끌은
    SEM 실측에서 흔하고, 11개가 창을 장악하면 i_hi가 400으로 잡혀 문턱이 전극
    밝기보다 높아진다. 그러면 에지를 티끌에서 찾아 폭 20px가 177px이 된다.
    10% 창(k=20)은 티끌에 넘어가고 20% 창(k=40)은 버틴다.
    """
    p = erf_profile(n=200, gap_px=20.0, sigma=2.0)
    p[:11] = 400.0
    p[-11:] = 400.0

    a = analyze_profile(p)

    assert a.i_hi_left == pytest.approx(200.0), (
        f"밝은 티끌 11개가 평탄부 중앙값을 장악했다 (i_hi_left {a.i_hi_left:.1f})")
    assert a.width_px == pytest.approx(20.0, abs=0.2)


def test_hysteresis_keeps_a_noisy_line_from_being_called_multi_edge():
    """잡음이 문턱을 다시 넘는 것을 다중 패턴으로 오판하면 안 된다.

    기존 잡음 검사(noise=6)는 히스테리시스를 0으로 둬도 통과한다 — 잡음이 문턱
    근처까지 오지 않기 때문이다. 대비 160에 sigma 20인 저선량 스캔에서야
    가짜 교차가 실제로 세어진다. 그때 `classify_line`이 이 멀쩡한 라인을
    multi_edge로 돌려보내고, 그 라인은 통계에서 빠진 채 "ROI에 다른 패턴이
    포함된 것으로 보임"이라는 틀린 이유를 달고 사용자에게 보고된다.
    """
    p = erf_profile(n=301, gap_px=20.0, sigma=2.0, noise=20.0, seed=6)
    a = analyze_profile(p)

    assert (a.n_cross_left, a.n_cross_right) == (1, 1), (
        f"가짜 교차가 세어졌다 (좌 {a.n_cross_left}, 우 {a.n_cross_right})")
    status, _, reason = classify_line(a)
    assert status == "valid", f"멀쩡한 라인이 {status}로 판정됐다: {reason}"


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


def test_analysis_exposes_the_thresholds_it_actually_used():
    """문턱은 분석 결과에 실려 나온다. 소비자가 다시 계산하면 갈라진다."""
    profile = np.array([200.0] * 20 + [40.0] * 10 + [160.0] * 20)

    a50 = analyze_profile(profile, threshold_fraction=0.5)
    assert a50.threshold_left == pytest.approx(120.0)
    assert a50.threshold_right == pytest.approx(100.0)

    a30 = analyze_profile(profile, threshold_fraction=0.3)
    assert a30.threshold_left == pytest.approx(88.0)
    assert a30.threshold_right == pytest.approx(76.0)
    # 갭은 어둡고 전극은 밝다. 에지는 갭 바닥에서 바깥으로 나가며 밝기가 문턱을
    # 처음 넘는 곳이므로, 문턱이 낮아지면 더 일찍 넘어 갭이 좁게 잡힌다.
    # 이 관계가 깨지면 threshold_fraction이 안 먹은 것이다.
    assert a30.width_px < a50.width_px
