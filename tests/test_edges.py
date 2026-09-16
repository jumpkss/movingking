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
