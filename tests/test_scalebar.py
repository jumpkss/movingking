import numpy as np
import pytest

from ebl_gap.scalebar import (
    detect_databar_top,
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


def bright_run_in_the_databar(length_px):
    """어두운 데이터바 위에 주어진 길이의 밝은 수평 런 하나만 둔다.

    `test_ignores_a_run_spanning_almost_the_whole_width`는 데이터바 전체가 균일하게
    밝아서 `region.max() - region.min() == 0`인 조기 반환에 걸린다. 그래서 폭 가드를
    통째로 지워도 통과한다. 여기서는 데이터바 안에 어두운 배경을 남겨 span > 0으로
    만들어, 검사가 실제로 폭 가드까지 도달하게 한다.
    """
    img = np.full((943, 1024), 120.0)
    img[884:, :] = 10.0
    img[918:923, 0:length_px] = 250.0
    return img


def test_a_bright_run_wider_than_the_ratio_is_rejected():
    """가로폭의 90%를 넘는 밝은 런은 스케일바가 아니다.

    밝게 반전된 데이터바 행이 막대로 둔갑하는 것을 막는 유일한 가드다. 통과시키면
    950px를 막대로 잡고, 사용자가 "1 um"을 입력하는 순간 nm/px가 실제의 1/9로
    확정되어 이후 모든 갭 측정이 조용히 9배 틀어진다.
    """
    # 950 / 1024 = 92.8% > 90%. 값 자체를 단언하지 않는다 — 그런 단언은 상수를
    # 다시 쓴 것일 뿐 아무 동작도 지키지 않는다. 이 검사와 아래 검사가 함께
    # 비율을 (87.9%, 92.8%) 안으로 가둔다.
    assert detect_scalebar(bright_run_in_the_databar(950), databar_top=884) is None


def test_a_wide_bar_below_the_ratio_is_still_detected():
    """가드가 진짜 막대까지 잘라내면 안 된다. 경계 바로 아래는 통과해야 한다.

    위 검사만 있으면 비율을 얼마든지 낮춰도 통과한다. 900 / 1024 = 87.9%로
    경계 바로 아래에 두어 비율을 양쪽에서 고정한다.
    """
    hit = detect_scalebar(bright_run_in_the_databar(900), databar_top=884)
    assert hit is not None
    assert hit.length_px == 900


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


def test_detect_databar_top_finds_a_dark_band_without_metadata():
    """FEI 계열의 어두운 데이터바. 메타데이터가 없는 PNG 크롭이 이 경로다."""
    assert detect_databar_top(image_with_databar()) == 884


def test_detect_databar_top_finds_a_bright_band_too():
    """데이터바가 밝은 장비도 있다. 부호를 가리면 절반을 놓친다."""
    img = np.full((943, 1024), 10.0)
    img[884:, :] = 250.0
    assert detect_databar_top(img) == 884


def test_detect_databar_top_is_none_for_an_image_without_one():
    rng = np.random.default_rng(0)
    img = np.full((512, 512), 120.0) + rng.normal(0.0, 4.0, (512, 512))
    img[:, 250:262] = 40.0  # 세로 갭은 아래쪽 띠가 아니다
    assert detect_databar_top(img) is None


def test_detect_databar_top_ignores_a_band_taller_than_the_scan_area():
    """아래쪽 절반이 통째로 어두우면 데이터바가 아니라 시료의 일부다."""
    img = np.full((943, 1024), 120.0)
    img[500:, :] = 10.0
    assert detect_databar_top(img) is None


def test_detect_databar_top_ignores_a_one_row_streak():
    img = np.full((943, 1024), 120.0)
    img[942:, :] = 10.0
    assert detect_databar_top(img) is None
