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
