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
