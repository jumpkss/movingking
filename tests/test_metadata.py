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
    # tifffile may normalize scientific notation, so compare numerical values
    assert float(meta["Scan"]["PixelWidth"]) == pytest.approx(3.0517578125e-009)


def test_read_fei_metadata_raises_on_a_plain_tiff(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(MetadataNotFoundError, match="FEI"):
        read_fei_metadata(path)
