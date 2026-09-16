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


def test_unreadable_file_produces_a_record_with_an_error(tmp_path):
    path = tmp_path / "broken.tif"
    path.write_bytes(b"not a tiff at all")
    loaded = load_image(path)
    assert loaded.record.error is not None
    assert loaded.pixels.size == 0
