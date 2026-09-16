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


def test_infinite_resolution_does_not_raise(tmp_path):
    """TIFF도 읽히고 FEI 태그도 파싱되는데 필드 값만 이상한 경우.

    int(float("inf"))는 ValueError가 아니라 OverflowError를 던지므로
    databar_top_row 안의 좁은 except를 빠져나간다. 한 장 때문에 폴더 전체 스캔이
    멈추면 안 된다.
    """
    ini = FEI_INI.replace("ResolutionY=884", "ResolutionY=inf")
    path = tmp_path / "inf_300uC.tif"
    tifffile.imwrite(path, np.zeros((943, 1024), dtype=np.uint8),
                     extratags=[(34682, 's', 0, ini, True)])
    loaded = load_image(path)
    assert loaded.databar_top is None
    assert "데이터바" in loaded.record.error
    # 스케일은 정상이므로 측정은 계속할 수 있어야 한다.
    assert loaded.record.scale is not None


def test_nan_pixel_width_does_not_raise(tmp_path):
    """PixelWidth=nan은 float()을 통과하고 <= 0 검사도 통과한다.

    NaN 비교는 항상 거짓이므로 가드를 지나쳐 ScaleInfo가 ValueError를 던지는데,
    그것은 MetadataNotFoundError가 아니라 좁은 except를 빠져나간다.
    """
    ini = FEI_INI.replace("PixelWidth=3.0517578125e-009", "PixelWidth=nan")
    path = tmp_path / "nan_300uC.tif"
    tifffile.imwrite(path, np.zeros((943, 1024), dtype=np.uint8),
                     extratags=[(34682, 's', 0, ini, True)])
    loaded = load_image(path)
    assert loaded.record.scale is None
    assert "스케일" in loaded.record.error


def test_unreadable_file_produces_a_record_with_an_error(tmp_path):
    path = tmp_path / "broken.tif"
    path.write_bytes(b"not a tiff at all")
    loaded = load_image(path)
    assert loaded.record.error is not None
    assert loaded.pixels.size == 0


def test_a_png_gets_korean_guidance_instead_of_a_tiff_library_message(tmp_path):
    """PNG 크롭 폴더를 여는 사람에게 다음 행동을 한국어로 말해 준다.

    지금은 `메타데이터를 읽지 못했다: not a TIFF file: header=b'\\x89PNG'`가
    파일 목록과 상태 표시줄에 그대로 뜬다. 이것을 읽는 학생은 "내 파일이
    깨졌다"로 해석하고 멈추는데, 파일은 멀쩡하고 올바른 다음 행동은 스케일
    캘리브레이션이다. 원문 예외는 진단에 필요하니 버리지 않고 뒤에 남긴다.
    """
    path = tmp_path / "crop_300uC.png"
    Image.fromarray(np.full((32, 48), 200, dtype=np.uint8)).save(path)

    error = load_image(path).record.error

    assert error.startswith("스케일 메타데이터가 없습니다")
    assert "스케일 캘리브레이션" in error
    assert "TIFF" in error, f"진단용 원문이 사라졌다: {error}"


def test_an_unusable_pixel_width_also_gets_the_calibration_guidance(tmp_path):
    """스케일을 못 만드는 다른 경로에서도 같은 안내가 나와야 한다.

    PNG 경로만 고치면 FEI 태그는 있는데 값이 망가진 파일에서 다시 영문 예외만
    남는다. 사용자 입장에서는 둘 다 "스케일을 모른다"라는 같은 상황이다.
    """
    ini = FEI_INI.replace("PixelWidth=3.0517578125e-009", "PixelWidth=nan")
    path = tmp_path / "nan_300uC.tif"
    tifffile.imwrite(path, np.zeros((943, 1024), dtype=np.uint8),
                     extratags=[(34682, 's', 0, ini, True)])

    error = load_image(path).record.error

    assert "스케일 메타데이터가 없습니다" in error
    assert "스케일 캘리브레이션" in error
