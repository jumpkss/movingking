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
    # 스케일은 나왔으니 잴 수 있다 — 확인하라는 안내지 오류가 아니다.
    assert loaded.record.error is None
    assert any("HFW" in note for note in loaded.record.notes)


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
    assert any("데이터바" in note for note in loaded.record.notes)
    # 스케일은 정상이므로 측정은 계속할 수 있어야 한다.
    assert loaded.record.scale is not None
    assert loaded.record.error is None


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


def test_a_bottom_band_is_reported_when_metadata_cannot_locate_the_databar(
        tmp_path):
    """메타데이터가 없으면 거부할 수단이 없다. 최소한 말은 해 줘야 한다.

    데이터바를 걸친 ROI는 날조된 short 비율을 만든다. PNG 크롭과 비 FEI TIFF가
    바로 이 경로로 몰린다.
    """
    data = np.full((300, 400), 120, dtype=np.uint8)
    data[260:, :] = 12
    path = tmp_path / "crop_300uC.png"
    Image.fromarray(data).save(path)

    loaded = load_image(path)

    assert loaded.databar_top is None  # 메타데이터가 없으니 확정은 못 한다
    note = " | ".join(loaded.record.notes)
    assert "데이터바" in note
    assert "260" in note
    # 스케일이 없는 것은 안내가 아니라 측정을 막는 사유다 — 채널이 다르다.
    assert "스케일" in loaded.record.error


def test_no_band_note_for_an_image_without_a_databar(tmp_path):
    path = tmp_path / "plain_300uC.png"
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(path)
    loaded = load_image(path)
    assert "데이터바" not in loaded.record.error
    assert not any("데이터바" in note for note in loaded.record.notes)


def _fei_tiff_without_a_databar_field(path, data):
    """ResolutionY가 이미지 높이와 같은 FEI TIFF — 데이터바가 없는 촬영이다."""
    ini = (FEI_INI.replace("ResolutionY=884", f"ResolutionY={data.shape[0]}")
                  .replace("ResolutionX=1024", f"ResolutionX={data.shape[1]}"))
    tifffile.imwrite(path, data, extratags=[(34682, 's', 0, ini, True)])
    return path


def test_a_dark_bottom_on_a_measurable_image_is_a_note_not_an_error(tmp_path):
    """멀쩡한 FEI 이미지가 아래쪽이 어둡다는 이유로 오류 취급을 받으면 안 된다.

    `detect_databar_top`은 휴리스틱이고 측정을 막지도 않는다. 그런데 안내가
    `record.error`를 타면 리포트에 `오류:`로 찍히고 요약 CSV의 경고 칸에
    들어간다 — 아무 문제도 없는 이미지의 실험실 기록이 그렇게 남는다.
    """
    data = np.full((300, 400), 120, dtype=np.uint8)
    data[250:, :] = 12  # 아래쪽이 어두운 시료
    path = _fei_tiff_without_a_databar_field(tmp_path / "dark_300uC.tif", data)

    loaded = load_image(path)

    assert loaded.record.scale is not None
    assert loaded.record.error is None, loaded.record.error
    assert any("데이터바" in note for note in loaded.record.notes)


def test_error_is_reserved_for_images_that_cannot_be_measured(tmp_path):
    """`error`가 붙은 것과 스케일이 없는 것은 같은 집합이어야 한다.

    GUI가 측정을 거부하는 조건이 `scale is None` 하나이므로, 그 밖의 사유로
    `error`를 붙이면 "측정할 수 없다"는 표시가 실제로는 잴 수 있는 이미지에
    붙는다.
    """
    good = _fei_tiff_without_a_databar_field(
        tmp_path / "good_300uC.tif", np.full((300, 400), 120, dtype=np.uint8))
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(
        tmp_path / "crop_300uC.png")

    for path in (good, tmp_path / "crop_300uC.png"):
        record = load_image(path).record
        assert (record.error is None) is (record.scale is not None), record
