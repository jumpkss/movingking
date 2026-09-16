"""이미지 한 장을 열어 픽셀, 스케일, dose, 데이터바 위치를 한 번에 돌려준다.

파일을 못 읽거나 메타데이터가 없어도 예외를 던지지 않고 ImageRecord에 사유를
적는다. 폴더를 통째로 여는 도중 한 장 때문에 전체가 멈추면 안 되기 때문이다.

사유는 두 채널로 나뉜다. 측정을 못 하게 만드는 것(읽기 실패, 스케일 미확정)만
`record.error`로 가고, 재는 데 지장이 없는 안내는 `record.notes`로 간다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ebl_gap.dataset import DEFAULT_DOSE_PATTERN, parse_dose
from ebl_gap.metadata import (
    MetadataNotFoundError,
    databar_top_row,
    read_fei_metadata,
    scale_from_metadata,
)
from ebl_gap.scalebar import detect_databar_top
from ebl_gap.types import ImageRecord

TIFF_SUFFIXES = {".tif", ".tiff"}

#: 스케일을 확정하지 못한 모든 경로가 붙이는 안내.
#: 하부 예외의 영문(tifffile의 "not a TIFF file: header=b'\x89PNG'" 등)만 보이면
#: PNG 크롭 폴더를 여는 사람은 "내 파일이 깨졌다"로 읽고 멈춘다. 파일은 멀쩡하고
#: 올바른 다음 행동은 스케일 캘리브레이션이다. 원문 예외는 진단에 필요하므로
#: 버리지 않고 이 문장 뒤에 괄호로 남긴다.
NO_SCALE_HINT = ("스케일 메타데이터가 없습니다 — 스케일 캘리브레이션으로 "
                 "직접 지정하세요")


def _read_pixels(path: Path) -> np.ndarray:
    if path.suffix.lower() in TIFF_SUFFIXES:
        import tifffile

        data = tifffile.imread(str(path))
    else:
        from PIL import Image

        with Image.open(path) as handle:
            data = np.asarray(handle)

    array = np.asarray(data)
    if array.ndim == 3:
        # RGB/RGBA는 ITU-R 601 가중치로 회색조로 바꾼다.
        rgb = array[..., :3].astype(np.float64)
        array = rgb @ np.array([0.299, 0.587, 0.114])
    if array.ndim != 2:
        raise ValueError(f"2차원 이미지가 아니다 (차원 {array.ndim})")
    return np.ascontiguousarray(array, dtype=np.float64)


@dataclass(frozen=True)
class LoadedImage:
    """파일 한 장에서 읽어낸 모든 것."""

    record: ImageRecord
    pixels: np.ndarray
    databar_top: int | None


def _finish(record: ImageRecord, pixels, databar_top: int | None,
            errors: list[str], notes: list[str]) -> LoadedImage:
    """사유를 채널별로 기록에 옮기고 결과를 만든다.

    데이터바 위치를 메타데이터로 확정하지 못했으면 픽셀에서 찾아보고, 찾으면
    안내를 남긴다. `measure_roi`는 `databar_top`이 있어야 ROI 침범을 거부할 수
    있으므로, 없을 때 할 수 있는 것은 말해 주는 것뿐이다. 휴리스틱 결과를
    `databar_top`에 넣어 거부 근거로 삼지는 않는다 — 아래쪽이 어두운 멀쩡한
    시료를 영영 못 재게 된다.

    같은 이유로 이 안내는 `notes`로 간다. 측정을 막지 않는 휴리스틱의 결과이므로
    `error`에 넣으면 아무 문제도 없는 FEI 이미지가 리포트에 `오류:`로 남는다.
    """
    if databar_top is None:
        try:
            found = detect_databar_top(pixels)
        except Exception:  # 휴리스틱 하나 때문에 로딩이 멈추면 안 된다
            found = None
        if found is not None:
            notes.append(
                f"아래쪽 {found}행부터 데이터바로 보이는 띠가 있습니다 — "
                f"ROI가 이 영역에 걸치지 않게 하세요"
            )
    if errors:
        record.error = " | ".join(errors)
    record.notes = list(notes)
    return LoadedImage(record=record, pixels=pixels, databar_top=databar_top)


def load_image(path: str | Path, *,
               dose_pattern: str = DEFAULT_DOSE_PATTERN) -> LoadedImage:
    """이미지를 읽고 가능한 만큼 메타데이터를 채운다."""
    path = Path(path)
    record = ImageRecord(path=path, dose=parse_dose(path.name, dose_pattern))

    try:
        pixels = _read_pixels(path)
    except Exception as exc:  # 손상된 파일, 지원하지 않는 포맷 등
        record.error = f"이미지를 읽지 못했다: {exc}"
        return LoadedImage(record=record, pixels=np.empty((0, 0)),
                           databar_top=None)

    databar_top: int | None = None
    # 스케일을 못 정하면 잴 수 없다(GUI가 그 조건으로 측정을 거부한다) — errors.
    # 그 밖의 것은 재는 데 지장이 없는 안내 — notes.
    errors: list[str] = []
    notes: list[str] = []

    try:
        meta = read_fei_metadata(path)
    except MetadataNotFoundError as exc:
        errors.append(str(exc))
        return _finish(record, pixels, None, errors, notes)
    except Exception as exc:
        errors.append(f"{NO_SCALE_HINT} (메타데이터를 읽지 못했다: {exc})")
        return _finish(record, pixels, None, errors, notes)

    # 아래 두 호출도 각각 감싼다. 둘 다 "TIFF는 읽히고 FEI 태그도 파싱되는데
    # 필드 값만 이상한" 경우에 터지고, 그 예외는 MetadataNotFoundError가 아니다:
    #   ResolutionY=inf  -> int(float("inf"))가 OverflowError
    #   PixelWidth=nan   -> float()을 통과하고 <= 0 검사도 통과(NaN 비교는 항상
    #                       거짓)한 뒤 ScaleInfo가 ValueError
    # 한 장 때문에 폴더 전체 스캔이 멈추면 안 된다는 것이 이 함수의 존재 이유다.
    try:
        databar_top = databar_top_row(meta, pixels.shape[0])
    except Exception as exc:
        databar_top = None
        # 위치를 모르면 ROI 침범을 거부하지 못할 뿐, 측정 자체는 된다.
        notes.append(f"데이터바 위치를 읽지 못했다: {exc}")

    try:
        scale, warnings = scale_from_metadata(meta, pixels.shape[1])
    except MetadataNotFoundError as exc:
        errors.append(f"{NO_SCALE_HINT} ({exc})")
        return _finish(record, pixels, databar_top, errors, notes)
    except Exception as exc:
        errors.append(f"{NO_SCALE_HINT} (스케일을 계산하지 못했다: {exc})")
        return _finish(record, pixels, databar_top, errors, notes)

    record.scale = scale
    # HFW 불일치는 스케일이 나온 뒤의 확인 요청이다. 잴 수는 있다.
    notes.extend(warnings)
    return _finish(record, pixels, databar_top, errors, notes)
