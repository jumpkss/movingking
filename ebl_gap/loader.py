"""이미지 한 장을 열어 픽셀, 스케일, dose, 데이터바 위치를 한 번에 돌려준다.

파일을 못 읽거나 메타데이터가 없어도 예외를 던지지 않고 ImageRecord.error에 사유를
적는다. 폴더를 통째로 여는 도중 한 장 때문에 전체가 멈추면 안 되기 때문이다.
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
from ebl_gap.types import ImageRecord

TIFF_SUFFIXES = {".tif", ".tiff"}


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
    try:
        meta = read_fei_metadata(path)
    except MetadataNotFoundError as exc:
        record.error = str(exc)
        return LoadedImage(record=record, pixels=pixels, databar_top=None)
    except Exception as exc:
        record.error = f"메타데이터를 읽지 못했다: {exc}"
        return LoadedImage(record=record, pixels=pixels, databar_top=None)

    databar_top = databar_top_row(meta, pixels.shape[0])
    try:
        scale, warnings = scale_from_metadata(meta, pixels.shape[1])
    except MetadataNotFoundError as exc:
        record.error = str(exc)
        return LoadedImage(record=record, pixels=pixels, databar_top=databar_top)

    record.scale = scale
    if warnings:
        record.error = " | ".join(warnings)
    return LoadedImage(record=record, pixels=pixels, databar_top=databar_top)
