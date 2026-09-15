"""FEI TIFF에 내장된 INI 메타데이터를 읽어 스케일을 확정한다.

FEI/Thermo 장비는 TIFF 태그 34680(SFEG) 또는 34682(Helios)에 INI 형식 텍스트를
넣는다. [Scan] PixelWidth가 미터 단위 픽셀 크기이므로, 데이터바의 스케일바를 픽셀로
재서 추정하는 것보다 정확하다.
"""

from __future__ import annotations

from pathlib import Path

import tifffile

from ebl_gap.types import ScaleInfo

FEI_TAG_CODES = (34680, 34682, 34683)

#: PixelWidth * 이미지 가로폭과 HFW가 이 비율 이상 어긋나면 경고한다.
HFW_MISMATCH_TOLERANCE = 0.05


class MetadataNotFoundError(RuntimeError):
    """필요한 메타데이터가 없을 때."""


def parse_ini(text: str) -> dict[str, dict[str, str]]:
    """FEI의 INI 블록을 섹션 딕셔너리로 파싱한다.

    configparser를 쓰지 않는 이유는 FEI 블록에 중복 키와 빈 섹션이 섞여 있어
    strict 모드에서 예외가 나기 때문이다. 알아볼 수 없는 줄은 조용히 버린다.
    """
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.strip().lstrip("\x00")
        if not line or line.startswith((";", "#")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
            continue
        if current is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        current[key.strip()] = value.strip()
    return sections


def read_fei_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    """TIFF에서 FEI 메타데이터 섹션을 읽는다."""
    with tifffile.TiffFile(str(path)) as tif:
        meta = getattr(tif, "fei_metadata", None)
        if meta:
            return {str(k): {str(kk): str(vv) for kk, vv in v.items()}
                    for k, v in meta.items() if isinstance(v, dict)}
        for page in tif.pages:
            for code in FEI_TAG_CODES:
                tag = page.tags.get(code)
                if tag is None:
                    continue
                value = tag.value
                if isinstance(value, bytes):
                    value = value.decode("latin-1", errors="replace")
                if isinstance(value, str) and "[" in value:
                    parsed = parse_ini(value)
                    if parsed:
                        return parsed
    raise MetadataNotFoundError(
        f"FEI 메타데이터를 찾지 못했다: {path}. "
        "스케일바 자동 검출 또는 수동 캘리브레이션을 쓴다."
    )


def _first_float(meta: dict[str, dict[str, str]], candidates) -> float | None:
    for section, key in candidates:
        raw = meta.get(section, {}).get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def scale_from_metadata(meta: dict[str, dict[str, str]],
                        image_width: int) -> tuple[ScaleInfo, list[str]]:
    """PixelWidth에서 ScaleInfo를 만들고 HFW와 교차 검증한다."""
    pixel_width_m = _first_float(meta, [("Scan", "PixelWidth"),
                                        ("EScan", "PixelWidth")])
    if pixel_width_m is None or pixel_width_m <= 0:
        raise MetadataNotFoundError(
            "메타데이터에 PixelWidth가 없거나 값이 올바르지 않다"
        )

    nm_per_px = pixel_width_m * 1e9
    warnings: list[str] = []

    hfw_m = _first_float(meta, [("Scan", "HorFieldsize"),
                                ("EScan", "HorFieldsize"),
                                ("EBeam", "HorFieldsize")])
    if hfw_m and hfw_m > 0:
        expected_nm = hfw_m * 1e9
        actual_nm = nm_per_px * image_width
        if abs(actual_nm - expected_nm) / expected_nm > HFW_MISMATCH_TOLERANCE:
            warnings.append(
                f"PixelWidth x 가로폭({actual_nm:.1f} nm)이 HFW"
                f"({expected_nm:.1f} nm)와 어긋난다. 이미지가 크롭 또는 "
                "리사이즈되었을 수 있으니 수동 캘리브레이션을 권장한다"
            )

    return ScaleInfo(nm_per_px=nm_per_px, source="fei_metadata"), warnings


def databar_top_row(meta: dict[str, dict[str, str]],
                    image_height: int) -> int | None:
    """데이터바가 시작되는 행 번호. 데이터바가 없으면 None.

    FEI는 [Image] ResolutionY에 데이터바를 뺀 실제 스캔 높이를 적는다. TIFF 전체
    높이가 그보다 크면 차이만큼이 데이터바다.
    """
    raw = meta.get("Image", {}).get("ResolutionY")
    if raw is None:
        return None
    try:
        scan_height = int(float(raw))
    except ValueError:
        return None
    if 0 < scan_height < image_height:
        return scan_height
    return None
