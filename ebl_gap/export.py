"""측정 결과를 CSV, 오버레이 PNG, 요약 텍스트로 내보낸다.

CSV는 utf-8-sig로 쓴다. 엑셀에서 한글이 깨지지 않게 하기 위해서다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from ebl_gap.dataset import Session
from ebl_gap.profile import aligned_to_image, uv_extent
from ebl_gap.types import UNCERTAIN_STATUSES, ImageRecord, Roi, RoiResult

#: 전 구간 short인 dose 옆에 반드시 따라붙는 문장.
#: 엔진은 픽셀만으로 닫힌 갭과 평탄한 금속을 구별할 수 없다. 둘 다 대비가 없고,
#: 실측하면 빗나간 ROI도 no_edge가 아니라 전 구간 short를 낸다. 리포트의 이 줄이
#: 사용자가 dose를 고르는 자리이므로, 구별할 수 없다는 사실을 여기에 적는다.
CLOSED_DOSE_AMBIGUITY = ("갭이 닫혔거나 ROI가 패턴을 벗어났습니다 — "
                         "오버레이로 확인하세요")

SUMMARY_COLUMNS = [
    "file", "dose_uC", "nm_per_px", "scale_source", "roi_index", "angle_deg",
    "mean_nm", "std_nm", "n_valid", "n_short", "n_uncertain",
    "n_low_confidence", "warnings",
]

LINE_COLUMNS = [
    "row", "left_px", "right_px", "width_px", "width_nm", "status", "flags",
    "reason",
]

STATUS_COLORS = {
    "valid": (0, 255, 0),
    "short": (255, 0, 0),
}
UNCERTAIN_COLOR = (255, 140, 0)
ROI_COLOR = (255, 255, 0)


def record_notices(record: ImageRecord) -> list[str]:
    """이미지 한 장에 붙은 사유를 채널이 드러나는 접두사와 함께 낸다.

    `오류:`는 "이 이미지로는 측정할 수 없다", `참고:`는 "측정은 되지만 확인하라".
    한 칸에 섞어 쓰던 때는 아래쪽이 어두운 멀쩡한 FEI 이미지가 실험 노트와
    요약 CSV에 `오류:`로 남았다. 리포트, CSV, 상태 표시줄이 같은 문구를 쓰도록
    한 자리에 둔다 — 갈라지면 같은 이미지가 화면과 파일에서 다르게 읽힌다.

    `error`가 있어도 스케일이 잡혀 있으면 오류로 내지 않는다. `error`가 붙는
    경로는 읽기 실패와 스케일 미확정 둘뿐이고, 둘 다 스케일이 생기는 순간
    거짓이 된다. 캘리브레이션은 오류 메시지가 하라고 시킨 바로 그 행동이므로,
    그러고도 `오류:`가 남으면 실험 노트가 자기가 낸 숫자를 의심하게 된다.
    판단은 여기서 하고 `record.error`는 지우지 않는다 — 지우면 나중에 이
    이미지의 스케일이 왜 수동인지 알 방법이 없다.
    """
    notices: list[str] = []
    if record.error and record.scale is None:
        notices.append(f"오류: {record.error}")
    notices.extend(f"참고: {note}" for note in record.notes)
    return notices


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_summary_csv(path: str | Path, session: Session) -> None:
    """이미지별, ROI별 요약을 한 행씩 쓴다."""
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for record in session.records:
            if not record.roi_results:
                writer.writerow({
                    "file": record.path.name,
                    "dose_uC": _fmt(record.dose, 2),
                    "nm_per_px": _fmt(record.scale.nm_per_px if record.scale else None),
                    "scale_source": record.scale.source if record.scale else "",
                    "roi_index": "",
                    "angle_deg": "", "mean_nm": "", "std_nm": "",
                    "n_valid": "", "n_short": "", "n_uncertain": "",
                    "n_low_confidence": "",
                    "warnings": " | ".join(record_notices(record)),
                })
                continue
            for index, result in enumerate(record.roi_results):
                writer.writerow({
                    "file": record.path.name,
                    "dose_uC": _fmt(record.dose, 2),
                    "nm_per_px": _fmt(result.scale.nm_per_px),
                    "scale_source": result.scale.source,
                    "roi_index": index,
                    "angle_deg": _fmt(result.angle_deg, 3),
                    "mean_nm": _fmt(result.mean_nm, 3),
                    "std_nm": _fmt(result.std_nm, 3),
                    "n_valid": result.n_valid,
                    "n_short": result.n_short,
                    "n_uncertain": result.n_uncertain,
                    "n_low_confidence": result.n_low_confidence,
                    # 이미지 한 장에 붙은 안내도 함께 싣는다. ROI별 경고만 싣던
                    # 때는 HFW 불일치 — 보고되는 모든 nm를 조용히 편향시키는
                    # 유일한 조건 — 이 요약 CSV에서 통째로 사라졌고, 픽셀 크기가
                    # 세 배 틀렸을 수 있는 이미지가 출처 `fei_metadata`에 경고
                    # 칸이 빈 가장 믿음직한 행으로 남았다.
                    "warnings": " | ".join([*record_notices(record),
                                            *result.warnings]),
                })


def write_lines_csv(path: str | Path, record: ImageRecord,
                    roi_index: int) -> None:
    """한 ROI의 스캔라인별 원시 측정값을 쓴다."""
    result = record.roi_results[roi_index]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LINE_COLUMNS)
        writer.writeheader()
        for line in result.lines:
            writer.writerow({
                "row": line.row,
                "left_px": _fmt(line.left_px, 3),
                "right_px": _fmt(line.right_px, 3),
                "width_px": _fmt(line.width_px, 3),
                "width_nm": _fmt(line.width_nm, 3),
                "status": line.status,
                "flags": " ".join(sorted(line.flags)),
                "reason": line.reason,
            })


def _to_rgb(image) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    lo, hi = float(img.min()), float(img.max())
    span = hi - lo if hi > lo else 1.0
    gray = np.clip((img - lo) / span * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _put(canvas: np.ndarray, x: float, y: float, color) -> None:
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < canvas.shape[0] and 0 <= xi < canvas.shape[1]:
        canvas[yi, xi] = color


def render_overlay(image, roi: Roi, result: RoiResult) -> np.ndarray:
    """원본 이미지 위에 ROI와 검출된 에지를 그린 RGB 배열을 만든다."""
    canvas = _to_rgb(image)

    # 테두리 좌표를 이미지 안으로 자른다. extract_profiles는 경계를 살짝 벗어난
    # ROI를 mode="nearest"로 허용하므로 같은 ROI로 measure_roi가 성공한다.
    # 여기서 자르지 않으면 측정은 되는데 오버레이만 IndexError로 죽어서,
    # CSV에는 값이 남고 그림만 안 나오는 상태가 된다.
    y0 = max(0, min(roi.y0, canvas.shape[0] - 1))
    y1 = max(0, min(roi.y1, canvas.shape[0] - 1))
    x0 = max(0, min(roi.x0, canvas.shape[1] - 1))
    x1 = max(0, min(roi.x1, canvas.shape[1] - 1))
    canvas[y0, x0 : x1 + 1] = ROI_COLOR
    canvas[y1, x0 : x1 + 1] = ROI_COLOR
    canvas[y0 : y1 + 1, x0] = ROI_COLOR
    canvas[y0 : y1 + 1, x1] = ROI_COLOR

    for line in result.lines:
        if line.status in STATUS_COLORS:
            color = STATUS_COLORS[line.status]
        elif line.status in UNCERTAIN_STATUSES:
            color = UNCERTAIN_COLOR
        else:
            continue

        if line.left_px is None or line.right_px is None:
            # 에지가 없는 라인은 정렬 좌표계의 중앙에 한 점만 찍는다. 중앙은
            # ROI의 가로가 아니라 측정 방향 표본 수에서 온다 — 측정 방향이
            # 세로면 둘이 맞바뀌고, 가로를 그대로 쓰면 점이 ROI 밖에 찍힌다.
            u_extent, _ = uv_extent(roi, result.angle_deg)
            x, y = aligned_to_image(roi, result.angle_deg,
                                    (u_extent - 1) / 2.0, line.row)
            _put(canvas, x, y, color)
            continue

        for u in (line.left_px, line.right_px):
            x, y = aligned_to_image(roi, result.angle_deg, u, line.row)
            _put(canvas, x, y, color)
    return canvas


def write_overlay_png(path: str | Path, image, roi: Roi,
                      result: RoiResult) -> None:
    Image.fromarray(render_overlay(image, roi, result), mode="RGB").save(str(path))


def format_report(session: Session) -> str:
    """사람이 읽는 요약 텍스트."""
    lines: list[str] = ["EBL dose test 갭 측정 요약", "=" * 40, ""]

    # 세션 전체의 진단은 머리에 둔다. dose 블록의 줄마다 붙는 확인 요청과 달리
    # 이것은 "이 세션 전체가 이상하다"는 말이라 가장 먼저 읽혀야 한다.
    session_warnings = [*session.session_warnings(), *session.scale_warnings()]
    for warning in session_warnings:
        lines.append(f"[세션 경고] {warning}")
    if session_warnings:
        lines.append("")

    for record in session.records:
        dose = "미상" if record.dose is None else f"{record.dose:g} uC"
        lines.append(f"- {record.path.name} (dose {dose})")
        for notice in record_notices(record):
            lines.append(f"    {notice}")
        if not record.roi_results:
            lines.append("    측정 결과 없음")
            lines.append("")
            continue
        for index, result in enumerate(record.roi_results):
            if result.mean_nm is None:
                head = "갭 측정 불가"
            else:
                spread = "" if result.std_nm is None else f" +- {result.std_nm:.2f}"
                head = f"갭 {result.mean_nm:.2f}{spread} nm"
            lines.append(
                f"    ROI {index}: {head} "
                f"(유효 {result.n_valid} / short {result.n_short} / "
                f"판정보류 {result.n_uncertain} 라인, 각도 {result.angle_deg:.2f}도, "
                f"{result.scale.nm_per_px:.4f} nm/px [{result.scale.source}])"
            )
            for warning in result.warnings:
                lines.append(f"        ! {warning}")
        lines.append("")

    curve = session.dose_curve()
    closed = session.closed_doses()
    if curve or closed:
        lines.append("dose - 갭 관계")
        lines.append("-" * 40)
        # 측정된 점과 갭이 닫힌 dose를 dose 순서로 한 줄씩 섞어 낸다. 닫힌 dose를
        # 빼면 "어느 dose에서 갭이 닫히는가"라는 dose test의 답이 리포트에서
        # 사라진다. 갭 폭 자리에 0을 쓰지 않는다 — 재지 않은 값이기 때문이다.
        rows: list[tuple[float, str]] = []
        for point in curve:
            spread = "" if point.std_nm is None else f" +- {point.std_nm:.2f}"
            rows.append((point.dose,
                         f"  {point.dose:>8.1f} uC : {point.mean_nm:7.2f}{spread} nm "
                         f"(유효 {point.n_valid}, short {point.n_short})"))
        for dose_point in closed:
            # 두 줄로 낸다. 엔진은 닫힌 갭과 패턴을 벗어난 ROI를 구별할 수 없다 —
            # 평탄한 금속도 전 구간 short를 내기 때문이다. 이 블록이 사용자가
            # dose를 고르는 자리이므로 없는 확신을 적으면 안 된다.
            head = f"  {dose_point.dose:>8.1f} uC : "
            rows.append((dose_point.dose,
                         f"{head}전 구간 short "
                         f"({dose_point.n_short}/{dose_point.n_total} 라인, 유효 0)"
                         f"\n{'':>{len(head)}}{CLOSED_DOSE_AMBIGUITY}"))
        lines.extend(text for _, text in sorted(rows, key=lambda r: r[0]))
    return "\n".join(lines)
