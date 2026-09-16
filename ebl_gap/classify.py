"""스캔라인 한 줄의 상태를 판정한다.

판정은 status와 flags로 나뉜다. 이 둘을 섞지 않는 것이 중요하다. status는 라인이
통계에 들어가는지를 혼자서 결정하고, flags는 통계 포함 여부를 바꾸지 않는 부가
정보다.
"""

from __future__ import annotations

from ebl_gap.edges import ProfileAnalysis

_EPS = 1e-9


def classify_line(
    analysis: ProfileAnalysis,
    *,
    contrast_k: float = 5.0,
    min_width_px: float = 3.0,
    low_confidence_px: float = 10.0,
) -> tuple[str, frozenset[str], str]:
    """(status, flags, reason)을 돌려준다.

    판정 순서는 스펙 5.1절을 따른다. 대비 검사를 가장 먼저 하는 이유는, 갭이 닫힌
    라인에서도 잡음 때문에 에지가 "찾아지는" 일이 있기 때문이다. 그 값을 정상으로
    보고하면 short를 놓친다.
    """
    contrast = analysis.contrast
    noise_floor = contrast_k * max(analysis.sigma_noise, _EPS)
    if contrast <= noise_floor:
        return (
            "short",
            frozenset(),
            f"대비 {contrast:.1f} <= 노이즈 {analysis.sigma_noise:.1f} x {contrast_k:g}",
        )

    if analysis.left_px is None or analysis.right_px is None:
        missing = "왼쪽" if analysis.left_px is None else "오른쪽"
        return "no_edge", frozenset(), f"{missing} 문턱 상향 교차를 찾지 못함"

    if analysis.n_cross_left > 1 or analysis.n_cross_right > 1:
        return (
            "multi_edge",
            frozenset(),
            f"교차 횟수 좌 {analysis.n_cross_left} 우 {analysis.n_cross_right} "
            "(ROI에 다른 패턴이 포함된 것으로 보임)",
        )

    width_px = analysis.width_px
    assert width_px is not None  # 위에서 None을 걸렀다
    if width_px < min_width_px:
        return (
            "sub_resolution",
            frozenset(),
            f"폭 {width_px:.2f}px < {min_width_px:g}px (해상도 한계 이하)",
        )

    flags = frozenset({"low_confidence"}) if width_px < low_confidence_px else frozenset()
    return "valid", flags, ""
