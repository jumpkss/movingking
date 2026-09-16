from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from ebl_gap.dataset import Session  # noqa: E402
from ebl_gap.types import ImageRecord, RoiResult, ScaleInfo  # noqa: E402
from ebl_gap_gui.dose_plot import DosePlot  # noqa: E402

SCALE = ScaleInfo(nm_per_px=3.0, source="fei_metadata")


def result(mean_nm, std_nm=1.0, n_valid=200, n_short=0, scale=SCALE):
    return RoiResult(mean_nm=mean_nm, std_nm=std_nm, n_valid=n_valid,
                     n_short=n_short, n_uncertain=0, n_low_confidence=0,
                     angle_deg=0.0, lines=(), warnings=(), scale=scale)


def session_with(*pairs, scale=SCALE):
    session = Session()
    for dose, mean_nm in pairs:
        session.add(ImageRecord(path=Path(f"{dose:g}uC.tif"), scale=scale,
                                dose=dose, roi_results=[result(mean_nm)]))
    return session


def test_plots_one_point_per_measured_dose(qapp):
    plot = DosePlot()
    plot.set_session(session_with((300.0, 80.0), (400.0, 50.0), (500.0, 20.0)))
    assert plot.point_count() == 3


def test_ignores_images_without_a_dose(qapp):
    session = session_with((300.0, 80.0))
    session.add(ImageRecord(path=Path("nodose.tif"), scale=SCALE, dose=None,
                            roi_results=[result(40.0)]))
    plot = DosePlot()
    plot.set_session(session)
    assert plot.point_count() == 1


def test_replacing_the_session_clears_the_previous_curve(qapp):
    plot = DosePlot()
    plot.set_session(session_with((300.0, 80.0), (400.0, 50.0)))
    plot.set_session(Session())
    assert plot.point_count() == 0


def test_shows_a_warning_when_magnification_is_mixed(qapp):
    session = session_with((300.0, 80.0))
    coarse = ScaleInfo(nm_per_px=9.0, source="fei_metadata")
    session.add(ImageRecord(path=Path("b.tif"), scale=coarse, dose=400.0,
                            roi_results=[result(50.0, scale=coarse)]))
    plot = DosePlot()
    plot.set_session(session)
    assert "배율" in plot.warning_text()


def test_no_warning_for_a_consistent_session(qapp):
    plot = DosePlot()
    plot.set_session(session_with((300.0, 80.0), (400.0, 50.0)))
    assert plot.warning_text() == ""


def test_short_ratio_uses_valid_plus_short_as_the_denominator(qapp):
    """분모가 n_valid + n_short인지 경계에서 확인한다.

    52/(1000+52) = 0.0494 -> 임계 5% 미만이라 표시 안 됨.
    분모를 n_valid로 잘못 쓰면 52/1000 = 0.052로 임계를 넘어 표시된다.
    값이 극단적인 케이스만 테스트하면 분모를 틀려도 통과하는데, 이 분모가
    dose 점에 "short 발생" 표시를 붙일지 결정하고 사용자는 그걸 보고 dose를 고른다.
    """
    session = Session()
    session.add(ImageRecord(path=Path("a.tif"), scale=SCALE, dose=300.0,
                            roi_results=[result(80.0, n_valid=1000, n_short=52)]))
    plot = DosePlot()
    plot.set_session(session)
    assert plot.point_count() == 1
    assert plot.shorted_point_count() == 0


def test_shorted_doses_are_marked_separately(qapp):
    session = Session()
    session.add(ImageRecord(path=Path("a.tif"), scale=SCALE, dose=300.0,
                            roi_results=[result(80.0)]))
    session.add(ImageRecord(path=Path("b.tif"), scale=SCALE, dose=500.0,
                            roi_results=[result(15.0, n_valid=50, n_short=250)]))
    plot = DosePlot()
    plot.set_session(session)
    assert plot.point_count() == 2
    assert plot.shorted_point_count() == 1


def test_exactly_five_percent_short_is_marked(qapp):
    """정확히 5%면 곡선도 표시한다 — 결과 패널과 같은 경계를 쓴다.

    `stats.py`는 `>= 0.05`에서 "short 발생 구간 있음"을 띄우는데 여기가 `> 0.05`면,
    딱 5%인 dose에서 패널은 경고하고 곡선에는 아무 표시가 없다. 사용자는 dose를
    곡선에서 고르므로, 두 화면이 다른 답을 주는 그 한 점이 정확히 판단이 갈리는
    자리다. 50/(950+50) = 0.05로 경계 위에 정확히 올려 둔다.
    """
    session = Session()
    session.add(ImageRecord(path=Path("a.tif"), scale=SCALE, dose=300.0,
                            roi_results=[result(80.0, n_valid=950, n_short=50)]))
    plot = DosePlot()
    plot.set_session(session)
    assert plot.point_count() == 1
    assert plot.shorted_point_count() == 1


def closed(dose, n_short=300):
    """갭이 전 구간에서 닫힌 이미지 레코드."""
    return ImageRecord(path=Path(f"{dose:g}uC.tif"), scale=SCALE, dose=dose,
                       roi_results=[result(None, std_nm=None, n_valid=0,
                                           n_short=n_short)])


def test_a_closed_dose_is_drawn_at_zero_gap(qapp):
    """갭이 닫힌 dose는 갭 0 자리에 빨간 X로 찍힌다.

    이것이 이 도구의 결론이 되는 화면이다. "이 dose에서 갭이 닫힌다"가 dose
    test의 답이므로, 그 점이 빠진 곡선은 답의 절반을 지운 것이다. 표와 CSV에는
    남아 있지만 사용자가 dose를 고르는 곳은 곡선이다.

    플래그가 아니라 그려진 항목의 좌표를 읽는다 — 카운터만 올리고 그리지 않아도
    통과하는 검사는 아무것도 지키지 못한다.
    """
    session = session_with((300.0, 80.0), (400.0, 50.0))
    session.add(closed(500.0))

    plot = DosePlot()
    plot.set_session(session)

    assert plot.point_count() == 2          # 측정된 점은 그대로 둘
    assert plot.closed_dose_marks() == [(500.0, 0.0)]


def test_a_series_where_every_dose_closes_still_draws_the_marks(qapp):
    """측정된 점이 하나도 없어도 닫힌 dose는 그려야 한다.

    측정 점이 없으면 일찍 반환하던 자리다. 전 구간이 닫힌 시리즈에서 곡선이
    통째로 비면, 사용자는 "아직 아무것도 안 쟀다"로 읽는다.
    """
    session = Session()
    session.add(closed(400.0))
    session.add(closed(500.0))

    plot = DosePlot()
    plot.set_session(session)

    assert plot.point_count() == 0
    assert plot.closed_dose_marks() == [(400.0, 0.0), (500.0, 0.0)]


def test_replacing_the_session_clears_the_closed_dose_marks(qapp):
    """세션을 갈아 끼우면 이전 시리즈의 닫힘 표시가 남으면 안 된다."""
    first = Session()
    first.add(closed(500.0))
    plot = DosePlot()
    plot.set_session(first)
    assert plot.closed_dose_marks() == [(500.0, 0.0)]

    plot.set_session(session_with((300.0, 80.0)))

    assert plot.closed_dose_marks() == []
