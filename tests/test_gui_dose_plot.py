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
