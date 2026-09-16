"""GUI 테스트를 화면 없이 돌리기 위한 설정.

QT_QPA_PLATFORM은 PySide6가 import되기 전에 설정되어야 한다. conftest.py는 테스트
수집 전에 실행되므로 여기가 맞는 자리다.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """세션 전체가 공유하는 QApplication. Qt는 프로세스당 하나만 허용한다."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
