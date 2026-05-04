"""
geeXHP Desktop — entry point.

Run from the project root:
    python -m desktop_app.main
or:
    python desktop_app/main.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Expose the bundled share/ directory so the desktop environment can find the
# app icon (taskbar, app menu) even when launched directly instead of via
# run_geexhp.sh (which sets XDG_DATA_DIRS itself).
_share = ROOT / "desktop_app" / "share"
if _share.exists():
    _existing = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")
    if str(_share) not in _existing.split(":"):
        os.environ["XDG_DATA_DIRS"] = f"{_share}:{_existing}"

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class LoadingWindow(QWidget):
    def __init__(self, icon_path: Path):
        super().__init__(None)
        self.setObjectName("LoadingWindow")
        self.setWindowTitle("geeXHP Desktop")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setFixedSize(620, 320)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#0b1424"))
        self.setPalette(palette)
        self.setStyleSheet(
            """
            QWidget#LoadingWindow {
                background-color: #0b1424;
            }
            QFrame#LoadingCard {
                background-color: #0b1424;
                border: 1px solid #243254;
                border-radius: 0px;
            }
            QLabel#LoadingKicker {
                color: #e8b657;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 1.4px;
            }
            QLabel#LoadingTitle {
                color: white;
                font-size: 30px;
                font-weight: 800;
            }
            QLabel#LoadingBody {
                color: #c8d5e8;
                font-size: 14px;
            }
            QLabel#LoadingStatus {
                color: #d7e1ef;
                font-size: 12px;
                font-weight: 700;
            }
            QProgressBar {
                background-color: #243254;
                border: none;
                border-radius: 4px;
                height: 8px;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background-color: #3fbfb0;
                border-radius: 4px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setObjectName("LoadingCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(34, 30, 34, 28)
        card_lay.setSpacing(22)

        top = QHBoxLayout()
        top.setSpacing(22)
        logo_lbl = QLabel()
        logo_lbl.setFixedSize(104, 104)
        logo_lbl.setAlignment(Qt.AlignCenter)
        if icon_path.exists():
            pm = QPixmap(str(icon_path))
            if not pm.isNull():
                logo_lbl.setPixmap(
                    pm.scaled(104, 104, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        top.addWidget(logo_lbl)

        copy = QVBoxLayout()
        copy.setSpacing(6)
        kicker = QLabel("ATMOSPHERIC RETRIEVAL")
        kicker.setObjectName("LoadingKicker")
        title = QLabel("geeXHP Desktop")
        title.setObjectName("LoadingTitle")
        body = QLabel("Loading spectra, neural-network models, and interface...")
        body.setObjectName("LoadingBody")
        body.setWordWrap(True)
        copy.addStretch(1)
        copy.addWidget(kicker)
        copy.addWidget(title)
        copy.addWidget(body)
        copy.addStretch(1)
        top.addLayout(copy, 1)
        card_lay.addLayout(top, 1)

        bottom = QVBoxLayout()
        bottom.setSpacing(10)
        self.status = QLabel("Preparing application...")
        self.status.setObjectName("LoadingStatus")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        bottom.addWidget(self.status)
        bottom.addWidget(self.progress)
        card_lay.addLayout(bottom)

        layout.addWidget(card)

    def set_message(self, message: str) -> None:
        self.status.setText(message)
        self.repaint()
        QApplication.processEvents()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("geeXHP")
    app.setApplicationDisplayName("geeXHP Desktop")
    app.setOrganizationName("geeXHP")
    app.setDesktopFileName("geexhp")
    app.setStyle("Fusion")

    icon_path = ROOT / "desktop_app" / "assets" / "geexhp.png"
    icon = QIcon()
    if icon_path.exists():
        icon = QIcon(str(icon_path))
        app.setWindowIcon(icon)

    splash = LoadingWindow(icon_path)
    splash.show()
    splash.raise_()
    splash.activateWindow()
    splash.repaint()
    app.loading_window = splash

    def set_loading_message(message: str) -> None:
        splash.set_message(message)

    def load_application() -> None:
        try:
            splash.set_message("Loading visual theme...")
            from desktop_app.app.theme import stylesheet

            app.setStyleSheet(stylesheet())

            splash.set_message("Loading scientific plotting tools...")
            QApplication.processEvents()

            from desktop_app.app.main_window import MainWindow

            splash.set_message("Preparing retrieval interface...")
            QApplication.processEvents()
            win = MainWindow(project_root=ROOT, defer_loading=True)
        except Exception as exc:
            splash.close()
            QMessageBox.critical(
                None,
                "geeXHP could not start",
                f"The application failed while loading:\n\n{exc!r}",
            )
            app.quit()
            return

        if not icon.isNull():
            win.setWindowIcon(icon)
        app.main_window = win

        def show_main_window() -> None:
            splash.set_message("Opening geeXHP...")
            win.show()
            splash.close()

        splash.set_message("Loading test-set samples...")
        win.startupReady.connect(show_main_window)
        win.start_loading()

    QTimer.singleShot(500, lambda: set_loading_message("Checking Python environment..."))
    QTimer.singleShot(1700, lambda: set_loading_message("Preparing scientific libraries..."))
    QTimer.singleShot(3000, lambda: set_loading_message("Loading spectra and interface assets..."))
    QTimer.singleShot(4200, lambda: set_loading_message("Almost ready..."))
    QTimer.singleShot(5000, load_application)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
