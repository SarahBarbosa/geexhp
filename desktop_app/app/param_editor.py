from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from desktop_app.app.constants import BANDS, TELESCOPE_CONFIG


class ParameterEditor(QGroupBox):
    spectrumRequested = Signal(str)
    exampleRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(
            "CUSTOM SPECTRUM  ·  PASTE FLUX, THEN RETRIEVE DIRECTLY", parent
        )

        self._telescope = "LUVOIR"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.context = QLabel()
        self.context.setObjectName("SectionTag")
        top.addWidget(self.context, 1)

        example_btn = QPushButton("Load selected test spectrum as example")
        example_btn.setProperty("accent", True)
        example_btn.setMinimumHeight(32)
        example_btn.clicked.connect(self.exampleRequested.emit)
        top.addWidget(example_btn)
        outer.addLayout(top)

        self.editor = QPlainTextEdit()
        self.editor.setMinimumHeight(220)
        self.editor.setPlaceholderText(
            "Accepted formats:\n"
            "0.2019, 0.1824, 0.1632, ...\n\n"
            "or one row per bin:\n"
            "wavelength_um, noisy_albedo, noise_1sigma\n"
            "0.2019, 0.1824, 0.0120"
        )
        outer.addWidget(self.editor, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        hint = QLabel(
            "The spectrum is converted to the model input tensors without "
            "searching the test set."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        bottom.addWidget(hint, 1)

        use_btn = QPushButton("Run retrieval + corner")
        use_btn.setProperty("primary", True)
        use_btn.setMinimumHeight(36)
        use_btn.setMinimumWidth(180)
        use_btn.clicked.connect(lambda: self.spectrumRequested.emit(self.text()))
        bottom.addWidget(use_btn)
        outer.addLayout(bottom)

        self.set_telescope("LUVOIR")

    def set_telescope(self, telescope: str) -> None:
        self._telescope = telescope
        cfg = TELESCOPE_CONFIG[telescope]
        bands = " + ".join(f"{cfg['bins'][b]} {b}" for b in BANDS)
        total = sum(cfg["bins"][b] for b in BANDS)
        self.context.setText(
            f"{cfg['label'].upper()} INPUT  ·  {total} BINS  ·  {bands}"
        )

    def text(self) -> str:
        return self.editor.toPlainText()

    def set_text(self, text: str) -> None:
        self.editor.setPlainText(text)

    def set_hide_o2o3(self, hide: bool) -> None:
        return None
