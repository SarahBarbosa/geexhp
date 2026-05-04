from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QObject, QRect, QSize, QThread, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImageReader,
    QMovie,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop_app.app.constants import (
    ALL_PARAMS,
    DETECTION_SIGMA,
    ERA_ORDER,
    PARAM_LABELS,
    PARAM_PLAIN,
    PARAM_UNITS_TEXT,
    TELESCOPES,
    TELESCOPE_CONFIG,
)
from desktop_app.app.data import DataStore, SampleMeta, Spectrum
from desktop_app.app.inference import Retriever
from desktop_app.app.network_view import NetworkDiagramView
from desktop_app.app.param_editor import ParameterEditor
from desktop_app.app.plots import (
    CompareCanvas,
    CornerCanvas,
    IGCanvas,
    NetworkCanvas,
    RetrievalCanvas,
    SpectrumCanvas,
)
from desktop_app.app import theme


class _LoadWorker(QObject):
    progress = Signal(int)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, store: DataStore):
        super().__init__()
        self.store = store

    def run(self) -> None:
        try:
            metas = self.store.load_metadata(progress=lambda i: self.progress.emit(i))
            self.finished.emit(metas)
        except Exception as exc:
            self.failed.emit(repr(exc))


class MainWindow(QMainWindow):
    startupReady = Signal()

    def __init__(self, project_root: Path, defer_loading: bool = False):
        super().__init__()
        self.setWindowTitle("geeXHP")
        self.resize(1340, 880)

        self.store = DataStore(project_root)
        self.retriever = Retriever(self.store)
        self.metas: list[SampleMeta] = []
        self.current_index: Optional[int] = None
        self.current_custom_spec: Optional[Spectrum] = None
        self.last_results: dict[str, tuple[dict, dict]] = {}
        self.last_truth: Optional[dict] = None
        self.hide_o2o3 = False
        self.network_stages: list[dict] = []
        self.network_stage_index = 0
        self._movies: list[QMovie] = []
        self.network_timer = QTimer(self)
        self.network_timer.setInterval(950)
        self.network_timer.timeout.connect(self._advance_network_stage)

        self._build_ui()
        if not defer_loading:
            self.start_loading()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_target_tab(), "1.  Target")
        self.tabs.addTab(self._build_network_tab(), "2.  Network")
        self.tabs.addTab(self._build_retrieve_tab(), "3.  Retrieve")
        self.tabs.addTab(self._build_sensitivity_tab(), "4.  Sensitivity")
        self.tabs.addTab(self._build_compare_tab(), "5.  Compare")
        self.corner_tab = self._build_corner_tab()
        self.tabs.addTab(self.corner_tab, "6.  Corner")
        self.tabs.addTab(self._build_about_tab(), "7.  About")
        root.addWidget(self.tabs, 1)

        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setRange(0, 0)
        self.statusbar.addPermanentWidget(self.progress)

    def _build_header(self) -> QWidget:
        w = QFrame()
        w.setObjectName("Header")
        w.setFixedHeight(96)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(24, 14, 28, 14)
        lay.setSpacing(18)

        # Logo
        logo_path = self.store.project_root / "desktop_app" / "assets" / "geexhp.png"
        logo_lbl = QLabel()
        if logo_path.exists():
            pm = QPixmap(str(logo_path))
            if not pm.isNull():
                logo_lbl.setPixmap(
                    pm.scaled(
                        QSize(54, 54), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
        logo_lbl.setFixedSize(54, 54)
        lay.addWidget(logo_lbl)

        text_box = QVBoxLayout()
        text_box.setSpacing(0)
        tag = QLabel("ATMOSPHERIC RETRIEVAL")
        tag.setObjectName("HeaderTag")
        title = QLabel("geeXHP Desktop")
        title.setObjectName("HeaderTitle")
        sub = QLabel(
            "Reflected-light spectra, LUVOIR-B / HabEx retrievals, uncertainty, and model sensitivity"
        )
        sub.setObjectName("HeaderSubtitle")
        text_box.addWidget(tag)
        text_box.addWidget(title)
        text_box.addWidget(sub)
        lay.addLayout(text_box)
        lay.addStretch(1)

        chip_box = QVBoxLayout()
        chip_box.setSpacing(5)
        chip_box.addWidget(self._make_chip("HWO CONTEXT", theme.GOLD))
        chip_box.addWidget(self._make_chip("LUVOIR-B  ·  8 m", theme.TEL_LUVOIR))
        chip_box.addWidget(self._make_chip("HabEx/SS  ·  4 m", theme.TEL_HABEX))
        lay.addLayout(chip_box)

        return w

    @staticmethod
    def _make_chip(text: str, color: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"background-color: {color}22;"
            f"color: #f7fbff;"
            f"border: 1px solid {color}88;"
            f"border-radius: 10px;"
            f"padding: 3px 10px;"
            f"font-size: 11.5px;"
            f"font-weight: 700;"
            f"letter-spacing: 0.6px;"
        )
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    def _build_target_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(14)

        outer.addWidget(self._build_target_toolbar())

        self.target_stack = QStackedWidget()
        self.target_stack.addWidget(self._build_filter_panel())
        self.target_stack.addWidget(self._build_custom_panel())
        outer.addWidget(self.target_stack)

        bottom = QHBoxLayout()
        bottom.setSpacing(14)
        bottom.addWidget(self._build_target_info())
        self.spectrum_canvas = SpectrumCanvas()
        bottom.addWidget(self._wrap_canvas(self.spectrum_canvas), 1)
        outer.addLayout(bottom, 1)
        return page

    def _build_target_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("ToolbarCard")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(14)

        tag = QLabel("SOURCE")
        tag.setObjectName("SectionTag")
        lay.addWidget(tag)

        self.rb_filter = QRadioButton("Filter test set")
        self.rb_custom = QRadioButton("Paste spectrum")
        self.rb_filter.setChecked(True)
        self.rb_filter.toggled.connect(self._on_mode_change)
        grp = QButtonGroup(bar)
        grp.addButton(self.rb_filter)
        grp.addButton(self.rb_custom)
        lay.addWidget(self.rb_filter)
        lay.addWidget(self.rb_custom)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color:{theme.BORDER};")
        sep.setFixedHeight(20)
        lay.addWidget(sep)

        tlbl = QLabel("TELESCOPE")
        tlbl.setObjectName("SectionTag")
        lay.addWidget(tlbl)
        self.cmb_telescope = QComboBox()
        self.cmb_telescope.addItems([TELESCOPE_CONFIG[t]["label"] for t in TELESCOPES])
        self.cmb_telescope.setMinimumWidth(150)
        self.cmb_telescope.currentIndexChanged.connect(self._on_target_change)
        lay.addWidget(self.cmb_telescope)
        lay.addStretch(1)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet(f"color:{theme.BORDER};")
        sep2.setFixedHeight(20)
        lay.addWidget(sep2)

        self.chk_hide_o2o3 = QCheckBox("Hide O₂ / O₃   (Archean atmospheres)")
        self.chk_hide_o2o3.setToolTip(
            "Archean Earth has effectively zero O₂ and O₃, toggling this hides "
            "those panels everywhere so log-scale plots stay clean."
        )
        self.chk_hide_o2o3.toggled.connect(self._on_hide_toggle)
        lay.addWidget(self.chk_hide_o2o3)
        return bar

    def _build_filter_panel(self) -> QWidget:
        box = QGroupBox("FILTER THE TEST SET  ·  10 826 SIMULATED PLANETS")
        outer = QHBoxLayout(box)
        outer.setSpacing(28)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setContentsMargins(4, 6, 4, 4)

        self.cmb_star = QComboBox()
        self.cmb_star.addItems(["Any", "F", "G"])
        self.cmb_star.currentIndexChanged.connect(self._refresh_sample_list)

        self.cmb_era = QComboBox()
        self.cmb_era.addItems(["Any"] + [e.capitalize() for e in ERA_ORDER])
        self.cmb_era.currentIndexChanged.connect(self._refresh_sample_list)

        self.cmb_distance = QComboBox()
        self.cmb_distance.addItems(
            ["Any", "Near (≤ 8 pc)", "Mid (8–12 pc)", "Far (> 12 pc)"]
        )
        self.cmb_distance.currentIndexChanged.connect(self._refresh_sample_list)

        for cmb in (self.cmb_star, self.cmb_era, self.cmb_distance):
            cmb.setMinimumWidth(180)

        form.addRow("Star type", self.cmb_star)
        form.addRow("Geological era", self.cmb_era)
        form.addRow("Distance", self.cmb_distance)
        outer.addLayout(form, 1)

        right_form = QFormLayout()
        right_form.setHorizontalSpacing(14)
        right_form.setVerticalSpacing(10)
        right_form.setContentsMargins(4, 6, 4, 4)
        self.cmb_sample = QComboBox()
        self.cmb_sample.setEnabled(False)
        self.cmb_sample.currentIndexChanged.connect(self._on_target_change)
        self.cmb_sample.setMinimumWidth(360)
        right_form.addRow("Pick sample", self.cmb_sample)
        outer.addLayout(right_form, 2)
        return box

    def _build_custom_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.editor = ParameterEditor()
        self.editor.spectrumRequested.connect(self._on_custom_spectrum)
        self.editor.exampleRequested.connect(self._load_custom_example)
        scroll.setWidget(self.editor)
        return scroll

    def _build_target_info(self) -> QWidget:
        info_box = QGroupBox("SELECTED TARGET")
        info_box.setMaximumWidth(290)
        info_box.setMinimumWidth(260)
        ilay = QFormLayout(info_box)
        ilay.setHorizontalSpacing(14)
        ilay.setVerticalSpacing(9)
        ilay.setLabelAlignment(Qt.AlignRight)

        def _val_label() -> QLabel:
            l = QLabel("—")
            l.setObjectName("KeyValue")
            return l

        self.lbl_star = _val_label()
        self.lbl_dist = _val_label()
        self.lbl_era = _val_label()
        self.lbl_snr_l = _val_label()
        self.lbl_snr_h = _val_label()
        ilay.addRow("Star type", self.lbl_star)
        ilay.addRow("Distance", self.lbl_dist)
        ilay.addRow("Earth analogue", self.lbl_era)
        ilay.addRow("LUVOIR-B (Vis)", self.lbl_snr_l)
        ilay.addRow("HabEx/SS (Vis)", self.lbl_snr_h)
        return info_box

    @staticmethod
    def _wrap_canvas(canvas: QWidget) -> QWidget:
        frame = QFrame()
        frame.setObjectName("CanvasCard")
        l = QVBoxLayout(frame)
        l.setContentsMargins(10, 10, 10, 10)
        l.addWidget(canvas)
        return frame

    def _build_network_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(14)

        info_card = QFrame()
        info_card.setObjectName("InfoCard")
        ilay = QHBoxLayout(info_card)
        ilay.setContentsMargins(18, 14, 18, 14)
        info = QLabel(
            "<b style='color:#1a2334;'>How the neural network works</b> "
            "<span style='color:#5a6478;'>·  Select a spectrum on tab 1, then run "
            "a walkthrough. Each step is computed from the saved Keras model: "
            "normalization, convolutional feature maps, attention, latent vectors, "
            "and final output heads.</span>"
        )
        info.setWordWrap(True)
        ilay.addWidget(info, 1)
        outer.addWidget(info_card)

        body = QHBoxLayout()
        body.setSpacing(14)

        controls = QGroupBox("NETWORK WALKTHROUGH")
        controls.setMaximumWidth(330)
        controls.setMinimumWidth(310)
        clay = QVBoxLayout(controls)
        clay.setSpacing(11)

        self.lbl_network_context = QLabel("Use the selected target and telescope.")
        self.lbl_network_context.setObjectName("Hint")
        self.lbl_network_context.setWordWrap(True)
        clay.addWidget(self.lbl_network_context)

        self.btn_network_run = QPushButton("▶   Run walkthrough")
        self.btn_network_run.setProperty("primary", True)
        self.btn_network_run.setMinimumHeight(38)
        self.btn_network_run.clicked.connect(self._run_network_walkthrough)
        clay.addWidget(self.btn_network_run)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_network_prev = QPushButton("‹")
        self.btn_network_prev.setToolTip("Previous network step")
        self.btn_network_prev.clicked.connect(
            lambda: self._show_network_stage(self.network_stage_index - 1)
        )
        self.btn_network_play = QPushButton("Play")
        self.btn_network_play.clicked.connect(self._toggle_network_play)
        self.btn_network_next = QPushButton("›")
        self.btn_network_next.setToolTip("Next network step")
        self.btn_network_next.clicked.connect(
            lambda: self._show_network_stage(self.network_stage_index + 1)
        )
        for btn in (
            self.btn_network_prev,
            self.btn_network_play,
            self.btn_network_next,
        ):
            btn.setEnabled(False)
            btn.setMinimumHeight(34)
            row.addWidget(btn)
        clay.addLayout(row)

        self.network_slider = QSlider(Qt.Horizontal)
        self.network_slider.setRange(0, 0)
        self.network_slider.valueChanged.connect(self._on_network_slider)
        clay.addWidget(self.network_slider)

        self.lbl_network_stage = QLabel("No walkthrough loaded.")
        self.lbl_network_stage.setObjectName("MutedValue")
        self.lbl_network_stage.setWordWrap(True)
        clay.addWidget(self.lbl_network_stage)

        inspector = QFrame()
        inspector.setObjectName("NetworkInspector")
        il = QVBoxLayout(inspector)
        il.setContentsMargins(13, 12, 13, 12)
        il.setSpacing(8)
        self.lbl_network_inspector_title = QLabel("Layer inspector")
        self.lbl_network_inspector_title.setObjectName("NetworkInspectorTitle")
        self.lbl_network_inspector_body = QLabel(
            "Run the walkthrough, then hover or click the architecture."
        )
        self.lbl_network_inspector_body.setObjectName("NetworkInspectorBody")
        self.lbl_network_inspector_body.setWordWrap(True)
        self.lbl_network_inspector_body.setTextFormat(Qt.RichText)
        self.lbl_network_inspector_stats = QLabel("")
        self.lbl_network_inspector_stats.setObjectName("NetworkInspectorStats")
        self.lbl_network_inspector_stats.setWordWrap(True)
        self.lbl_network_inspector_stats.setTextFormat(Qt.RichText)
        il.addWidget(self.lbl_network_inspector_title)
        il.addWidget(self.lbl_network_inspector_body)
        il.addWidget(self.lbl_network_inspector_stats)
        clay.addWidget(inspector)

        note = QLabel(
            "The top panel is the model architecture. The bottom panel is the live "
            "tensor for the selected layer."
        )
        note.setObjectName("Hint")
        note.setWordWrap(True)
        clay.addWidget(note)
        clay.addStretch(1)

        self.network_diagram = NetworkDiagramView()
        self.network_diagram.stageSelected.connect(self._show_network_stage)
        self.network_canvas = NetworkCanvas()
        detail_split = QSplitter(Qt.Vertical)
        detail_split.setHandleWidth(8)

        diagram_card = QFrame()
        diagram_card.setObjectName("CanvasCard")
        dlay = QVBoxLayout(diagram_card)
        dlay.setContentsMargins(10, 10, 10, 10)
        dlay.addWidget(self.network_diagram)

        detail_split.addWidget(diagram_card)
        detail_split.addWidget(self._wrap_canvas(self.network_canvas))
        detail_split.setStretchFactor(0, 2)
        detail_split.setStretchFactor(1, 3)

        body.addWidget(controls)
        body.addWidget(detail_split, 1)
        outer.addLayout(body, 1)
        return page

    def _build_retrieve_tab(self) -> QWidget:
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(14)

        controls = QGroupBox("RUN RETRIEVAL")
        controls.setMaximumWidth(330)
        controls.setMinimumWidth(310)
        clay = QVBoxLayout(controls)
        clay.setSpacing(10)

        hint = QLabel("Pick a target on tab 1, then run the model below.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        clay.addWidget(hint)

        self.btn_run_luvoir = QPushButton("▶   Run  LUVOIR-B")
        self.btn_run_luvoir.setProperty("primary", True)
        self.btn_run_luvoir.setMinimumHeight(38)
        self.btn_run_luvoir.clicked.connect(lambda: self._run_retrieval("LUVOIR"))

        self.btn_run_habex = QPushButton("▶   Run  HabEx/SS")
        self.btn_run_habex.setProperty("accent", True)
        self.btn_run_habex.setMinimumHeight(38)
        self.btn_run_habex.clicked.connect(lambda: self._run_retrieval("HABEX"))

        clay.addWidget(self.btn_run_luvoir)
        clay.addWidget(self.btn_run_habex)
        clay.addSpacing(6)

        note = QLabel(
            "<span style='color:#5a6478;'>Uncertainties σ<sub>total</sub> combine "
            "MC-dropout (model) and bootstrap (data noise), pre-computed across "
            "the full test set.</span>"
        )
        note.setObjectName("Hint")
        note.setWordWrap(True)
        clay.addWidget(note)
        clay.addStretch(1)

        self.detection_box = QGroupBox("DETECTION SUMMARY  ·  3σ THRESHOLD")
        dlay = QVBoxLayout(self.detection_box)
        self.detection_label = QLabel(
            "<span style='color:#8b94a6;'>Run a retrieval to see "
            "detection significance.</span>"
        )
        self.detection_label.setWordWrap(True)
        self.detection_label.setTextFormat(Qt.RichText)
        dlay.addWidget(self.detection_label)

        side_lay = QVBoxLayout()
        side_lay.setSpacing(14)
        side_lay.addWidget(controls)
        side_lay.addWidget(self.detection_box)
        side_lay.addStretch(1)
        side_holder = QWidget()
        side_holder.setLayout(side_lay)
        side_holder.setMaximumWidth(330)

        right = QSplitter(Qt.Vertical)
        right.setHandleWidth(8)
        self.retrieval_canvas = RetrievalCanvas()
        right.addWidget(self._wrap_canvas(self.retrieval_canvas))

        table_card = QFrame()
        table_card.setObjectName("CanvasCard")
        tlay = QVBoxLayout(table_card)
        tlay.setContentsMargins(0, 0, 0, 0)

        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(
            ["Parameter", "Truth", "Predicted", "± σ_total"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setShowGrid(False)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setFrameShape(QFrame.NoFrame)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tlay.addWidget(self.results_table)
        right.addWidget(table_card)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        outer.addWidget(side_holder)
        outer.addWidget(right, 1)
        return page

    def _build_sensitivity_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(14)

        info_card = QFrame()
        info_card.setObjectName("InfoCard")
        ilay = QHBoxLayout(info_card)
        ilay.setContentsMargins(18, 14, 18, 14)
        ilay.setSpacing(20)

        text = QLabel(
            "<b style='color:#1a2334;'>Wavelength sensitivity</b> "
            "<span style='color:#5a6478;'>·  Integrated Gradients averaged across "
            "all test-set planets that share the geological era of the currently "
            "selected target. Warm cells = the model relies on that wavelength to "
            "predict the chemical on the row; cool = the opposite.</span>"
        )
        text.setWordWrap(True)
        ilay.addWidget(text, 1)

        self.lbl_ig_context = QLabel("—")
        self.lbl_ig_context.setStyleSheet(
            f"color:{theme.NAVY_700}; font-weight:700; "
            f"font-size:11px; letter-spacing:1.2px;"
        )
        self.lbl_ig_context.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ilay.addWidget(self.lbl_ig_context)
        outer.addWidget(info_card)

        self.ig_canvas = IGCanvas()
        outer.addWidget(self._wrap_canvas(self.ig_canvas), 1)
        return page

    def _build_compare_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(14)

        info_card = QFrame()
        info_card.setObjectName("InfoCard")
        ilay = QHBoxLayout(info_card)
        ilay.setContentsMargins(18, 14, 18, 14)
        info = QLabel(
            "<b style='color:#1a2334;'>Side-by-side retrieval</b> "
            "<span style='color:#5a6478;'>·  Run both telescopes on tab 3. "
            "Physical parameters and chemical abundances use separate axes so "
            "their very different numerical ranges stay readable. This view is "
            "for test-set examples with known truth.</span>"
        )
        info.setWordWrap(True)
        ilay.addWidget(info)
        outer.addWidget(info_card)

        self.compare_canvas = CompareCanvas()
        outer.addWidget(self._wrap_canvas(self.compare_canvas), 1)
        return page

    def _build_corner_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(14)

        info_card = QFrame()
        info_card.setObjectName("InfoCard")
        ilay = QHBoxLayout(info_card)
        ilay.setContentsMargins(18, 14, 18, 14)
        info = QLabel(
            "<b style='color:#1a2334;'>Custom-spectrum uncertainty</b> "
            "<span style='color:#5a6478;'>·  When a pasted spectrum is retrieved, "
            "this panel shows the corner plot from bootstrap noise "
            "realizations and MC Dropout samples.</span>"
        )
        info.setWordWrap(True)
        ilay.addWidget(info)
        outer.addWidget(info_card)

        self.corner_canvas = CornerCanvas()
        self.corner_canvas.show_placeholder()
        outer.addWidget(self._wrap_canvas(self.corner_canvas), 1)
        return page

    def _build_about_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(24, 22, 24, 24)
        outer.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("AboutHero")
        hero_lay = QHBoxLayout(hero)
        hero_lay.setContentsMargins(36, 28, 36, 28)
        hero_lay.setSpacing(28)

        geexhp_logo = self._about_logo("geexhp.png", 90, 90, framed=False)
        if geexhp_logo is not None:
            hero_lay.addWidget(geexhp_logo)

        copy = QVBoxLayout()
        copy.setSpacing(6)
        kicker = QLabel("TOWARDS THE HABITABLE WORLDS OBSERVATORY  ·  2026")
        kicker.setObjectName("AboutKicker")
        title = QLabel("geeXHP")
        title.setObjectName("PageTitle")
        lead = QLabel(
            "A 1D Convolutional Neural Network framework for retrieving atmospheric "
            "and planetary parameters from reflected-light spectra of Earth-analogue "
            "exoplanets, trained on 108,246 synthetic observations spanning Archean, "
            "Proterozoic, and Modern atmospheres under LUVOIR-B and HabEx/SS noise models."
        )
        lead.setObjectName("PageLead")
        lead.setWordWrap(True)
        copy.addWidget(kicker)
        copy.addWidget(title)
        copy.addWidget(lead)
        hero_lay.addLayout(copy, 1)
        outer.addWidget(hero)

        outer.addWidget(self._build_authors_block())

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(12)
        metrics.addWidget(self._about_metric("108,246", "TRAINING SPECTRA"), 0, 0)
        metrics.addWidget(self._about_metric("10,826", "TEST SAMPLES"), 0, 1)
        metrics.addWidget(self._about_metric("3", "EARTH ERAS"), 0, 2)
        metrics.addWidget(self._about_metric("10", "RETRIEVED PARAMS"), 0, 3)
        metrics.addWidget(self._about_metric("2", "TELESCOPES"), 0, 4)
        outer.addLayout(metrics)

        outer.addWidget(self._build_science_overview_block())

        outer.addStretch(1)
        scroll.setWidget(page)
        return scroll

    # ── Author photo helpers ──────────────────────────────────────────────────

    @staticmethod
    def _round_pixmap(pm: QPixmap, size: int) -> QPixmap:
        """Scale + center-crop a pixmap and apply a circular mask."""
        scaled = pm.scaled(
            size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        x = (scaled.width() - size) // 2
        y = (scaled.height() - size) // 2
        cropped = scaled.copy(x, y, size, size)
        result = QPixmap(size, size)
        result.fill(Qt.transparent)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addEllipse(0.0, 0.0, float(size), float(size))
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, cropped)
        painter.end()
        return result

    @staticmethod
    def _initials_pixmap(initials: str, size: int, color: str) -> QPixmap:
        """Create a circular avatar with initials for authors without a photo."""
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, size, size)
        font = QFont("Inter", size // 3)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor("white"))
        painter.drawText(QRect(0, 0, size, size), Qt.AlignCenter, initials)
        painter.end()
        return pm

    def _build_authors_block(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("AboutCard")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        atitle = QLabel("Authors")
        atitle.setObjectName("AboutCardTitle")
        outer.addWidget(atitle)

        AVATAR = 88
        # (display_name, photo_file_or_None, initials, avatar_color, affiliation, corresponding)
        authors = [
            (
                "Sarah G. A. Barbosa",
                "sarah.jpeg",
                "SB",
                theme.CYAN_DARK,
                "Dept. of Physics\nUFC, Fortaleza, Brazil",
                True,
            ),
            (
                "Raissa Estrela",
                "raissa.png",
                "RE",
                theme.TEL_HABEX,
                "Jet Propulsion Laboratory\nCaltech, USA",
                False,
            ),
            (
                "Cleber da Silva Filho",
                "cleber.jpeg",
                "PS",
                theme.VIOLET,
                "Dept. of Physics\nUFC, Fortaleza, Brazil",
                False,
            ),
            (
                "Lorenzo V. Mugnai",
                "lorenzo.png",
                "LM",
                theme.SKY,
                "School of Physics & Astronomy\nCardiff University, UK",
                False,
            ),
            (
                "Daniel B. de Freitas",
                "daniel.png",
                "DF",
                theme.LIME,
                "Dept. of Physics\nUFC, Fortaleza, Brazil",
                False,
            ),
        ]

        row = QHBoxLayout()
        row.setSpacing(10)

        assets = self.store.project_root / "desktop_app" / "assets"
        for name, photo, initials, color, aff, is_corr in authors:
            chip = QFrame()
            chip.setObjectName("AuthorChip")
            chip_lay = QVBoxLayout(chip)
            chip_lay.setContentsMargins(12, 14, 12, 14)
            chip_lay.setSpacing(6)
            chip_lay.setAlignment(Qt.AlignHCenter)

            # Avatar
            avatar_lbl = QLabel()
            avatar_lbl.setFixedSize(AVATAR, AVATAR)
            avatar_lbl.setAlignment(Qt.AlignCenter)
            if photo:
                path = assets / photo
                if path.exists():
                    src = QPixmap(str(path))
                    if not src.isNull():
                        avatar_lbl.setPixmap(self._round_pixmap(src, AVATAR))
                    else:
                        avatar_lbl.setPixmap(
                            self._initials_pixmap(initials, AVATAR, color)
                        )
                else:
                    avatar_lbl.setPixmap(self._initials_pixmap(initials, AVATAR, color))
            else:
                avatar_lbl.setPixmap(self._initials_pixmap(initials, AVATAR, color))
            chip_lay.addWidget(avatar_lbl, 0, Qt.AlignHCenter)

            display = f"{name}  ★" if is_corr else name
            name_lbl = QLabel(display)
            name_lbl.setObjectName("AuthorName")
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setWordWrap(True)
            chip_lay.addWidget(name_lbl)

            aff_lbl = QLabel(aff)
            aff_lbl.setObjectName("AuthorAff")
            aff_lbl.setAlignment(Qt.AlignCenter)
            aff_lbl.setWordWrap(True)
            chip_lay.addWidget(aff_lbl)
            chip_lay.addStretch(1)

            row.addWidget(chip, 1)

        outer.addLayout(row)
        return frame

    @staticmethod
    def _about_metric(value: str, label: str) -> QWidget:
        card = QFrame()
        card.setObjectName("MetricCard")
        card.setMinimumWidth(118)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(2)
        v = QLabel(value)
        v.setObjectName("MetricValue")
        l = QLabel(label)
        l.setObjectName("MetricLabel")
        lay.addWidget(v)
        lay.addWidget(l)
        return card

    def _stellar_identity_block(self) -> QWidget:
        block = QFrame()
        block.setObjectName("IdentityBlock")
        block.setFixedWidth(300)
        lay = QVBoxLayout(block)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(8)
        lay.addStretch(1)

        logo = self._about_logo("stellar_team.png", 260, 110, framed=False)
        if logo is not None:
            lay.addWidget(logo, 0, Qt.AlignCenter)
        body = QLabel("Federal University of Ceará")
        body.setObjectName("IdentityBody")
        body.setAlignment(Qt.AlignCenter)
        lay.addWidget(body)
        lay.addStretch(1)
        return block

    def _about_logo(
        self, filename: str, width: int, height: int, framed: bool = True
    ) -> Optional[QLabel]:
        path = self.store.project_root / "desktop_app" / "assets" / filename
        if not path.exists():
            return None
        pm = QPixmap(str(path))
        if pm.isNull():
            return None
        logo = QLabel()
        logo.setPixmap(
            pm.scaled(QSize(width, height), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        logo.setFixedSize(width, height)
        logo.setAlignment(Qt.AlignCenter)
        if framed:
            logo.setStyleSheet(
                f"background:{theme.SURFACE}; border:1px solid {theme.BORDER}; "
                "border-radius:8px; padding:8px;"
            )
        return logo

    @staticmethod
    def _about_card(title: str, body: str) -> QWidget:
        card = QFrame()
        card.setObjectName("AboutCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(8)
        t = QLabel(title)
        t.setObjectName("AboutCardTitle")
        b = QLabel(body)
        b.setObjectName("AboutBody")
        b.setWordWrap(True)
        lay.addWidget(t)
        lay.addWidget(b)
        lay.addStretch(1)
        return card

    def _build_science_overview_block(self) -> QWidget:
        card = QFrame()
        card.setObjectName("AboutCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(22, 20, 22, 20)
        lay.setSpacing(14)

        lay.addWidget(
            self._about_text_section(
                "Scientific mission",
                "Direct-imaging missions such as the Habitable Worlds Observatory will "
                "rely on low-resolution reflected-light spectroscopy to characterize "
                "Earth-analogue planets across multiple geological epochs. geeXHP "
                "addresses the atmospheric retrieval problem end-to-end: from "
                "PSG-generated synthetic spectra to simultaneous recovery of six "
                "atmospheric mixing ratios and four planetary parameters.",
            )
        )
        lay.addWidget(
            self._about_text_section(
                "Synthetic dataset",
                "Each of the 108,246 spectra is generated with NASA's Planetary "
                "Spectrum Generator under realistic LUVOIR-B and HabEx/SS instrument "
                "noise, at distances 5–16 pc around F and G stars. Atmospheric "
                "compositions follow literature-based templates for Archean (2.5–4.0 Ga), "
                "Proterozoic (0.5–2.5 Ga), and Modern Earth, with log-space perturbations "
                "in CH4, CO2, H2O, N2, O2, and O3.",
            )
        )
        lay.addWidget(
            self._about_text_section(
                "1D CNN architecture",
                "Three parallel spectral inputs are merged into a 1D CNN with residual "
                "convolutional blocks, attention, global pooling, and dense output "
                "heads that retrieve all 10 parameters simultaneously. Integrated "
                "Gradients attribution identifies UV/blue wavelengths as carrying "
                "the strongest O2 and O3 signal, with secondary contributions from "
                "optical and near-infrared windows for CH4, CO2, and H2O.",
            )
        )
        lay.addWidget(
            self._about_text_section(
                "Uncertainty quantification",
                "sigma_total combines model uncertainty (MC Dropout, pre-computed "
                "across the test set) and data-noise uncertainty (spectral "
                "bootstrapping). At the default noise level, chemical species are "
                "predominantly model-limited rather than photon-limited, future gains "
                "will depend on improved inference as much as on increased photon collection.",
            )
        )

        ref_line = QFrame()
        ref_line.setFrameShape(QFrame.HLine)
        ref_line.setStyleSheet(f"color:{theme.BORDER};")
        lay.addWidget(ref_line)

        ref = QHBoxLayout()
        ref.setSpacing(18)
        logo = self._about_gif_logo(
            "stellar_team_logo_white_bg_24s_24fps.gif", 150, 170
        )
        if logo is None:
            logo = self._about_logo("stellar_team.png", 230, 110, framed=False)
        if logo is not None:
            ref.addWidget(logo, 0, Qt.AlignTop | Qt.AlignLeft)

        ref_copy = QVBoxLayout()
        ref_copy.setSpacing(7)
        rtitle = QLabel("Reference")
        rtitle.setObjectName("AboutCardTitle")
        rbody = QLabel(
            "<b style='color:#1a2334;'>Barbosa, S. G. A., Estrela, R., "
            "da Silva Filho, P. C. F., Mugnai, L. V., &amp; de Freitas, D. B. (2026).</b>"
            "<br><i>Towards the Habitable Worlds Observatory: Retrieval of Reflection "
            "Spectra from Evolving Earth Analogues using 1D CNNs.</i>"
            "<br>RASTI SUBMISSION."
            "<br><br>Data, trained models, and normalization artifacts:"
            "<br><b style='color:#1a2334;'>Zenodo · DOI 10.5281/zenodo.15648637</b>"
            "<br>BSD 2-Clause License."
        )
        rbody.setObjectName("AboutBody")
        rbody.setWordWrap(True)
        rbody.setTextFormat(Qt.RichText)
        ref_copy.addWidget(rtitle)
        ref_copy.addWidget(rbody)
        ref.addLayout(ref_copy, 1)
        lay.addLayout(ref)
        return card

    @staticmethod
    def _about_text_section(title: str, body: str) -> QWidget:
        section = QWidget()
        lay = QVBoxLayout(section)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        t = QLabel(title)
        t.setObjectName("AboutCardTitle")
        b = QLabel(body)
        b.setObjectName("AboutBody")
        b.setWordWrap(True)
        lay.addWidget(t)
        lay.addWidget(b)
        return section

    def _about_gif_logo(
        self, filename: str, width: int, height: int
    ) -> Optional[QLabel]:
        path = self.store.project_root / "desktop_app" / "assets" / filename
        if not path.exists():
            return None
        movie = QMovie(str(path))
        if not movie.isValid():
            return None
        natural = QImageReader(str(path)).size()
        if natural.isEmpty():
            natural = movie.frameRect().size()
        if natural.isEmpty():
            natural = QSize(width, height)
        scaled = natural.scaled(QSize(width, height), Qt.KeepAspectRatio)
        movie.setScaledSize(scaled)
        label = QLabel()
        label.setFixedSize(scaled)
        label.setAlignment(Qt.AlignCenter)
        label.setMovie(movie)
        self._movies.append(movie)
        movie.start()
        return label

    def start_loading(self) -> None:
        self._kick_off_loading()

    def _kick_off_loading(self) -> None:
        self.statusbar.showMessage("Loading test set …")
        self.thread = QThread()
        self.worker = _LoadWorker(self.store)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_load_progress)
        self.worker.finished.connect(self._on_load_finished)
        self.worker.failed.connect(self._on_load_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def _on_load_progress(self, n: int) -> None:
        self.statusbar.showMessage(f"Loaded {n} samples …")

    def _on_load_finished(self, metas: list) -> None:
        self.metas = metas
        self.progress.hide()
        self.statusbar.showMessage(f"Ready — {len(metas):,} samples available.", 6000)
        self._refresh_sample_list()
        self._refresh_ig()
        self.startupReady.emit()

    def _on_load_failed(self, msg: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        QMessageBox.critical(self, "Failed to load TFRecord", msg)
        self.startupReady.emit()

    def _current_network_spec(self) -> Optional[Spectrum]:
        tel = self._selected_telescope()
        if self.rb_custom.isChecked():
            if (
                self.current_custom_spec is None
                or self.current_custom_spec.telescope != tel
            ):
                self._on_custom_spectrum(self.editor.text(), run_after=False)
            if (
                self.current_custom_spec is None
                or self.current_custom_spec.telescope != tel
            ):
                return None
            return self.current_custom_spec

        if self.current_index is None:
            QMessageBox.information(
                self, "Pick a target", "Select a sample on tab 1 first."
            )
            return None
        try:
            return self.store.load_spectrum(self.current_index, tel)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to read spectrum", repr(exc))
            return None

    def _run_network_walkthrough(self) -> None:
        spec = self._current_network_spec()
        if spec is None:
            return
        self.statusbar.showMessage(
            f"Tracing {TELESCOPE_CONFIG[spec.telescope]['label']} network internals …"
        )
        QApplication.processEvents()
        try:
            self.network_stages = self.retriever.trace_network(spec)
        except Exception as exc:
            QMessageBox.critical(self, "Network walkthrough failed", repr(exc))
            self.statusbar.clearMessage()
            return

        self.network_slider.blockSignals(True)
        self.network_slider.setRange(0, max(len(self.network_stages) - 1, 0))
        self.network_slider.blockSignals(False)
        self.network_diagram.set_stages(self.network_stages, 0)
        for btn in (
            self.btn_network_prev,
            self.btn_network_play,
            self.btn_network_next,
        ):
            btn.setEnabled(True)
        self._show_network_stage(0)
        self.statusbar.showMessage(
            f"{TELESCOPE_CONFIG[spec.telescope]['label']} walkthrough ready.", 5000
        )

    def _clear_network_trace(self) -> None:
        self.network_timer.stop()
        self.network_stages = []
        self.network_stage_index = 0
        if not hasattr(self, "network_canvas"):
            return
        self.network_canvas.show_placeholder()
        if hasattr(self, "network_diagram"):
            self.network_diagram.clear()
        self.network_slider.blockSignals(True)
        self.network_slider.setRange(0, 0)
        self.network_slider.setValue(0)
        self.network_slider.blockSignals(False)
        self.lbl_network_stage.setText("No walkthrough loaded.")
        self.lbl_network_inspector_title.setText("Layer inspector")
        self.lbl_network_inspector_body.setText(
            "Run the walkthrough, then hover or click the architecture."
        )
        self.lbl_network_inspector_stats.setText("")
        self.btn_network_play.setText("Play")
        for btn in (
            self.btn_network_prev,
            self.btn_network_play,
            self.btn_network_next,
        ):
            btn.setEnabled(False)

    def _show_network_stage(self, index: int) -> None:
        if not self.network_stages:
            return
        index = max(0, min(index, len(self.network_stages) - 1))
        self.network_stage_index = index
        self.network_slider.blockSignals(True)
        self.network_slider.setValue(index)
        self.network_slider.blockSignals(False)
        self.btn_network_prev.setEnabled(index > 0)
        self.btn_network_next.setEnabled(index < len(self.network_stages) - 1)
        self.btn_network_play.setEnabled(True)
        self.lbl_network_stage.setText(
            f"Step {index + 1} of {len(self.network_stages)}  ·  "
            f"{self.network_stages[index]['title']}"
        )
        self._update_network_inspector(self.network_stages[index])
        self.network_canvas.show_stage(
            self.network_stages,
            index,
            hide=self._hidden_set(),
        )
        self.network_diagram.set_active_index(index)

    def _update_network_inspector(self, stage: dict) -> None:
        values = np.asarray(stage.get("values"), dtype=float)
        shape = " × ".join(str(s) for s in values.shape) if values.size else "—"
        finite = values[np.isfinite(values)] if values.size else np.asarray([])
        if finite.size:
            stats = (
                f"<b>Tensor</b> {shape}<br>"
                f"<b>Mean</b> {float(np.mean(finite)):.4g} &nbsp; "
                f"<b>Min</b> {float(np.min(finite)):.4g} &nbsp; "
                f"<b>Max</b> {float(np.max(finite)):.4g}"
            )
        else:
            stats = f"<b>Tensor</b> {shape}"
        self.lbl_network_inspector_title.setText(stage.get("title", "Layer"))
        self.lbl_network_inspector_body.setText(
            f"{stage.get('explain', stage.get('subtitle', ''))}"
            f"<br><br><span style='color:#5a6478;'>"
            f"{stage.get('reading', '')}</span>"
        )
        self.lbl_network_inspector_stats.setText(stats)

    def _on_network_slider(self, value: int) -> None:
        self.network_timer.stop()
        self.btn_network_play.setText("Play")
        self._show_network_stage(value)

    def _advance_network_stage(self) -> None:
        if not self.network_stages:
            self.network_timer.stop()
            return
        if self.network_stage_index >= len(self.network_stages) - 1:
            self.network_timer.stop()
            self.btn_network_play.setText("Play")
            return
        self._show_network_stage(self.network_stage_index + 1)

    def _toggle_network_play(self) -> None:
        if not self.network_stages:
            self._run_network_walkthrough()
            if not self.network_stages:
                return
        if self.network_timer.isActive():
            self.network_timer.stop()
            self.btn_network_play.setText("Play")
            return
        if self.network_stage_index >= len(self.network_stages) - 1:
            self._show_network_stage(0)
        self.network_timer.start()
        self.btn_network_play.setText("Pause")

    def _on_mode_change(self) -> None:
        self.target_stack.setCurrentIndex(0 if self.rb_filter.isChecked() else 1)
        if self.rb_custom.isChecked():
            self.editor.set_telescope(self._selected_telescope())
        self._clear_network_trace()

    def _on_hide_toggle(self, checked: bool) -> None:
        self.hide_o2o3 = bool(checked)
        self.editor.set_hide_o2o3(self.hide_o2o3)
        if self.last_results:
            tel = next(iter(self.last_results))
            pred, sd = self.last_results[tel]
            self.retrieval_canvas.show_retrieval(
                tel, pred, sd, truth=self.last_truth, hide=self._hidden_set()
            )
            self._populate_table(pred, sd, self.last_truth)
            self._update_detection(tel, pred, sd, self.last_truth)
            if self.last_truth:
                self.compare_canvas.show_compare(
                    self.last_results, truth=self.last_truth, hide=self._hidden_set()
                )
        if self.network_stages:
            self._show_network_stage(self.network_stage_index)

    def _hidden_set(self) -> tuple[str, ...]:
        return ("O2", "O3") if self.hide_o2o3 else ()

    def _filter_metas(self) -> list[SampleMeta]:
        st = self.cmb_star.currentText()
        era = self.cmb_era.currentText()
        dist = self.cmb_distance.currentText()
        out = []
        for m in self.metas:
            if st != "Any" and m.star_type != st:
                continue
            if era != "Any" and m.era.lower() != era.lower():
                continue
            if dist == "Near (≤ 8 pc)" and not (m.distance_pc <= 8):
                continue
            if dist == "Mid (8–12 pc)" and not (8 < m.distance_pc <= 12):
                continue
            if dist == "Far (> 12 pc)" and not (m.distance_pc > 12):
                continue
            out.append(m)
        return out

    def _refresh_sample_list(self) -> None:
        if not self.metas:
            return
        filtered = self._filter_metas()[:1500]
        self.cmb_sample.blockSignals(True)
        self.cmb_sample.clear()
        for m in filtered:
            self.cmb_sample.addItem(m.short_label(), userData=m.index)
        self.cmb_sample.setEnabled(bool(filtered))
        self.cmb_sample.blockSignals(False)
        if filtered:
            self._on_target_change()
        else:
            self.spectrum_canvas.clear()
            self.statusbar.showMessage("No samples match the current filters.", 4000)

    def _selected_telescope(self) -> str:
        return TELESCOPES[self.cmb_telescope.currentIndex()]

    def _on_target_change(self) -> None:
        if hasattr(self, "editor"):
            self.editor.set_telescope(self._selected_telescope())
        if hasattr(self, "lbl_network_context"):
            self.lbl_network_context.setText(
                f"Using {TELESCOPE_CONFIG[self._selected_telescope()]['label']} "
                "for the currently selected spectrum."
            )
        if self.rb_filter.isChecked():
            if self.cmb_sample.count() == 0:
                return
            idx = self.cmb_sample.currentData()
            if idx is None:
                return
            self._set_current(int(idx))
        else:
            if (
                self.current_custom_spec is not None
                and self.current_custom_spec.telescope == self._selected_telescope()
            ):
                self.spectrum_canvas.show_spectrum(
                    self.current_custom_spec,
                    title_suffix="custom pasted spectrum",
                )
            else:
                self.spectrum_canvas.clear()

    def _load_custom_example(self) -> None:
        if self.current_index is None and self.cmb_sample.count() > 0:
            idx = self.cmb_sample.currentData()
        else:
            idx = self.current_index
        if idx is None:
            QMessageBox.information(
                self,
                "No example selected",
                "Select a test sample first, then load it as the paste example.",
            )
            return
        tel = self._selected_telescope()
        try:
            spec = self.store.load_spectrum(int(idx), tel)
            self.editor.set_text(self.store.spectrum_to_text(spec))
            self._set_custom_spectrum(spec, "test-spectrum example")
        except Exception as exc:
            QMessageBox.critical(self, "Example failed", repr(exc))

    def _on_custom_spectrum(self, text: str, run_after: bool = True) -> None:
        tel = self._selected_telescope()
        try:
            spec = self.store.spectrum_from_text(text, tel)
        except Exception as exc:
            QMessageBox.critical(self, "Spectrum format error", str(exc))
            return
        self._set_custom_spectrum(spec, "custom pasted spectrum")
        if run_after:
            self._run_retrieval(tel)

    def _set_custom_spectrum(self, spec: Spectrum, label: str) -> None:
        self.current_custom_spec = spec
        self.current_index = None
        self.lbl_star.setText("Custom")
        self.lbl_dist.setText("—")
        self.lbl_era.setText("Unknown")
        self.lbl_snr_l.setText("—")
        self.lbl_snr_h.setText("—")
        self.last_results.clear()
        self.last_truth = None
        self.compare_canvas.clear()
        self.corner_canvas.show_placeholder()
        self._clear_network_trace()
        self.spectrum_canvas.show_spectrum(spec, title_suffix=label)
        self.statusbar.showMessage(
            f"{TELESCOPE_CONFIG[spec.telescope]['label']} custom spectrum is ready.",
            5000,
        )

    def _set_current(self, idx: int) -> None:
        self.current_index = idx
        self.current_custom_spec = None
        meta = self.metas[idx]
        self.lbl_star.setText(meta.star_type)
        self.lbl_dist.setText(f"{meta.distance_pc:.2f} pc")
        self.lbl_era.setText(meta.era.capitalize())
        self.lbl_snr_l.setText(f"{meta.snr_luvoir_vis:.1f}")
        self.lbl_snr_h.setText(f"{meta.snr_habex_vis:.1f}")

        if meta.era.lower() == "archean" and not self.chk_hide_o2o3.isChecked():
            self.chk_hide_o2o3.setChecked(True)
        elif meta.era.lower() != "archean" and self.chk_hide_o2o3.isChecked():
            self.chk_hide_o2o3.setChecked(False)

        tel = self._selected_telescope()
        try:
            spec = self.store.load_spectrum(idx, tel)
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "Failed to read spectrum", repr(exc))
            return
        self.spectrum_canvas.show_spectrum(spec, title_suffix=meta.short_label())

        self.last_results.clear()
        self.last_truth = meta.truth
        self.compare_canvas.clear()
        self.corner_canvas.show_placeholder()
        self._clear_network_trace()
        self._refresh_ig()

    def _run_retrieval(self, telescope: str) -> None:
        custom_mode = self.rb_custom.isChecked()
        if custom_mode:
            if (
                self.current_custom_spec is None
                or self.current_custom_spec.telescope != telescope
            ):
                self._on_custom_spectrum(self.editor.text(), run_after=False)
            if (
                self.current_custom_spec is None
                or self.current_custom_spec.telescope != telescope
            ):
                return
        elif self.current_index is None:
            QMessageBox.information(
                self, "Pick a target", "Select a sample on tab 1 first."
            )
            return
        self.statusbar.showMessage(
            f"Running {TELESCOPE_CONFIG[telescope]['label']} model …"
        )
        try:
            spec = (
                self.current_custom_spec
                if custom_mode
                else self.store.load_spectrum(self.current_index, telescope)
            )
            predicted, z_pred = self.retriever.predict(spec)
            sigma_z = (
                self.store.reference_uncertainty(telescope)
                if custom_mode
                else self.store.uncertainty(telescope, self.current_index)
            )
            sigma_phys = self.retriever.sigma_z_to_phys(sigma_z, z_pred)
        except Exception as exc:
            QMessageBox.critical(self, "Retrieval failed", repr(exc))
            self.statusbar.clearMessage()
            return

        truth = None if custom_mode else self.metas[self.current_index].truth
        hide = self._hidden_set()
        self.retrieval_canvas.show_retrieval(
            telescope,
            predicted,
            sigma_phys,
            truth=truth,
            hide=hide,
        )
        self._populate_table(predicted, sigma_phys, truth)
        self._update_detection(telescope, predicted, sigma_phys, truth)

        self.last_results[telescope] = (predicted, sigma_phys)
        self.last_truth = truth
        if truth and len(self.last_results) >= 1:
            self.compare_canvas.show_compare(
                self.last_results,
                truth=truth,
                hide=hide,
            )
        elif custom_mode:
            self.compare_canvas.clear()
            self.statusbar.showMessage(
                "Building bootstrap and MC-Dropout samples for the corner plot …"
            )
            QApplication.processEvents()
            try:
                bs = self.retriever.bootstrap_samples(spec, n_samples=5000)
                mc = self.retriever.mc_dropout_samples(spec, n_samples=5000)
                self.corner_canvas.show_corner(bs, mc, spec, hide=hide)
                self.tabs.setCurrentWidget(self.corner_tab)
            except Exception as exc:
                QMessageBox.warning(self, "Corner plot failed", repr(exc))

        self.statusbar.showMessage(
            f"{TELESCOPE_CONFIG[telescope]['label']} retrieval complete.", 5000
        )

    def _refresh_ig(self) -> None:
        tel = self._selected_telescope()
        if self.current_index is not None:
            era = self.metas[self.current_index].era.lower()
        else:
            era = "modern"
        try:
            wave, hmap = self.store.ig_heatmap(tel, era)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load IG heatmaps", repr(exc))
            return
        spec = None
        if self.current_index is not None:
            try:
                spec = self.store.load_spectrum(self.current_index, tel)
            except Exception:
                spec = None
        self.ig_canvas.show_heatmap(wave, hmap, tel, era, spectrum=spec)
        self.lbl_ig_context.setText(
            f"{TELESCOPE_CONFIG[tel]['label'].upper()}   ·   " f"{era.upper()} EARTH"
        )

    def _populate_table(self, predicted, sigma, truth) -> None:
        hidden = self._hidden_set()
        rows = [(i, name) for i, name in enumerate(ALL_PARAMS) if name not in hidden]
        self.results_table.setRowCount(len(rows))
        for r, (i, name) in enumerate(rows):
            unit = PARAM_UNITS_TEXT[i]
            chems = ("O2", "O3", "CH4", "CO2", "H2O", "N2")
            if name in chems:
                unit = "log₁₀ vmr"
            label = f"{PARAM_PLAIN[i]}  ·  {unit}"
            t_val = truth.get(name, np.nan) if truth else np.nan
            p_val = predicted[name]
            s_val = sigma.get(name, 0.0)
            cells = [
                label,
                self._fmt_truth(t_val, name),
                self._fmt_pred(p_val, name),
                self._fmt_sigma(s_val, p_val, name),
            ]
            for j, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                if j == 0:
                    item.setFont(QFont("", -1, QFont.Bold))
                self.results_table.setItem(r, j, item)

    @staticmethod
    def _fmt_truth(val: float, name: str) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        chems = ("O2", "O3", "CH4", "CO2", "H2O", "N2")
        if name in chems:
            if val <= 0:
                return "—"
            return f"{np.log10(val):+.2f}"
        return MainWindow._fmt_phys(val, name)

    @staticmethod
    def _fmt_pred(val: float, name: str) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        chems = ("O2", "O3", "CH4", "CO2", "H2O", "N2")
        if name in chems:
            if val <= 0:
                return "—"
            return f"{np.log10(val):+.2f}"
        return MainWindow._fmt_phys(val, name)

    @staticmethod
    def _fmt_sigma(sigma: float, mean: float, name: str) -> str:
        chems = ("O2", "O3", "CH4", "CO2", "H2O", "N2")
        if name in chems:
            if mean <= 0 or sigma <= 0:
                return "—"
            log_sigma = sigma / (mean * np.log(10))
            return f"±{log_sigma:.2f} dex"
        return MainWindow._fmt_phys(sigma, name)

    @staticmethod
    def _fmt_phys(val: float, name: str) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if "TEMPERATURE" in name:
            return f"{val:.1f}"
        if "PRESSURE" in name:
            return f"{val:.0f}"
        if "RADIUS" in name:
            return f"{val:.3f}"
        if "GRAVITY" in name:
            return f"{val:.2f}"
        return f"{val:.3g}"

    def _update_detection(self, telescope, predicted, sigma, truth) -> None:
        chems = ("O2", "O3", "CH4", "CO2", "H2O", "N2")
        chem_lbl = {
            "O2": "O₂",
            "O3": "O₃",
            "CH4": "CH₄",
            "CO2": "CO₂",
            "H2O": "H₂O",
            "N2": "N₂",
        }
        hidden = self._hidden_set()
        lines = [
            f"<div style='font-size:11px; font-weight:700; "
            f"letter-spacing:1.2px; color:{theme.NAVY_700};'>"
            f"{TELESCOPE_CONFIG[telescope]['label'].upper()}</div>"
            f"<div style='margin-top:6px;'></div>"
        ]
        for chem in chems:
            if chem in hidden:
                continue
            mu = predicted[chem]
            sd = sigma.get(chem, 0.0)
            t = truth.get(chem, 0.0) if truth else 0.0
            sig = mu / sd if sd > 0 else float("inf")
            ok = sig >= DETECTION_SIGMA
            badge_bg = "#dff5e6" if ok else "#fde8ec"
            badge_fg = "#1a8a3a" if ok else "#a13144"
            mark = "✓ detected" if ok else "× below 3σ"
            mu_log = np.log10(mu) if mu > 0 else float("nan")
            t_log = np.log10(t) if t > 0 else float("nan")
            sd_log = sd / (mu * np.log(10)) if (mu > 0 and sd > 0) else float("nan")
            mu_str = "—" if np.isnan(mu_log) else f"{mu_log:+.2f}"
            sd_str = "—" if np.isnan(sd_log) else f"{sd_log:.2f}"
            t_str = "—" if np.isnan(t_log) else f"{t_log:+.2f}"
            ref = (
                f"S/N {sig:.1f}  ·  truth log₁₀ {t_str}"
                if truth
                else f"S/N {sig:.1f}  ·  reference uncertainty"
            )
            lines.append(
                f"<div style='margin: 6px 0;'>"
                f"<span style='background:{badge_bg}; color:{badge_fg};"
                f"padding:1px 8px; border-radius:9px; font-size:10.5px; "
                f"font-weight:700;'>{mark}</span>"
                f"&nbsp;&nbsp;<b style='color:{theme.NAVY_800};'>{chem_lbl[chem]}</b>"
                f"&nbsp;&nbsp;<span style='font-family: monospace; "
                f"color:{theme.INK_DIM}; font-size:11.5px;'>"
                f"log₁₀ {mu_str} ± {sd_str} dex</span>"
                f"<br><span style='color:{theme.INK_FAINT}; font-size:11px; "
                f"margin-left:48px;'>{ref}</span>"
                f"</div>"
            )
        self.detection_label.setText("".join(lines))
