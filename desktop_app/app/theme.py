NAVY_900 = "#0b1424"
NAVY_800 = "#111c30"
NAVY_700 = "#1a2740"
NAVY_600 = "#243254"

PAPER = "#eef3f8"
SURFACE = "#ffffff"
SURFACE_2 = "#f8fbff"
BORDER = "#d9e1ec"
BORDER_2 = "#b9c5d4"
INK = "#1a2334"
INK_DIM = "#5a6478"
INK_FAINT = "#8b94a6"

CYAN = "#3fbfb0"
CYAN_DARK = "#2da99a"
CYAN_DEEP = "#1f8f82"
GOLD = "#e8b657"
GOLD_DARK = "#c89538"
LIME = "#9bd76b"
ROSE = "#d5677b"
VIOLET = "#6f73d2"
SKY = "#4f9ad9"

TEL_LUVOIR = "#1f8f82"
TEL_HABEX = "#c4564f"

FONT_STACK = '"Inter", "Lato", "SF Pro Text", "Segoe UI", system-ui, sans-serif'


def stylesheet() -> str:
    return f"""
    /* ---------- base ---------- */
    QMainWindow, QDialog {{
        background-color: {PAPER};
    }}
    QWidget {{
        color: {INK};
        font-family: {FONT_STACK};
        font-size: 14px;
    }}
    QLabel {{
        background: transparent;
    }}
    QToolTip {{
        background-color: {NAVY_800};
        color: white;
        border: 1px solid {NAVY_600};
        padding: 6px 8px;
        border-radius: 4px;
    }}

    /* ---------- header ---------- */
    QFrame#Header {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 {NAVY_900},
                                    stop:0.58 {NAVY_800},
                                    stop:1 {NAVY_700});
        border: none;
        border-bottom: 1px solid {NAVY_600};
    }}
    QLabel#HeaderTitle {{
        color: white;
        font-size: 25px;
        font-weight: 700;
        letter-spacing: 0.2px;
    }}
    QLabel#HeaderSubtitle {{
        color: #c8d5e8;
        font-size: 13.5px;
    }}
    QLabel#HeaderTag {{
        color: {GOLD};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.4px;
    }}

    /* ---------- tabs ---------- */
    QTabWidget::pane {{
        border: none;
        background: {PAPER};
        top: -1px;
    }}
    QTabBar {{
        background: {NAVY_800};
        border-bottom: 1px solid {NAVY_700};
        qproperty-drawBase: 0;
    }}
    QTabBar::tab {{
        background: transparent;
        color: #aebbd0;
        padding: 11px 22px;
        border: none;
        border-bottom: 2px solid transparent;
        font-weight: 600;
        font-size: 13.5px;
        min-width: 110px;
    }}
    QTabBar::tab:hover {{
        color: white;
        background: #16243b;
    }}
    QTabBar::tab:selected {{
        color: white;
        border-bottom: 2px solid {CYAN};
        background: {NAVY_700};
    }}

    /* ---------- group boxes (cards) ---------- */
    QGroupBox {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        margin-top: 18px;
        padding: 18px 14px 12px 14px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 14px;
        padding: 0 8px;
        background: {SURFACE};
        color: {NAVY_700};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
    }}

    QFrame#ToolbarCard, QFrame#CanvasCard, QFrame#InfoCard, QFrame#AboutCard,
    QFrame#AboutHero, QFrame#MetricCard, QFrame#IdentityBlock {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    QFrame#ToolbarCard {{
        background-color: {SURFACE_2};
    }}
    QFrame#AboutHero {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 {NAVY_900},
                                    stop:0.62 {NAVY_800},
                                    stop:1 #20314f);
        border: 1px solid {NAVY_600};
    }}
    QFrame#IdentityBlock {{
        background-color: {SURFACE_2};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    QFrame#MetricCard {{
        background-color: #f9fbfd;
    }}
    QFrame#AuthorChip {{
        background-color: #f4f8fd;
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}

    QLabel#PageTitle {{
        color: white;
        font-size: 29px;
        font-weight: 700;
    }}
    QLabel#PageLead {{
        color: #d7e1ef;
        font-size: 15px;
        line-height: 1.35;
    }}
    QLabel#AboutKicker {{
        color: {GOLD};
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.4px;
    }}
    QLabel#AboutCardTitle {{
        color: {NAVY_800};
        font-size: 15px;
        font-weight: 700;
    }}
    QLabel#AboutBody {{
        color: {INK_DIM};
        font-size: 14px;
        line-height: 1.35;
    }}
    QLabel#IdentityTitle {{
        color: white;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 1.2px;
    }}
    QLabel#IdentityBody {{
        color: {INK_DIM};
        font-size: 12px;
        font-weight: 600;
    }}
    QLabel#AuthorName {{
        color: {INK};
        font-size: 13px;
        font-weight: 700;
    }}
    QLabel#AuthorAff {{
        color: {INK_FAINT};
        font-size: 11px;
        font-weight: 400;
    }}
    QLabel#MetricValue {{
        color: {NAVY_900};
        font-size: 21px;
        font-weight: 800;
    }}
    QLabel#MetricLabel {{
        color: {INK_FAINT};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.1px;
    }}

    /* ---------- inputs ---------- */
    QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox, QPlainTextEdit {{
        background: {SURFACE};
        border: 1px solid {BORDER_2};
        border-radius: 5px;
        padding: 6px 10px;
        min-height: 22px;
        color: {INK};
        selection-background-color: {CYAN};
    }}
    QPlainTextEdit {{
        font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
        font-size: 12.5px;
        line-height: 1.35;
    }}
    QComboBox:hover, QLineEdit:hover, QDoubleSpinBox:hover, QPlainTextEdit:hover {{
        border: 1px solid {CYAN};
    }}
    QComboBox:focus, QLineEdit:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{
        border: 1px solid {CYAN_DARK};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}
    QComboBox::down-arrow {{
        image: none;
        width: 0;
        height: 0;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {INK_DIM};
        margin-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background: {SURFACE};
        color: {INK};
        border: 1px solid {BORDER};
        selection-background-color: {CYAN};
        selection-color: white;
        padding: 4px;
        outline: 0;
    }}

    /* ---------- checkboxes ---------- */
    QCheckBox {{
        spacing: 8px;
        color: {INK_DIM};
        padding: 4px 6px;
        font-weight: 600;
        font-size: 12px;
    }}
    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border: 1.5px solid {BORDER_2};
        border-radius: 3px;
        background: {SURFACE};
    }}
    QCheckBox::indicator:hover {{
        border: 1.5px solid {CYAN};
    }}
    QCheckBox::indicator:checked {{
        background: {CYAN_DARK};
        border: 1.5px solid {CYAN_DARK};
        image: none;
    }}

    /* ---------- radio buttons ---------- */
    QRadioButton {{
        spacing: 8px;
        color: {INK};
        padding: 4px 6px;
    }}
    QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 1.5px solid {BORDER_2};
        border-radius: 9px;
        background: {SURFACE};
    }}
    QRadioButton::indicator:hover {{
        border: 1.5px solid {CYAN};
    }}
    QRadioButton::indicator:checked {{
        border: 1.5px solid {CYAN_DARK};
        background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                                     fx:0.5, fy:0.5,
                                     stop:0 {CYAN_DARK},
                                     stop:0.45 {CYAN_DARK},
                                     stop:0.5 {SURFACE},
                                     stop:1 {SURFACE});
    }}

    /* ---------- buttons ---------- */
    QPushButton {{
        background-color: {SURFACE};
        color: {INK};
        border: 1px solid {BORDER_2};
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 13.5px;
    }}
    QPushButton:hover {{
        background-color: #edf4fb;
        border: 1px solid {CYAN};
        color: {NAVY_800};
    }}
    QPushButton:pressed {{
        background-color: #e1e8f1;
    }}
    QPushButton:disabled {{
        color: {INK_FAINT};
        background-color: {PAPER};
    }}
    QPushButton[primary="true"] {{
        background-color: {CYAN_DARK};
        color: white;
        border: 1px solid {CYAN_DEEP};
    }}
    QPushButton[primary="true"]:hover {{
        background-color: {CYAN_DEEP};
    }}
    QPushButton[primary="true"]:pressed {{
        background-color: #176a60;
    }}
    QPushButton[accent="true"] {{
        background-color: white;
        color: {CYAN_DEEP};
        border: 1px solid {CYAN};
    }}
    QPushButton[accent="true"]:hover {{
        background-color: #e9f6f4;
    }}

    /* ---------- sliders ---------- */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {BORDER};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {CYAN};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {SURFACE};
        border: 2px solid {CYAN_DARK};
        width: 14px;
        height: 14px;
        margin: -6px 0;
        border-radius: 9px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {CYAN};
        border-color: {CYAN_DEEP};
    }}

    /* ---------- tables ---------- */
    QTableWidget {{
        background-color: {SURFACE};
        alternate-background-color: #fafbfd;
        gridline-color: transparent;
        border: 1px solid {BORDER};
        border-radius: 6px;
        selection-background-color: #e9f6f4;
        selection-color: {NAVY_800};
    }}
    QTableWidget::item {{
        padding: 8px 10px;
        border: none;
    }}
    QTableWidget::item:selected {{
        background: #e9f6f4;
        color: {NAVY_800};
    }}
    QHeaderView::section {{
        background-color: {NAVY_700};
        color: white;
        padding: 8px 10px;
        border: none;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.6px;
    }}
    QHeaderView::section:first {{
        border-top-left-radius: 6px;
    }}
    QHeaderView::section:last {{
        border-top-right-radius: 6px;
    }}

    /* ---------- scroll areas ---------- */
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER_2};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {INK_FAINT};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_2};
        border-radius: 4px;
        min-width: 30px;
    }}

    /* ---------- progress bar ---------- */
    QProgressBar {{
        background-color: {BORDER};
        border: none;
        border-radius: 3px;
        height: 6px;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{
        background-color: {CYAN};
        border-radius: 3px;
    }}

    /* ---------- status bar ---------- */
    QStatusBar {{
        background-color: {SURFACE};
        color: {INK_DIM};
        border-top: 1px solid {BORDER};
        font-size: 12.5px;
    }}
    QStatusBar::item {{
        border: none;
    }}

    /* ---------- splitter ---------- */
    QSplitter::handle {{
        background: transparent;
    }}
    QSplitter::handle:horizontal {{
        width: 6px;
    }}
    QSplitter::handle:vertical {{
        height: 6px;
    }}

    /* ---------- info / detection panel labels ---------- */
    QLabel#Hint {{
        color: {INK_DIM};
        font-size: 12px;
    }}
    QLabel#SectionTag {{
        color: {NAVY_700};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
    }}
    QLabel#MutedValue {{
        color: {INK_DIM};
        font-weight: 600;
    }}
    QLabel#KeyValue {{
        color: {INK};
        font-weight: 600;
    }}
    """
