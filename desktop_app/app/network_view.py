from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from desktop_app.app import theme
from desktop_app.app.constants import BANDS, TELESCOPE_CONFIG


class NetworkDiagramView(QWidget):
    stageSelected = Signal(int)

    ARCH = (
        {
            "label": "Spectrum",
            "kind": "Input",
            "stage": 0,
            "body": "Observed reflected-light bins in UV, Visible, and NIR.",
        },
        {
            "label": "Normalize",
            "kind": "Preprocess",
            "stage": 1,
            "body": "Band-wise standardization with training statistics.",
        },
        {
            "label": "Residual Conv",
            "kind": "Conv1D block",
            "stage": 2,
            "body": "Local spectral shapes become learned feature channels.",
        },
        {
            "label": "Downsample",
            "kind": "Stride Conv",
            "stage": 3,
            "body": "The wavelength axis is compressed into wider contexts.",
        },
        {
            "label": "Attention",
            "kind": "MHA + norm",
            "stage": 4,
            "body": "Distant wavelengths can exchange information.",
        },
        {
            "label": "Global Pool",
            "kind": "Embedding",
            "stage": 5,
            "body": "The sequence becomes a compact spectral fingerprint.",
        },
        {
            "label": "Dense Trunk",
            "kind": "Dropout MLP",
            "stage": 6,
            "body": "A shared latent context feeds the retrieval heads.",
        },
    )

    HEADS = (
        {
            "label": "Physical",
            "kind": "4 targets",
            "body": "Radius, gravity, temperature, and pressure.",
            "color": theme.CYAN_DARK,
        },
        {
            "label": "O2 / O3",
            "kind": "2 targets",
            "body": "Main biosignature chemical head.",
            "color": theme.GOLD_DARK,
        },
        {
            "label": "Trace gases",
            "kind": "4 targets",
            "body": "CH4, CO2, H2O, and N2.",
            "color": theme.SKY,
        },
    )

    BAND_COLORS = {
        "UV": "#7e6dbf",
        "Vis": theme.CYAN_DARK,
        "NIR": theme.GOLD_DARK,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(330)
        self._stages: list[dict] = []
        self._active_index = 0
        self._hover: Optional[int] = None
        self._regions: list[dict] = []
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(32)
        self._timer.timeout.connect(self._tick)

    def sizeHint(self) -> QSize:
        return QSize(1080, 350)

    def set_stages(self, stages: list[dict], active_index: int = 0) -> None:
        self._stages = stages
        self._active_index = active_index
        self._hover = None
        if stages and not self._timer.isActive():
            self._timer.start()
        self.update()

    def set_active_index(self, active_index: int) -> None:
        self._active_index = active_index
        self.update()

    def clear(self) -> None:
        self._timer.stop()
        self._stages = []
        self._active_index = 0
        self._hover = None
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        hovered = None
        tip = ""
        for i, region in enumerate(self._regions):
            if region["rect"].contains(pos):
                hovered = i
                tip = region["tip"]
                break
        if hovered != self._hover:
            self._hover = hovered
            self.update()
            if tip:
                QToolTip.showText(event.globalPosition().toPoint(), tip, self)
            else:
                QToolTip.hideText()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = None
        QToolTip.hideText()
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            pos = event.position()
            for region in self._regions:
                if region["rect"].contains(pos):
                    self.stageSelected.emit(region["stage"])
                    break
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_background(painter)
        self._regions = []

        if not self._stages:
            self._draw_empty(painter)
            return

        layout = self._layout()
        self._draw_title_strip(painter, layout)
        self._draw_connections(painter, layout)

        spectrum = self.ARCH[0]
        self._draw_spectrum_card(painter, layout["spectrum"], spectrum, 0)
        for idx, item in enumerate(self.ARCH[1:], start=1):
            self._draw_module_card(
                painter,
                layout[f"stage_{item['stage']}"],
                item,
                idx,
            )
        for head_idx, head in enumerate(self.HEADS):
            self._draw_head_card(
                painter,
                layout[f"head_{head_idx}"],
                head,
                len(self.ARCH) + head_idx,
            )

    def _tick(self) -> None:
        self._phase = (self._phase + 0.018) % 1.0
        self.update()

    def _draw_background(self, painter: QPainter) -> None:
        rect = QRectF(self.rect())
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QColor("#fbfdff"))
        grad.setColorAt(0.62, QColor(theme.SURFACE))
        grad.setColorAt(1.0, QColor("#eef7f6"))
        painter.fillRect(rect, grad)

        painter.setPen(QPen(QColor("#e5edf6"), 1.0))
        y = 34
        while y < self.height():
            painter.drawLine(24, y, self.width() - 24, y)
            y += 44

    def _draw_empty(self, painter: QPainter) -> None:
        card = QRectF(46, 52, self.width() - 92, self.height() - 104)
        self._rounded_panel(painter, card, QColor("#ffffff"), QColor(theme.BORDER))
        painter.setPen(QColor(theme.NAVY_700))
        title = QFont("Inter", 18)
        title.setWeight(QFont.Weight.ExtraBold)
        painter.setFont(title)
        painter.drawText(
            card.adjusted(30, 44, -30, -30),
            Qt.AlignHCenter | Qt.AlignTop,
            "Neural-network walkthrough",
        )
        painter.setPen(QColor(theme.INK_DIM))
        body = QFont("Inter", 11)
        body.setWeight(QFont.Weight.DemiBold)
        painter.setFont(body)
        painter.drawText(
            card.adjusted(70, 94, -70, -30),
            Qt.AlignHCenter | Qt.TextWordWrap,
            "Run the walkthrough to load the live Keras graph. Then hover a layer "
            "to inspect tensor shape and activation statistics, or click a layer "
            "to open its live tensor below.",
        )

    def _layout(self) -> dict[str, QRectF]:
        w = max(float(self.width()), 900.0)
        h = max(float(self.height()), 310.0)
        left = 30.0
        right = 30.0
        top = 58.0
        bottom = 30.0
        graph_h = h - top - bottom
        center_y = top + graph_h * 0.55

        cols = 8
        xs = np.linspace(left + 72.0, w - right - 72.0, cols)
        card_w = float(np.clip((w - left - right) / cols * 0.76, 98.0, 142.0))
        card_h = float(np.clip(graph_h * 0.58, 118.0, 146.0))

        out: dict[str, QRectF] = {
            "title": QRectF(left, 14, w - left - right, 36),
            "spectrum": QRectF(xs[0] - card_w / 2, center_y - card_h / 2, card_w, card_h),
        }
        for item_idx, item in enumerate(self.ARCH[1:], start=1):
            out[f"stage_{item['stage']}"] = QRectF(
                xs[item_idx] - card_w / 2,
                center_y - card_h / 2,
                card_w,
                card_h,
            )

        head_w = card_w + 8.0
        head_h = min(64.0, max(52.0, graph_h / 4.2))
        gap = 12.0
        stack_h = head_h * 3 + gap * 2
        start_y = center_y - stack_h / 2
        for i in range(3):
            out[f"head_{i}"] = QRectF(
                xs[-1] - head_w / 2,
                start_y + i * (head_h + gap),
                head_w,
                head_h,
            )
        return out

    def _draw_title_strip(self, painter: QPainter, layout: dict[str, QRectF]) -> None:
        rect = layout["title"]
        active = self._stage(self._active_index)
        tel = active.get("telescope", "LUVOIR") if active else "LUVOIR"
        tel_label = TELESCOPE_CONFIG.get(tel, {}).get("label", tel)
        painter.setPen(QColor(theme.NAVY_700))
        font = QFont("Inter", 11)
        font.setWeight(QFont.Weight.ExtraBold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, f"{tel_label} live graph")

        painter.setPen(QColor(theme.INK_FAINT))
        small = QFont("Inter", 8)
        small.setWeight(QFont.Weight.Bold)
        painter.setFont(small)
        painter.drawText(
            rect,
            Qt.AlignRight | Qt.AlignVCenter,
            "hover for tensor stats  |  click a layer to inspect",
        )

    def _draw_connections(self, painter: QPainter, layout: dict[str, QRectF]) -> None:
        previous = layout["spectrum"]
        for item in self.ARCH[1:]:
            current = layout[f"stage_{item['stage']}"]
            self._draw_wire(
                painter,
                QPointF(previous.right(), previous.center().y()),
                QPointF(current.left(), current.center().y()),
                self._active_index >= item["stage"],
                theme.CYAN_DARK,
            )
            previous = current

        trunk = layout["stage_6"]
        for i, head in enumerate(self.HEADS):
            current = layout[f"head_{i}"]
            self._draw_wire(
                painter,
                QPointF(trunk.right(), trunk.center().y()),
                QPointF(current.left(), current.center().y()),
                self._active_index >= 7,
                head["color"],
            )

    def _draw_wire(
        self,
        painter: QPainter,
        start: QPointF,
        end: QPointF,
        active: bool,
        color: str,
    ) -> None:
        base = QColor(color if active else theme.BORDER_2)
        base.setAlpha(210 if active else 150)
        pen = QPen(base, 2.4 if active else 1.15)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        path = QPainterPath(start)
        dx = max(32.0, (end.x() - start.x()) * 0.44)
        c1 = QPointF(start.x() + dx, start.y())
        c2 = QPointF(end.x() - dx, end.y())
        path.cubicTo(c1, c2, end)
        painter.drawPath(path)

        if active:
            for offset in (0.0, 0.33, 0.66):
                t = (self._phase + offset) % 1.0
                point = self._cubic_point(start, c1, c2, end, t)
                glow = QColor(color)
                glow.setAlpha(70)
                painter.setBrush(glow)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(point, 8.0, 8.0)
                dot = QColor(color)
                dot.setAlpha(235)
                painter.setBrush(dot)
                painter.drawEllipse(point, 3.4, 3.4)

        painter.setBrush(base)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(end, 3.2, 3.2)

    @staticmethod
    def _cubic_point(
        p0: QPointF, p1: QPointF, p2: QPointF, p3: QPointF, t: float
    ) -> QPointF:
        u = 1.0 - t
        return QPointF(
            (u**3) * p0.x()
            + 3 * (u**2) * t * p1.x()
            + 3 * u * (t**2) * p2.x()
            + (t**3) * p3.x(),
            (u**3) * p0.y()
            + 3 * (u**2) * t * p1.y()
            + 3 * u * (t**2) * p2.y()
            + (t**3) * p3.y(),
        )

    def _draw_spectrum_card(
        self, painter: QPainter, rect: QRectF, item: dict, region_index: int
    ) -> None:
        self._draw_card_shell(painter, rect, item["stage"], region_index)
        x = rect.left() + 12
        y = rect.top() + 12
        self._draw_card_text(painter, rect, item["label"], item["kind"], x, y)

        tel = self._stage(0).get("telescope", "LUVOIR") if self._stage(0) else "LUVOIR"
        cfg = TELESCOPE_CONFIG.get(tel, {})
        bins = cfg.get("bins", {})
        band_y = rect.top() + 52
        for i, band in enumerate(BANDS):
            band_rect = QRectF(x, band_y + i * 22, rect.width() - 24, 16)
            color = QColor(self.BAND_COLORS[band])
            color.setAlpha(42)
            painter.setBrush(color)
            painter.setPen(QPen(QColor(self.BAND_COLORS[band]), 1.0))
            painter.drawRoundedRect(band_rect, 8, 8)
            painter.setPen(QColor(self.BAND_COLORS[band]))
            font = QFont("Inter", 7)
            font.setWeight(QFont.Weight.ExtraBold)
            painter.setFont(font)
            painter.drawText(
                band_rect.adjusted(7, 0, -7, 0),
                Qt.AlignVCenter | Qt.AlignLeft,
                f"{band}  {bins.get(band, '-') } bins",
            )

        self._draw_shape(painter, rect, item["stage"])
        self._register_region(rect, item, region_index)

    def _draw_module_card(self, painter: QPainter, rect: QRectF, item: dict, region_index: int) -> None:
        self._draw_card_shell(painter, rect, item["stage"], region_index)
        x = rect.left() + 12
        y = rect.top() + 12
        self._draw_card_text(painter, rect, item["label"], item["kind"], x, y)
        self._draw_micro_tensor(painter, rect, item["stage"])
        self._draw_shape(painter, rect, item["stage"])
        self._register_region(rect, item, region_index)

    def _draw_head_card(self, painter: QPainter, rect: QRectF, item: dict, region_index: int) -> None:
        self._draw_card_shell(
            painter,
            rect,
            7,
            region_index,
            accent=item["color"],
            compact=True,
        )
        x = rect.left() + 12
        y = rect.top() + 9
        self._draw_card_text(painter, rect, item["label"], item["kind"], x, y, compact=True)
        self._register_region(rect, {**item, "stage": 7}, region_index)

    def _draw_card_shell(
        self,
        painter: QPainter,
        rect: QRectF,
        stage: int,
        region_index: int,
        accent: str = theme.CYAN_DARK,
        compact: bool = False,
    ) -> None:
        active = stage == self._active_index
        hovered = region_index == self._hover
        shadow = QColor(theme.NAVY_900)
        shadow.setAlpha(24 if active or hovered else 12)
        painter.setBrush(shadow)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect.translated(0, 4), 12, 12)

        fill = QColor("#ffffff")
        if active:
            fill = QColor("#e9f6f4")
        elif hovered:
            fill = QColor("#fff8e8")
        border = QColor(accent if active else theme.BORDER_2)
        if hovered:
            border = QColor(theme.GOLD_DARK)
        self._rounded_panel(painter, rect, fill, border, 12, 2.3 if active or hovered else 1.1)

        stripe = QRectF(rect.left(), rect.top(), 5.0, rect.height())
        stripe_color = QColor(accent if stage == 7 else theme.CYAN_DARK)
        stripe_color.setAlpha(230 if active else 145)
        painter.setBrush(stripe_color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(stripe, 3, 3)

        if active:
            painter.setPen(QPen(QColor(theme.CYAN_DARK), 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(-3, -3, 3, 3), 15, 15)

    def _draw_card_text(
        self,
        painter: QPainter,
        rect: QRectF,
        label: str,
        kind: str,
        x: float,
        y: float,
        compact: bool = False,
    ) -> None:
        painter.setPen(QColor(theme.NAVY_800))
        title = QFont("Inter", 8 if compact else 9)
        title.setWeight(QFont.Weight.ExtraBold)
        painter.setFont(title)
        painter.drawText(QRectF(x, y, rect.width() - 22, 18), Qt.AlignLeft, label)

        painter.setPen(QColor(theme.INK_DIM))
        kind_font = QFont("Inter", 6 if compact else 7)
        kind_font.setWeight(QFont.Weight.Bold)
        painter.setFont(kind_font)
        painter.drawText(
            QRectF(x, y + 18, rect.width() - 22, 14),
            Qt.AlignLeft,
            kind.upper(),
        )

    def _draw_micro_tensor(self, painter: QPainter, rect: QRectF, stage: int) -> None:
        values = self._stage_values(stage)
        area = QRectF(rect.left() + 14, rect.top() + 54, rect.width() - 28, 48)
        painter.setPen(QPen(QColor("#edf2f7"), 1))
        painter.setBrush(QColor("#f9fbfd"))
        painter.drawRoundedRect(area, 7, 7)

        if values is None or values.size == 0:
            return
        flat = np.asarray(values, dtype=float).reshape(-1)
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        samples = flat[np.linspace(0, flat.size - 1, 18).astype(int)]
        mag = np.abs(samples)
        scale = float(np.nanmax(mag)) if np.any(np.isfinite(mag)) else 1.0
        scale = scale if scale > 0 else 1.0
        bar_w = area.width() / len(samples)
        mid = area.center().y()
        for i, value in enumerate(samples):
            height = max(2.0, abs(float(value)) / scale * (area.height() * 0.42))
            color = QColor(theme.CYAN_DARK if value >= 0 else theme.ROSE)
            color.setAlpha(195)
            x = area.left() + i * bar_w + 1.5
            y = mid - height if value >= 0 else mid
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(x, y, max(2.0, bar_w - 3), height), 2, 2)
        painter.setPen(QPen(QColor(theme.BORDER_2), 0.8))
        painter.drawLine(QPointF(area.left() + 4, mid), QPointF(area.right() - 4, mid))

    def _draw_shape(self, painter: QPainter, rect: QRectF, stage: int) -> None:
        painter.setPen(QColor(theme.INK_FAINT))
        font = QFont("Inter", 7)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.drawText(
            QRectF(rect.left() + 12, rect.bottom() - 24, rect.width() - 24, 15),
            Qt.AlignLeft,
            self._shape_text(stage),
        )

    @staticmethod
    def _rounded_panel(
        painter: QPainter,
        rect: QRectF,
        fill: QColor,
        border: QColor,
        radius: float = 10.0,
        width: float = 1.0,
    ) -> None:
        painter.setBrush(fill)
        painter.setPen(QPen(border, width))
        painter.drawRoundedRect(rect, radius, radius)

    def _register_region(self, rect: QRectF, item: dict, region_index: int) -> None:
        stage = int(item["stage"])
        self._regions.append(
            {
                "rect": QRectF(rect),
                "stage": stage,
                "tip": self._tooltip(
                    stage,
                    item["label"],
                    item["kind"],
                    item["body"],
                    self._shape_text(stage),
                ),
            }
        )

    def _stage(self, stage: int) -> dict:
        if 0 <= stage < len(self._stages):
            return self._stages[stage]
        return {}

    def _stage_values(self, stage: int) -> Optional[np.ndarray]:
        stage_data = self._stage(stage)
        if stage_data:
            return np.asarray(stage_data.get("values"), dtype=float)
        return None

    def _shape_text(self, stage: int) -> str:
        values = self._stage_values(stage)
        if values is None:
            return "tensor pending"
        shape = " x ".join(str(s) for s in values.shape)
        return f"tensor {shape}"

    def _tooltip(
        self, stage: int, label: str, kind: str, body: str, shape: str
    ) -> str:
        values = self._stage_values(stage)
        stats = ""
        if values is not None and values.size:
            vals = np.asarray(values, dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                stats = (
                    f"<br><span style='color:#5a6478;'>mean "
                    f"{float(np.mean(finite)):.3g} | min "
                    f"{float(np.min(finite)):.3g} | max "
                    f"{float(np.max(finite)):.3g}</span>"
                )
        return (
            f"<b>{label}</b><br>"
            f"<span style='color:#5a6478;'>{kind}</span><br>"
            f"{body}<br>"
            f"<span style='color:#1a2740;'>{shape}</span>"
            f"{stats}<br>"
            "<span style='color:#8b94a6;'>Click to inspect this tensor.</span>"
        )
