import math
import random
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from desktop_app.app import theme
from desktop_app.app.data import SampleMeta

STAR_PROPS = {
    "F": {"radius_rsun": 1.30, "rgb": (240, 244, 255), "corona": (170, 200, 255)},
    "G": {"radius_rsun": 1.00, "rgb": (255, 240, 200), "corona": (255, 195, 120)},
}

ERA_PALETTE = {
    "modern": {
        "surface_dark": (15, 35, 80),
        "surface_lit": (110, 165, 215),
        "atm": (140, 195, 240),
        "cloud": (245, 250, 255),
    },
    "proterozoic": {
        "surface_dark": (25, 65, 75),
        "surface_lit": (115, 175, 165),
        "atm": (150, 210, 200),
        "cloud": (230, 240, 235),
    },
    "archean": {
        "surface_dark": (60, 30, 15),
        "surface_lit": (210, 145, 85),
        "atm": (230, 160, 90),
        "cloud": (250, 220, 180),
    },
}


def _make_bg_stars(seed: int = 7, n: int = 42) -> list[tuple[float, float, float]]:
    rng = random.Random(seed)
    return [(rng.random(), rng.random(), 0.25 + 0.75 * rng.random()) for _ in range(n)]


class SystemView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(310, 340)
        self.setMaximumSize(390, 430)
        sp = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setSizePolicy(sp)
        self._meta: Optional[SampleMeta] = None
        self._bg_stars = _make_bg_stars()
        self._rotation_phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def sizeHint(self) -> QSize:
        return QSize(350, 390)

    def set_target(self, meta: SampleMeta) -> None:
        self._meta = meta
        if not self._timer.isActive():
            self._timer.start()
        self.update()

    def clear(self) -> None:
        self._meta = None
        self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._rotation_phase = (self._rotation_phase + 0.010) % 1.0
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        card = QPainterPath()
        card.addRoundedRect(rect, 8, 8)
        painter.setClipPath(card)

        self._draw_background(painter, rect)
        self._draw_title(painter, rect)
        if self._meta is None:
            self._draw_empty(painter, rect)
            return
        self._draw_scene(painter, rect)
        self._draw_param_strip(painter, rect)

    def _draw_background(self, painter: QPainter, rect: QRectF) -> None:
        bg = QLinearGradient(rect.topLeft(), rect.bottomRight())
        bg.setColorAt(0.0, QColor("#091323"))
        bg.setColorAt(0.48, QColor("#102743"))
        bg.setColorAt(1.0, QColor("#07101e"))
        painter.fillPath(painter.clipPath(), bg)

        wash = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.bottom())
        wash.setColorAt(0.0, QColor(63, 191, 176, 34))
        wash.setColorAt(0.46, QColor(79, 154, 217, 18))
        wash.setColorAt(1.0, QColor(232, 182, 87, 24))
        painter.fillPath(painter.clipPath(), wash)

        painter.setPen(Qt.NoPen)
        for fx, fy, b in self._bg_stars:
            x = rect.left() + fx * rect.width()
            y = rect.top() + fy * rect.height()
            r = 0.35 + b * 0.85
            color = QColor(255, 255, 255)
            color.setAlphaF(0.26 * b)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, y), r, r)

        vig = QRadialGradient(rect.center(), max(rect.width(), rect.height()) * 0.78)
        vig.setColorAt(0.66, QColor(0, 0, 0, 0))
        vig.setColorAt(1.0, QColor(0, 0, 0, 92))
        painter.setBrush(vig)
        painter.drawRect(rect)

        painter.setPen(QPen(QColor(255, 255, 255, 24), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(1.0, 1.0, -1.0, -1.0), 8, 8)

    def _draw_title(self, painter: QPainter, rect: QRectF) -> None:
        title_rect = QRectF(rect.left() + 16, rect.top() + 12, rect.width() - 32, 26)
        painter.setPen(QColor(235, 244, 255, 238))
        title = QFont("Inter", 10)
        title.setWeight(QFont.Weight.ExtraBold)
        painter.setFont(title)
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignTop, "SYSTEM VIEW")

        if self._meta is None:
            detail = "select target"
        else:
            star = (self._meta.star_type or "G").upper()
            era = (self._meta.era or "modern").replace("_", " ").title()
            detail = f"{star[:1]} star · {era}"
        painter.setPen(QColor(170, 188, 210, 200))
        small = QFont("Inter", 7)
        small.setWeight(QFont.Weight.Bold)
        painter.setFont(small)
        painter.drawText(
            title_rect.adjusted(0, 12, 0, 0),
            Qt.AlignRight | Qt.AlignVCenter,
            detail,
        )

        painter.setPen(QPen(QColor(63, 191, 176, 105), 1.2))
        painter.drawLine(
            QPointF(title_rect.left(), title_rect.bottom() + 3),
            QPointF(
                title_rect.left() + min(76.0, title_rect.width() * 0.34),
                title_rect.bottom() + 3,
            ),
        )

    def _draw_empty(self, painter: QPainter, rect: QRectF) -> None:
        center = rect.center()
        orbit = QRectF(center.x() - 78, center.y() - 54, 156, 108)
        painter.setPen(QPen(QColor(170, 195, 220, 50), 1.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(orbit)
        painter.setBrush(QColor(63, 191, 176, 135))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(center.x() + 58, center.y() - 8), 4.0, 4.0)

        painter.setPen(QColor(218, 230, 244, 205))
        font = QFont("Inter", 10)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.drawText(
            rect.adjusted(34, 68, -34, -54),
            Qt.AlignCenter | Qt.TextWordWrap,
            "Select a sample to preview its host star, planet and atmospheric profile.",
        )

    def _draw_scene(self, painter: QPainter, rect: QRectF) -> None:
        meta = self._meta
        truth = meta.truth or {}

        star_type = (meta.star_type or "G").upper()
        star_type = star_type[0] if star_type else "G"
        if star_type not in STAR_PROPS:
            star_type = "G"
        star_prop = STAR_PROPS[star_type]

        scene_top = rect.top() + 48
        scene_bottom = rect.bottom() - 72
        scene = QRectF(
            rect.left() + 12, scene_top, rect.width() - 24, scene_bottom - scene_top
        )
        usable = min(scene.width(), scene.height())

        star_center = QPointF(
            scene.left() + scene.width() * 0.27,
            scene.top() + scene.height() * 0.33,
        )

        star_r = max(15.0, min(29.0, usable * 0.09 * star_prop["radius_rsun"]))
        orbit_rx = scene.width() * 0.58
        orbit_ry = scene.height() * 0.36
        orbit_angle = 0.42
        orbit = QRectF(
            star_center.x() - orbit_rx,
            star_center.y() - orbit_ry,
            orbit_rx * 2,
            orbit_ry * 2,
        )
        planet_center = QPointF(
            star_center.x() + math.cos(orbit_angle) * orbit_rx,
            star_center.y() + math.sin(orbit_angle) * orbit_ry,
        )

        r_earth = float(truth.get("OBJECT-RADIUS-REL-EARTH", 1.0) or 1.0)
        planet_r = max(15.0, min(star_r * 0.78, 17.0 + (r_earth - 1.0) * 8.0))

        self._draw_orbit(painter, orbit, planet_center)
        self._draw_star(painter, star_center, star_r, star_prop)
        self._draw_planet(
            painter, planet_center, planet_r, star_center, truth, meta.era
        )
        self._draw_scene_labels(
            painter, scene, star_center, planet_center, star_type, meta.era
        )

    def _draw_orbit(
        self,
        painter: QPainter,
        orbit: QRectF,
        planet_center: QPointF,
    ) -> None:
        painter.setBrush(Qt.NoBrush)
        pen = QPen(QColor(220, 235, 250, 42), 1.1)
        pen.setDashPattern([4.0, 5.5])
        painter.setPen(pen)
        painter.drawEllipse(orbit)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(63, 191, 176, 95))
        painter.drawEllipse(planet_center, 2.1, 2.1)

    def _draw_star(
        self, painter: QPainter, center: QPointF, radius: float, prop: dict
    ) -> None:
        r, g, b = prop["rgb"]
        cr, cg, cb = prop["corona"]
        pulse = 0.5 + 0.5 * math.sin(self._rotation_phase * math.tau)
        corona_scale = 1.0 + 0.08 * pulse
        corona_alpha = 70 + int(34 * pulse)

        wide = QRadialGradient(center, radius * 4.2 * corona_scale)
        wide.setColorAt(0.0, QColor(cr, cg, cb, corona_alpha))
        wide.setColorAt(0.35, QColor(cr, cg, cb, 28 + int(18 * pulse)))
        wide.setColorAt(1.0, QColor(cr, cg, cb, 0))
        painter.setBrush(wide)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            center, radius * 4.2 * corona_scale, radius * 4.2 * corona_scale
        )

        inner = QRadialGradient(center, radius * (1.62 + 0.07 * pulse))
        inner.setColorAt(0.0, QColor(255, 255, 255, 230))
        inner.setColorAt(0.40, QColor(r, g, b, 230))
        inner.setColorAt(0.80, QColor(cr, cg, cb, 72 + int(18 * pulse)))
        inner.setColorAt(1.0, QColor(cr, cg, cb, 0))
        painter.setBrush(inner)
        painter.drawEllipse(
            center, radius * (1.62 + 0.07 * pulse), radius * (1.62 + 0.07 * pulse)
        )

        disk = QRadialGradient(
            QPointF(center.x() - radius * 0.25, center.y() - radius * 0.30),
            radius * 1.4,
        )
        disk.setColorAt(0.0, QColor(255, 255, 255))
        disk.setColorAt(0.55, QColor(r, g, b))
        disk.setColorAt(1.0, QColor(max(0, r - 50), max(0, g - 50), max(0, b - 50)))
        painter.setBrush(disk)
        painter.drawEllipse(center, radius, radius)

    def _draw_planet(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        star_center: QPointF,
        truth: dict,
        era: str,
    ) -> None:
        era_key = (era or "modern").lower()
        if era_key not in ERA_PALETTE:
            era_key = "modern"
        pal = ERA_PALETTE[era_key]

        p_surf = float(truth.get("ATMOSPHERE-PRESSURE", 1000.0) or 1000.0)
        p_log = math.log10(max(p_surf, 1.0))
        p_norm = max(0.0, min(3.0, p_log - 1.0))
        halo_extent = radius * (1.18 + 0.07 * p_norm)
        halo_alpha = int(40 + 22 * p_norm)
        ar, ag, ab = pal["atm"]

        halo = QRadialGradient(center, halo_extent)
        halo.setColorAt(0.0, QColor(ar, ag, ab, 0))
        halo.setColorAt(min(0.99, radius / halo_extent), QColor(ar, ag, ab, halo_alpha))
        halo.setColorAt(1.0, QColor(ar, ag, ab, 0))
        painter.setBrush(halo)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, halo_extent, halo_extent)

        dx = star_center.x() - center.x()
        dy = star_center.y() - center.y()
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        focal = QPointF(
            center.x() + ux * radius * 0.55,
            center.y() + uy * radius * 0.55,
        )
        sd_r, sd_g, sd_b = pal["surface_dark"]
        sl_r, sl_g, sl_b = pal["surface_lit"]
        disk = QRadialGradient(focal, radius * 1.55, focal)
        disk.setColorAt(
            0.0, QColor(min(255, sl_r + 30), min(255, sl_g + 30), min(255, sl_b + 30))
        )
        disk.setColorAt(0.45, QColor(sl_r, sl_g, sl_b))
        disk.setColorAt(0.85, QColor(sd_r, sd_g, sd_b))
        disk.setColorAt(
            1.0, QColor(max(0, sd_r - 10), max(0, sd_g - 10), max(0, sd_b - 10))
        )
        painter.setBrush(disk)
        painter.drawEllipse(center, radius, radius)

        self._draw_planet_texture(painter, center, radius, star_center, pal)

        h2o = float(truth.get("H2O", 0.0) or 0.0)
        if h2o > 1e-4:
            self._draw_clouds(painter, center, radius, pal["cloud"], h2o)

        t_surf = float(truth.get("ATMOSPHERE-TEMPERATURE", 288.0) or 288.0)
        if t_surf < 273.0:
            self._draw_ice_caps(painter, center, radius, t_surf)

        edge = QPen(QColor(ar, ag, ab, 160), 1.2)
        painter.setPen(edge)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, radius + 0.6, radius + 0.6)

        gloss = QRadialGradient(
            QPointF(center.x() - radius * 0.34, center.y() - radius * 0.34),
            radius * 1.05,
        )
        gloss.setColorAt(0.0, QColor(255, 255, 255, 58))
        gloss.setColorAt(0.42, QColor(255, 255, 255, 12))
        gloss.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(gloss)
        painter.drawEllipse(center, radius * 0.98, radius * 0.98)

    def _draw_planet_texture(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        star_center: QPointF,
        pal: dict,
    ) -> None:
        sd_r, sd_g, sd_b = pal["surface_dark"]
        painter.save()
        clip = QPainterPath()
        clip.addEllipse(center, radius, radius)
        painter.setClipPath(clip)
        painter.setPen(Qt.NoPen)

        phase_px = ((self._rotation_phase * 2.0) % 1.0) * radius * 2.0
        for xoff, yoff, w, h, alpha in (
            (-0.72, -0.22, 0.62, 0.20, 74),
            (-0.24, 0.16, 0.76, 0.24, 58),
            (0.42, -0.02, 0.48, 0.18, 46),
        ):
            x = center.x() - radius + xoff * radius + phase_px
            while x > center.x() + radius:
                x -= radius * 2.35
            blob = QRectF(x, center.y() + yoff * radius, w * radius, h * radius)
            painter.setBrush(
                QColor(max(0, sd_r - 6), max(0, sd_g - 6), max(0, sd_b - 6), alpha)
            )
            painter.drawRoundedRect(blob, h * radius * 0.5, h * radius * 0.5)

        dx = star_center.x() - center.x()
        dy = star_center.y() - center.y()
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        night = QRadialGradient(
            QPointF(center.x() - ux * radius * 0.44, center.y() - uy * radius * 0.44),
            radius * 1.26,
        )
        night.setColorAt(0.0, QColor(0, 0, 0, 72))
        night.setColorAt(0.58, QColor(0, 0, 0, 28))
        night.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(night)
        painter.drawEllipse(center, radius, radius)
        painter.restore()

    def _draw_scene_labels(
        self,
        painter: QPainter,
        scene: QRectF,
        star_center: QPointF,
        planet_center: QPointF,
        star_type: str,
        era: str,
    ) -> None:
        painter.setPen(QColor(214, 226, 242, 200))
        value = QFont("Inter", 8)
        value.setWeight(QFont.Weight.ExtraBold)
        painter.setFont(value)
        painter.drawText(
            QRectF(star_center.x() - 36, star_center.y() + 34, 72, 16),
            Qt.AlignCenter,
            f"{star_type}-TYPE STAR",
        )

        painter.setPen(QColor(158, 178, 204, 190))
        label = QFont("Inter", 7)
        label.setWeight(QFont.Weight.Bold)
        painter.setFont(label)
        painter.drawText(
            QRectF(
                planet_center.x() - 76,
                min(scene.bottom() - 18, planet_center.y() + 52),
                152,
                16,
            ),
            Qt.AlignCenter,
            (era or "modern").replace("_", " ").upper(),
        )

    def _draw_clouds(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        color: tuple[int, int, int],
        h2o: float,
    ) -> None:
        intensity = max(0.0, min(1.0, math.log10(h2o + 1e-6) / 1.5 + 1.0))
        n_bands = 2 if intensity < 0.5 else 3
        cr, cg, cb = color
        painter.save()
        clip = QPainterPath()
        clip.addEllipse(center, radius, radius)
        painter.setClipPath(clip)
        painter.setPen(Qt.NoPen)
        phase_px = self._rotation_phase * radius * 2.0
        for k in range(n_bands):
            yoff = (-0.42 + 0.35 * k) * radius
            band_h = radius * (0.13 + 0.02 * k)
            band_left = (
                center.x() - radius - radius * 0.45 + phase_px * (1.0 + k * 0.18)
            )
            while band_left > center.x() - radius:
                band_left -= radius * 0.9
            band = QRectF(band_left, center.y() + yoff, radius * 2.85, band_h)
            grad = QLinearGradient(band.topLeft(), band.topRight())
            alpha = int(70 + 80 * intensity)
            grad.setColorAt(0.0, QColor(cr, cg, cb, 0))
            grad.setColorAt(0.25, QColor(cr, cg, cb, alpha // 2))
            grad.setColorAt(0.48, QColor(cr, cg, cb, alpha))
            grad.setColorAt(0.76, QColor(cr, cg, cb, alpha // 2))
            grad.setColorAt(1.0, QColor(cr, cg, cb, 0))
            painter.setBrush(grad)
            painter.drawRoundedRect(band, band_h / 2, band_h / 2)
        painter.restore()

    def _draw_ice_caps(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        t_surf: float,
    ) -> None:
        cap_alpha = int(min(220, 140 + (273.0 - t_surf) * 2))
        white = QColor(245, 250, 255, cap_alpha)
        painter.save()
        clip = QPainterPath()
        clip.addEllipse(center, radius, radius)
        painter.setClipPath(clip)
        painter.setPen(Qt.NoPen)
        painter.setBrush(white)
        cap_w = radius * 1.4
        cap_h = radius * 0.32
        painter.drawEllipse(
            QPointF(center.x(), center.y() - radius * 0.92), cap_w / 2, cap_h
        )
        painter.drawEllipse(
            QPointF(center.x(), center.y() + radius * 0.92), cap_w / 2, cap_h
        )
        painter.restore()

    def _draw_param_strip(self, painter: QPainter, rect: QRectF) -> None:
        truth = self._meta.truth or {}
        items: list[tuple[str, str]] = []
        r_e = truth.get("OBJECT-RADIUS-REL-EARTH")
        t_s = truth.get("ATMOSPHERE-TEMPERATURE")
        p_s = truth.get("ATMOSPHERE-PRESSURE")
        if r_e:
            items.append(("RADIUS", f"{r_e:.2f} R⊕"))
        if t_s:
            items.append(("TEMP", f"{t_s:.0f} K"))
        if p_s:
            p_value = f"{p_s:.0f} mbar" if p_s < 100000 else f"{p_s:.1e} mbar"
            items.append(("PRESSURE", p_value))
        if not items:
            return

        strip_h = 50.0
        strip = QRectF(
            rect.left() + 14,
            rect.bottom() - strip_h - 14,
            rect.width() - 28,
            strip_h,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 50))
        painter.drawRoundedRect(strip.adjusted(0, 2, 0, 2), 8, 8)
        painter.setBrush(QColor(8, 18, 34, 205))
        painter.drawRoundedRect(strip, 8, 8)

        accent = QLinearGradient(strip.topLeft(), strip.topRight())
        accent.setColorAt(0.0, QColor(63, 191, 176, 150))
        accent.setColorAt(0.55, QColor(79, 154, 217, 110))
        accent.setColorAt(1.0, QColor(232, 182, 87, 130))
        painter.setBrush(accent)
        painter.drawRoundedRect(
            QRectF(strip.left(), strip.top(), strip.width(), 3.0), 2, 2
        )

        cw = strip.width() / len(items)
        for i, (label, value) in enumerate(items):
            cell = QRectF(strip.left() + i * cw, strip.top(), cw, strip_h)
            if i:
                painter.setPen(QPen(QColor(220, 235, 250, 28), 1.0))
                painter.drawLine(
                    QPointF(cell.left(), cell.top() + 12),
                    QPointF(cell.left(), cell.bottom() - 10),
                )

            painter.setPen(QColor(145, 166, 194, 220))
            lf = QFont("Inter", 7)
            lf.setWeight(QFont.Weight.Bold)
            painter.setFont(lf)
            painter.drawText(cell.adjusted(0, 9, 0, -26), Qt.AlignCenter, label)
            painter.setPen(QColor(240, 247, 255, 242))
            vf = QFont("Inter", 10)
            vf.setWeight(QFont.Weight.ExtraBold)
            painter.setFont(vf)
            painter.drawText(cell.adjusted(4, 24, -4, -6), Qt.AlignCenter, value)
