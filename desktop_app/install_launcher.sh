#!/usr/bin/env bash
# Install a desktop launcher for geeXHP so it appears in the application menu.
# Nothing system-wide is modified — everything goes into ~/.local.

set -euo pipefail

HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$HERE/.." && pwd )"

# ── Icon ──────────────────────────────────────────────────────────────────────
ICON_SRC="$HERE/assets/geexhp.png"
if [[ ! -f "$ICON_SRC" ]]; then
    echo "ERROR: $ICON_SRC not found."
    exit 1
fi
ICON_DIR="$HOME/.local/share/icons/hicolor/1024x1024/apps"
mkdir -p "$ICON_DIR"
cp "$ICON_SRC" "$ICON_DIR/geexhp.png"

# ── Desktop entry ─────────────────────────────────────────────────────────────
DESKTOP_DIR="$HOME/.local/share/applications"
mkdir -p "$DESKTOP_DIR"
DESKTOP_FILE="$DESKTOP_DIR/geexhp.desktop"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=geeXHP
GenericName=Exoplanet Atmospheric Retrieval
Comment=Retrieve atmospheric parameters from reflected-light spectra
Exec=$HERE/run_geexhp.sh
Icon=$ICON_DIR/geexhp.png
Terminal=false
Categories=Science;Astronomy;Education;
Keywords=exoplanet;astronomy;atmosphere;ML;retrieval;
StartupWMClass=geeXHP
EOF

chmod +x "$DESKTOP_FILE"

# ── Refresh caches ────────────────────────────────────────────────────────────
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
fi

echo "✓ Installed: $DESKTOP_FILE"
echo "  Exec:  $HERE/run_geexhp.sh"
echo "  Icon:  $ICON_DIR/geexhp.png"
echo
echo "Launch from the application menu (search 'geeXHP'), or run:"
echo "    gtk-launch geexhp"
