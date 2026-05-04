#!/usr/bin/env bash
# Portable double-click launcher for the geeXHP desktop app.

set -euo pipefail

HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$HERE/.." && pwd )"

export XDG_DATA_DIRS="$HERE/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
export PATH="$HERE:$PATH"

show_error() {
    local title="$1"
    local message="$2"
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title="$title" --text="$message" 2>/dev/null || true
    elif command -v kdialog >/dev/null 2>&1; then
        kdialog --error "$message" --title "$title" 2>/dev/null || true
    elif command -v notify-send >/dev/null 2>&1; then
        notify-send "$title" "$message" 2>/dev/null || true
    else
        printf '%s\n%s\n' "$title" "$message" >&2
    fi
}

if [[ -x "$PROJECT_ROOT/psg-venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/psg-venv/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    show_error "geeXHP could not start" "Python 3 was not found."
    exit 1
fi

if ! "$PYTHON_BIN" -c "import PySide6" >/dev/null 2>&1; then
    show_error \
        "geeXHP dependencies are missing" \
        "The app found Python, but PySide6 is not installed in it.\n\nUse the project's psg-venv, or install desktop_app/requirements.txt in your environment."
    exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m desktop_app.main
