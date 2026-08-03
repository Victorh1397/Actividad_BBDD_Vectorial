"""Render informe.html to informe.pdf using a headless Chromium browser.

The report is authored in HTML rather than Markdown so the ten-page limit of
RF-29 can actually be controlled: page breaks, table widths and orphan lines are
decisions here, not accidents of a converter.

No dependency is added for this. Any Chromium-based browser already installed
prints to PDF from the command line, and every machine that can run this project
has one.

    uv run python docs/informe/generar_pdf.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "informe.html"
TARGET = HERE / "informe.pdf"
PAGE_LIMIT = 10

CANDIDATES = (
    "chrome",
    "chromium",
    "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_browser() -> str:
    for candidate in CANDIDATES:
        resolved = shutil.which(candidate) or (
            candidate if os.path.isfile(candidate) else None
        )
        if resolved:
            return resolved
    raise SystemExit(
        "No se encontró ningún navegador Chromium. Instale Chrome o Edge, o "
        "abra informe.html y use Imprimir → Guardar como PDF."
    )


def count_pages(pdf: Path) -> int:
    """Count /Type /Page objects. Crude, but enough to enforce the limit."""
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"No existe {SOURCE}")

    browser = find_browser()
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={TARGET}",
            SOURCE.as_uri(),
        ],
        check=True,
        capture_output=True,
    )

    pages = count_pages(TARGET)
    size_kb = round(TARGET.stat().st_size / 1024)
    print(f"{TARGET.name}: {pages} páginas · {size_kb} KB")

    if pages > PAGE_LIMIT:
        # El límite es del enunciado, así que incumplirlo es un fallo, no un aviso.
        print(f"EXCEDE el límite de {PAGE_LIMIT} páginas de RF-29.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
