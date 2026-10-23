"""Tests for shared app metadata, palette, and project-wide constraints."""

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from physlab import theme

ROOT = Path(__file__).resolve().parent.parent
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_palette_is_valid_unique_okabe_ito():
    assert len(theme.PALETTE) == 8
    assert all(HEX.match(c) for c in theme.PALETTE)
    assert len({c.lower() for c in theme.PALETTE}) == len(theme.PALETTE)
    # the two colours used for "positive" vs "negative" must differ
    assert theme.VERMILLION != theme.BLUE


def test_page_metadata_is_complete_and_unique():
    pages = theme.SIMULATION_PAGES
    assert len(pages) == 3
    for attr in ("key", "title", "icon", "path"):
        values = [getattr(p, attr) for p in pages]
        assert len(set(values)) == len(values)
        assert all(values)
    assert all(p.summary.strip().endswith(".") for p in pages)
    assert set(theme.PAGE_BY_KEY) == {p.key for p in pages}


def test_page_files_exist_and_are_registered_in_app():
    app_text = (ROOT / "app.py").read_text(encoding="utf-8-sig")
    for p in theme.SIMULATION_PAGES:
        assert (ROOT / p.path).is_file()
        assert "pages/" not in p.path  # a pages/ folder triggers Streamlit's legacy navigation
    assert "SIMULATION_PAGES" in app_text


def test_github_url_format():
    assert re.match(r"^https://github\.com/[\w.-]+/[\w.-]+$", theme.GITHUB_URL)


def test_no_matplotlib_anywhere_in_project_code():
    offenders = []
    for path in list(ROOT.glob("*.py")) + list(ROOT.glob("physlab/*.py")) + list(ROOT.glob("views/*.py")):
        text = path.read_text(encoding="utf-8-sig")
        if re.search(r"^\s*(import|from)\s+matplotlib", text, re.M):
            offenders.append(path.name)
    assert not offenders
    reqs = (ROOT / "requirements.txt").read_text().lower()
    assert "matplotlib" not in reqs


def test_homepage_test_counts_match_pytest_collection():
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    collected = Counter(line.split("::")[0] for line in out.splitlines() if "::" in line)
    listed = {name: n for name, n, _ in theme.TEST_SUITES}
    assert dict(collected) == listed
    assert theme.TEST_TOTAL == sum(collected.values())
