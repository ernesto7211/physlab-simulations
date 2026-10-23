"""Shared app metadata and plot colours (no Streamlit dependency).

Colours are the Okabe-Ito palette, designed to stay distinguishable under the
common forms of colour-vision deficiency. Series are also separated by marker
shape, line style or labels where meaning would otherwise depend on colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass

GITHUB_URL = "https://github.com/ernesto7211/physlab-simulations"
APP_NAME = "PhysLab"
APP_TITLE = "PhysLab: Interactive Numerical Physics Simulations"

# Okabe-Ito
BLACK = "#000000"
ORANGE = "#E69F00"
SKY_BLUE = "#56B4E9"
BLUISH_GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
REDDISH_PURPLE = "#CC79A7"

PALETTE = (BLACK, ORANGE, SKY_BLUE, BLUISH_GREEN, YELLOW, BLUE, VERMILLION, REDDISH_PURPLE)

# Sequential / diverging colour scales that are perceptually ordered and CVD-friendly.
SCALE_SEQUENTIAL = "Viridis"
SCALE_DIVERGING = "RdBu_r"


@dataclass(frozen=True)
class PageInfo:
    key: str
    title: str  # used for navigation label AND the page heading
    icon: str
    path: str  # relative to the repository root
    summary: str  # one sentence: what it shows and what you can change


SIMULATION_PAGES = (
    PageInfo(
        "orbital",
        "Orbital Mechanics",
        "🪐",
        "views/orbital_page.py",
        "Shows a two-body gravitational orbit with its energy and energy drift, "
        "and lets you change both masses, the starting distance and speed, and the duration "
        "(or start from circular, elliptical or escape presets).",
    ),
    PageInfo(
        "pendulum",
        "Double Pendulum",
        "🌀",
        "views/pendulum_page.py",
        "Shows a double pendulum animated next to a nearly identical copy so you can watch "
        "them diverge, and lets you change the angles, rod lengths, masses, gravity, "
        "duration and the size of the perturbation.",
    ),
    PageInfo(
        "electric",
        "Electric Fields",
        "⚡",
        "views/electric_page.py",
        "Shows the electric field lines, vectors and equipotentials of up to six point "
        "charges, and lets you change each charge's position and signed magnitude "
        "(or start from single-charge, dipole, like-charges and quadrupole presets).",
    ),
)

HOME_ICON = "🏠"
PAGE_BY_KEY = {p.key: p for p in SIMULATION_PAGES}


# Automated test suites, shown on the homepage. A test in tests/test_theme.py
# collects the suite with pytest and fails if these counts go stale.
# (file, number of collected tests, what it actually checks)
TEST_SUITES = (
    ("tests/test_orbital.py", 4,
     "Circular-orbit radius stays constant; orbital period matches Kepler's third law; energy and "
     "angular momentum are conserved; an orbit at escape speed has total energy about zero and moves outward."),
    ("tests/test_orbital_helpers.py", 16,
     "Orbit type, eccentricity, period and apsides against analytic values and the simulation; "
     "rejection of invalid inputs; CSV table contents."),
    ("tests/test_pendulum.py", 12,
     "Small-angle normal-mode periods against linear theory; closed-form frequencies for the equal case; "
     "energy conservation; rod lengths stay fixed; rest at equilibrium; large-angle pairs diverge far more "
     "than small-angle pairs; the growth-rate fit recovers a known synthetic exponential; invalid parameters are rejected."),
    ("tests/test_pendulum_helpers.py", 26,
     "Input validation, animation frame selection, CSV table contents."),
    ("tests/test_electric.py", 13,
     "Single-charge field and potential against kQ/r^2 and kQ/r; superposition; dipole field on the "
     "perpendicular bisector; masked points are NaN, not infinite; numerical gradient of V against -E; "
     "field-line behaviour (radial lines, end points, length limit); invalid inputs."),
    ("tests/test_electric_helpers.py", 32,
     "Input validation, presets, contour levels, arrow geometry, CSV table contents."),
    ("tests/test_theme.py", 6,
     "Palette and page metadata are valid, page files exist, no Matplotlib imports, and the test counts "
     "listed here match what pytest collects."),
)
TEST_TOTAL = sum(n for _, n, _ in TEST_SUITES)
