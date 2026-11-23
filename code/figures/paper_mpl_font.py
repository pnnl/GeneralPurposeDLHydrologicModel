"""Match figure text to the manuscript's Computer Modern roman."""

from pathlib import Path

from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

_FILES = (
    "lmroman10-regular.otf",
    "lmroman10-bold.otf",
    "lmroman10-italic.otf",
    "lmroman10-bolditalic.otf",
)
_READY = False


def _register():
    found = {}
    local = Path(__file__).resolve().parent / "fonts"
    for name in _FILES:
        candidate = local / name
        if candidate.is_file():
            found[name] = str(candidate)
    if len(found) < len(_FILES):
        for font_path in fm.findSystemFonts():
            name = Path(font_path).name
            if name in _FILES and name not in found:
                found[name] = font_path
    missing = [name for name in _FILES if name not in found]
    if missing:
        raise FileNotFoundError(
            "Latin Modern Roman is not on the font path. Missing "
            + ", ".join(missing)
            + ". Install the TeX Live lmroman fonts, or place those files in fonts/ next to this script."
        )
    for name in _FILES:
        fm.fontManager.addfont(found[name])


def use():
    """Register Latin Modern Roman and use it for new figures."""
    global _READY
    if not _READY:
        _register()
        _READY = True
    plt.rcParams.update({
        "font.family": "Latin Modern Roman",
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
