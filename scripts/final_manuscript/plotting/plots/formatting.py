from typing import TYPE_CHECKING

import matplotlib as mpl
import seaborn as sns
from matplotlib.font_manager import FontProperties

if TYPE_CHECKING:
    from matplotlib.legend import Legend

AX_WIDTH = 8
AX_HEIGHT = 3
FIG_ASPECT = AX_WIDTH / AX_HEIGHT

BLACK = "#000000"


class DiluentKindColor:
    # From IBM colorblind safe palette
    active = "#dc267f"
    inert = BLACK

    @classmethod
    def palette(cls):
        return [cls.active, cls.inert]


class DiluentColor:
    # From IBM colorblind safe palette
    co2 = "#fe6100"
    n2 = "#648fff"

    @classmethod
    def palette(cls):
        return [cls.co2, cls.n2]


class StateVariableColor:
    # From IBM colorblind safe palette
    temp = "#648fff"
    press = "#dc267f"


class ReactionOrSpeciesColor:
    # IBM colorblind safe palette plus black and some extra colors manually checked for probably-goodness
    # https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000-%E5B4FF-%2391ECFF
    r0 = "#ffb000"
    r1 = "#785ef0"
    r2 = "#fe6100"
    r3 = "#91ecff"
    r4 = "#e5b4ff"
    r5 = BLACK

    @classmethod
    def all(cls) -> list[str]:
        colors = []
        idx = 0
        while True:
            this_color_key = f"r{idx}"
            if hasattr(cls, this_color_key):
                colors.append(getattr(cls, this_color_key))
            else:
                break
        return colors


def set_diluent_palette():
    sns.set_palette(DiluentColor.palette())


def set_simulation_result_palette():
    sns.set_palette([StateVariableColor.temp, StateVariableColor.press, *ReactionOrSpeciesColor.all()])


def set_style(grid: bool = False):
    style = "white"
    if grid:
        style += "grid"
    sns.set_style(style)
    sns.set_context("notebook")
    mpl.rcParams.update(
        {
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "font.family": "serif",
            # "text.usetex": True,
        }
    )


def format_diluent_legend(leg: "Legend") -> None:
    t = leg.get_title()
    t.set_text("Diluent")
    t.set_weight("bold")
    leg.set_title("Diluent")
    for t in leg.texts:
        # Matplotlib sometimes requires access to private members
        # ruff: noqa: SLF001
        # noinspection PyProtectedMember,PyUnresolvedReferences
        t.set_text(t._text.replace("2", "$_{2}$"))
