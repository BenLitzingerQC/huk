"""Centralised matplotlib configuration.

Call :func:'setup_plotting' before creating any figures to activate
the non-interactive ''agg'' backend and apply a consistent visual style.

NOTE: Update to use the HUK colors
"""


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

def setup_plotting() -> None:
    
    huk_yellow = "#FFD500"
    huk_dark = "#16181B"
    huk_light = "#F5F5F5"

    huk_cmap = LinearSegmentedColormap.from_list(
        "huk_heatmap",
        ["#FFFFFF", huk_yellow, huk_dark]
    )

    plt.colormaps.register(cmap=huk_cmap, name="huk_heatmap", force=True)

    matplotlib.rcParams.update(
        {
            "backend": "agg",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "cm",
            
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,

            "figure.constrained_layout.use": True,

            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": huk_dark,

            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.color": "#CCCCCC",
            "axes.axisbelow": True,

            "axes.prop_cycle": matplotlib.cycler(color=[huk_yellow, huk_dark, "gray", "lightgray"])
        }
    )
