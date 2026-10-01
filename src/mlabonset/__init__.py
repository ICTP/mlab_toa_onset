"""
Pulse Onset Mlab Library

Modules:
    - simulator: Pulse generation (semi-Gaussian, ramp, datasets)
    - onset:     Onset/time-of-arrival detection algorithms (DLIM, CFD, DLED)
    - stats:     Statistical analysis for synthetic and real datasets
"""

from .simulator import PulseSimulator
from .onset import DLIM, DCFD, DLED
from .stats import (
    PulseStatistics,
    plot_overlaid_histograms,
    compare_methods,
    label_and_create_dataset,
    plot_relacion_acc_pre,
    dlim_window_scan,
)

__all__ = [
    "PulseSimulator",
    "DLIM",
    "DCFD",
    "DLED",
    "PulseStatistics",
    "plot_overlaid_histograms",
    "compare_methods",
    "label_and_create_dataset",
    "plot_relacion_acc_pre",
    "dlim_window_scan"
]
