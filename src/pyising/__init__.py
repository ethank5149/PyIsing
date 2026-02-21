"""
PyIsing — 2D Ising Model simulation with optional GPU acceleration.

Quick start::

    from pyising import IsingModel, BETA_CRIT
    import numpy as np

    ising = IsingModel(nrows=50, ncols=50, use_gpu=True)
    results = ising.simulate(
        beta_range=np.linspace(0.5 * BETA_CRIT, 10 * BETA_CRIT, 100),
        h_range=[0],
    )
    ising.save_results(results)

Hardware detection::

    from pyising.hardware import detect, print_summary
    info = detect()
    print_summary(info)
"""

from pyising.constants import KB, T_CRIT, BETA_CRIT
from pyising.hardware import detect as detect_hardware, print_summary as print_hardware_summary
from pyising.model import IsingModel

__all__ = [
    "KB",
    "T_CRIT",
    "BETA_CRIT",
    "IsingModel",
    "detect_hardware",
    "print_hardware_summary",
]

__version__ = "0.2.0"
