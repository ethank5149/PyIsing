"""
Benchmark runner for PyIsing Monte Carlo methods.

Runs all Monte Carlo methods with identical parameters and initial conditions,
streaming results via callbacks as each method completes.  The shared initial
lattice guarantees a fair comparison across algorithms.

Usage::

    from pyising.benchmark import run_benchmark, results_to_dataframe

    results = run_benchmark()
    print(results_to_dataframe(results))
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from pyising.model import IsingModel, _to_numpy
from pyising.constants import BETA_CRIT  # noqa: F401 — re-exported for convenience


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Parameters shared across every benchmarked method.

    Attributes
    ----------
    nrows, ncols : int
        Lattice dimensions.
    tf : int
        Total simulation time (seconds of model time) for frame-by-frame methods.
    fps : int
        Frames per second for frame-by-frame methods.
    beta : float
        Inverse temperature.  Default ≈ 1.36 × β_c, near the critical point.
    h : float
        External magnetic field.
    J : float
        Coupling constant.
    mu : float
        Magnetic moment.
    seed : int
        RNG seed used to generate the shared initial lattice.
    use_gpu : bool
        Whether to request GPU acceleration.
    wl_flatness : float
        Wang–Landau flatness criterion.
    wl_f_min : float
        Wang–Landau stopping modification factor.
    wl_max_mc_sweeps : int
        Maximum MC sweeps per Wang–Landau flatness iteration.
    pt_n_sweeps : int
        Number of sweeps for Parallel Tempering.
    pt_swap_interval : int
        Swap interval for Parallel Tempering.
    pt_n_replicas : int
        Number of temperature replicas for Parallel Tempering.
    """

    nrows: int = 20
    ncols: int = 20
    tf: int = 5
    fps: int = 20
    beta: float = 0.6  # ~1.36 * BETA_CRIT, near critical point
    h: float = 0.0
    J: float = 1.0
    mu: float = 1.0
    seed: int = 42
    use_gpu: bool = False
    # Wang-Landau specific
    wl_flatness: float = 0.8
    wl_f_min: float = 1e-4
    wl_max_mc_sweeps: int = 1000
    # Parallel Tempering specific
    pt_n_sweeps: int = 100
    pt_swap_interval: int = 1
    pt_n_replicas: int = 10


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Observables and timing for a single benchmarked method.

    Attributes
    ----------
    method : str
        Name of the Monte Carlo method (e.g. ``'wolff'``).
    category : str
        ``'local'``, ``'cluster'``, or ``'advanced'``.
    energy : float
        Final-frame energy ⟨E⟩ (per-spin for advanced methods).
    magnetization : float
        Final-frame magnetization ⟨M⟩.
    abs_magnetization : float
        Absolute value of the final magnetization |⟨M⟩|.
    specific_heat : float
        Final-frame specific heat capacity C.
    susceptibility : float
        Final-frame magnetic susceptibility χ.
    wall_time : float
        Wall-clock time in seconds.
    final_lattice : numpy.ndarray
        ``(nrows, ncols)`` int8 array — the final spin configuration.
    """

    method: str
    category: str
    energy: float
    magnetization: float
    abs_magnetization: float
    specific_heat: float
    susceptibility: float
    wall_time: float
    final_lattice: np.ndarray


# ---------------------------------------------------------------------------
# Method taxonomy
# ---------------------------------------------------------------------------

METHOD_CATEGORIES: dict[str, str] = {
    'metropolis-hastings': 'local',
    'glauber': 'local',
    'overrelaxation': 'local',
    'wolff': 'cluster',
    'swendsen-wang': 'cluster',
    'invaded-cluster': 'cluster',
    'kinetic-mc': 'local',
    'wang-landau': 'advanced',
    'parallel-tempering': 'advanced',
}

ALL_METHODS: list[str] = list(METHOD_CATEGORIES.keys())

_FRAME_METHODS: set[str] = {
    'metropolis-hastings',
    'glauber',
    'overrelaxation',
    'wolff',
    'swendsen-wang',
    'invaded-cluster',
    'kinetic-mc',
}


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    config: Optional[BenchmarkConfig] = None,
    methods: Optional[list[str]] = None,
    on_result: Optional[Callable[[BenchmarkResult], None]] = None,
    on_status: Optional[Callable[[str, str], None]] = None,
    on_complete: Optional[Callable[[list[BenchmarkResult]], None]] = None,
) -> list[BenchmarkResult]:
    """Run Monte Carlo benchmarks and stream results via callbacks.

    Parameters
    ----------
    config : BenchmarkConfig or None
        Simulation parameters.  Defaults to ``BenchmarkConfig()`` when *None*.
    methods : list[str] or None
        Which methods to benchmark.  Defaults to all nine methods when *None*.
    on_result : callable or None
        ``on_result(result)`` is called after each method completes.
    on_status : callable or None
        ``on_status(method, status)`` is called when a method starts
        (``status='running'``).
    on_complete : callable or None
        ``on_complete(results)`` is called once all methods have finished.

    Returns
    -------
    list[BenchmarkResult]
        One :class:`BenchmarkResult` per benchmarked method, in execution order.
    """
    if config is None:
        config = BenchmarkConfig()
    if methods is None:
        methods = ALL_METHODS

    # ------------------------------------------------------------------
    # 1. Generate the shared initial lattice from the seed
    # ------------------------------------------------------------------
    rng = np.random.default_rng(config.seed)
    initial_lattice = (
        2 * rng.integers(0, 2, size=(config.nrows, config.ncols)).astype(np.int8) - 1
    )

    results: list[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # 2. Iterate over requested methods
    # ------------------------------------------------------------------
    for method in methods:
        if method not in METHOD_CATEGORIES:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: {list(METHOD_CATEGORIES.keys())}"
            )

        if on_status is not None:
            on_status(method, 'running')

        t_start = time.perf_counter()

        # --------------------------------------------------------------
        # 2a. Frame-by-frame methods
        # --------------------------------------------------------------
        if method in _FRAME_METHODS:
            ising = IsingModel(
                tf=config.tf,
                fps=config.fps,
                nrows=config.nrows,
                ncols=config.ncols,
                J=config.J,
                mu=config.mu,
                beta=config.beta,
                h=config.h,
                method=method,
                use_gpu=config.use_gpu,
            )
            # Copy the shared initial lattice into frame 0
            if ising.use_gpu:
                ising.at[0] = ising.xp.asarray(initial_lattice.copy())
            else:
                ising.at[0] = initial_lattice.copy()

            ising.quench(verbose=False)
            ising.gather_data(verbose=False)

            energy = float(ising.energy[-1])
            magnetization = float(ising.magnetization[-1])
            abs_mag = abs(magnetization)
            specific_heat = float(ising.specific_heat_capacity[-1])
            susceptibility = float(ising.magnetic_susceptibility[-1])
            final_lattice = _to_numpy(ising.at[-1])

        # --------------------------------------------------------------
        # 2b. Wang-Landau
        # --------------------------------------------------------------
        elif method == 'wang-landau':
            ising = IsingModel(
                tf=1,
                fps=2,
                nrows=config.nrows,
                ncols=config.ncols,
                J=config.J,
                mu=config.mu,
                beta=config.beta,
                h=config.h,
                use_gpu=False,
            )
            wl = ising.simulate_wang_landau(
                flatness=config.wl_flatness,
                f_min=config.wl_f_min,
                max_mc_sweeps=config.wl_max_mc_sweeps,
            )
            df = ising.wang_landau_observables(wl, beta_range=[config.beta])

            energy = float(df['Energy'].iloc[0])
            specific_heat = float(df['SpecificHeatCapacity'].iloc[0])
            magnetization = 0.0  # WL doesn't track magnetization
            abs_mag = 0.0
            susceptibility = 0.0
            final_lattice = np.zeros(
                (config.nrows, config.ncols), dtype=np.int8
            )

        # --------------------------------------------------------------
        # 2c. Parallel Tempering
        # --------------------------------------------------------------
        elif method == 'parallel-tempering':
            ising = IsingModel(
                tf=1,
                fps=2,
                nrows=config.nrows,
                ncols=config.ncols,
                J=config.J,
                mu=config.mu,
                beta=config.beta,
                h=config.h,
                use_gpu=config.use_gpu,
            )
            beta_range = np.linspace(
                config.beta * 0.5, config.beta * 2.0, config.pt_n_replicas
            )
            df = ising.simulate_parallel_tempering(
                beta_range=beta_range,
                h=config.h,
                n_sweeps=config.pt_n_sweeps,
                swap_interval=config.pt_swap_interval,
            )
            # Find the row closest to the target beta
            idx = (df['ThermodynamicBeta'] - config.beta).abs().idxmin()
            energy = float(df.loc[idx, 'Energy'])
            magnetization = float(df.loc[idx, 'Magnetization'])
            abs_mag = abs(magnetization)
            specific_heat = float(df.loc[idx, 'SpecificHeatCapacity'])
            susceptibility = float(df.loc[idx, 'MagneticSusceptibility'])
            final_lattice = np.zeros(
                (config.nrows, config.ncols), dtype=np.int8
            )

        wall_time = time.perf_counter() - t_start

        result = BenchmarkResult(
            method=method,
            category=METHOD_CATEGORIES[method],
            energy=energy,
            magnetization=magnetization,
            abs_magnetization=abs_mag,
            specific_heat=specific_heat,
            susceptibility=susceptibility,
            wall_time=wall_time,
            final_lattice=final_lattice,
        )
        results.append(result)

        if on_result is not None:
            on_result(result)

    # ------------------------------------------------------------------
    # 3. All done
    # ------------------------------------------------------------------
    if on_complete is not None:
        on_complete(results)

    return results


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark results to a pandas DataFrame for easy display.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Output of :func:`run_benchmark`.

    Returns
    -------
    pd.DataFrame
        Human-readable table with formatted numeric columns.
    """
    return pd.DataFrame([
        {
            'Method': r.method,
            'Category': r.category,
            'Energy ⟨E⟩': f'{r.energy:.4f}',
            'Magnetization ⟨M⟩': f'{r.magnetization:.4f}',
            '|⟨M⟩|': f'{r.abs_magnetization:.4f}',
            'Specific Heat C': f'{r.specific_heat:.4f}',
            'Susceptibility χ': f'{r.susceptibility:.4f}',
            'Wall Time (s)': f'{r.wall_time:.3f}',
        }
        for r in results
    ])
