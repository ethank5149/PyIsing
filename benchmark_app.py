"""
PyIsing — Benchmark Dashboard (Marimo app).

Run locally:
    marimo run benchmark_app.py

Edit/develop:
    marimo edit benchmark_app.py

Runs all 9 Monte Carlo methods under identical conditions and displays
side-by-side comparison charts of physically relevant observables plus
wall-clock timing.  Results stream in progressively as each method completes.
"""

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full", app_title="PyIsing Benchmark")


# ---------------------------------------------------------------------------
# Cell: Imports
# ---------------------------------------------------------------------------

@app.cell
def _imports():
    import threading

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from pyising import detect_hardware, print_hardware_summary
    from pyising.benchmark import (
        ALL_METHODS,
        METHOD_CATEGORIES,
        BenchmarkConfig,
        BenchmarkResult,
        results_to_dataframe,
        run_benchmark,
    )
    return (
        ALL_METHODS,
        BenchmarkConfig,
        BenchmarkResult,
        METHOD_CATEGORIES,
        detect_hardware,
        mo,
        np,
        pd,
        plt,
        print_hardware_summary,
        results_to_dataframe,
        run_benchmark,
        threading,
    )


# ---------------------------------------------------------------------------
# Cell: Hardware detection
# ---------------------------------------------------------------------------

@app.cell
def _hardware(detect_hardware, mo, print_hardware_summary):
    import io as _io
    import sys as _sys

    hw = detect_hardware()
    _buf = _io.StringIO()
    _old = _sys.stdout
    _sys.stdout = _buf
    print_hardware_summary(hw)
    _sys.stdout = _old
    hw_text = _buf.getvalue()

    use_gpu = hw.recommended_backend == "gpu"
    gpu_badge = mo.callout(
        mo.md(f"```\n{hw_text}\n```"),
        kind="success" if use_gpu else "warn",
    )
    return gpu_badge, use_gpu


# ---------------------------------------------------------------------------
# Cell: Sidebar
# ---------------------------------------------------------------------------

@app.cell
def _sidebar(gpu_badge, mo):
    mo.sidebar([
        mo.md("## ⚙️ Settings & Info"),
        mo.accordion({
            "🖥️ Hardware Summary": gpu_badge,
            "ℹ️ About": mo.md(
                "**PyIsing Benchmark Dashboard**\n\n"
                "Runs all 9 Monte Carlo methods under identical conditions "
                "and compares observables side-by-side.\n\n"
                "Methods: Metropolis-Hastings, Glauber, Overrelaxation, "
                "Wolff, Swendsen-Wang, Invaded Cluster, Kinetic MC, "
                "Wang-Landau, Parallel Tempering."
            ),
        }),
    ])
    return


# ---------------------------------------------------------------------------
# Cell: Header
# ---------------------------------------------------------------------------

@app.cell
def _header(mo):
    mo.vstack([
        mo.md("# 🧲 PyIsing Method Benchmark Dashboard"),
        mo.md(
            "Compare all 9 Monte Carlo methods side-by-side.  "
            "Configure parameters below and click **▶ Run Benchmark** to start."
        ),
    ])
    return


# ---------------------------------------------------------------------------
# Cell: Controls
# ---------------------------------------------------------------------------

@app.cell
def _controls(mo):
    rows_input = mo.ui.number(
        start=4, stop=100, value=20, step=1,
        label="Lattice rows",
    )
    cols_input = mo.ui.number(
        start=4, stop=100, value=20, step=1,
        label="Lattice cols",
    )
    tf_input = mo.ui.number(
        start=1, stop=50, value=5, step=1,
        label="tf (simulation time)",
    )
    fps_input = mo.ui.number(
        start=1, stop=100, value=20, step=1,
        label="fps (frames/sec)",
    )
    beta_slider = mo.ui.slider(
        start=0.01, stop=2.0, step=0.01, value=0.6,
        label="β (inverse temperature)",
        show_value=True,
    )
    seed_input = mo.ui.number(
        start=0, stop=999999, value=42, step=1,
        label="Seed",
    )
    run_button = mo.ui.run_button(label="▶ Run Benchmark")
    return beta_slider, cols_input, fps_input, rows_input, run_button, seed_input, tf_input


@app.cell
def _controls_layout(beta_slider, cols_input, fps_input, mo, rows_input, run_button, seed_input, tf_input):
    mo.hstack([
        mo.vstack([rows_input, cols_input]),
        mo.vstack([tf_input, fps_input]),
        mo.vstack([beta_slider, seed_input]),
        mo.vstack([run_button]),
    ], justify="start", gap=2)
    return


# ---------------------------------------------------------------------------
# Cell: Reactive state
# ---------------------------------------------------------------------------

@app.cell
def _state(mo):
    results_state, set_results = mo.state([])
    status_state, set_status = mo.state("")
    return results_state, set_results, set_status, status_state


# ---------------------------------------------------------------------------
# Cell: Benchmark runner (triggered by button)
# ---------------------------------------------------------------------------

@app.cell
def _runner(
    BenchmarkConfig,
    beta_slider,
    cols_input,
    fps_input,
    mo,
    rows_input,
    run_benchmark,
    run_button,
    seed_input,
    set_results,
    set_status,
    tf_input,
    threading,
    use_gpu,
):
    mo.stop(not run_button.value)

    # Reset state for a fresh run
    set_results([])
    set_status("Starting benchmark...")

    _config = BenchmarkConfig(
        nrows=rows_input.value,
        ncols=cols_input.value,
        tf=tf_input.value,
        fps=fps_input.value,
        beta=beta_slider.value,
        seed=seed_input.value,
        use_gpu=use_gpu,
    )

    def _run():
        run_benchmark(
            config=_config,
            on_result=lambda r: set_results(lambda prev: prev + [r]),
            on_status=lambda m, s: set_status(f"Running: {m}..."),
            on_complete=lambda _: set_status("✅ Benchmark complete!"),
        )

    _thread = threading.Thread(target=_run, daemon=True)
    _thread.start()
    return


# ---------------------------------------------------------------------------
# Cell: Status display
# ---------------------------------------------------------------------------

@app.cell
def _status_display(mo, status_state):
    _status = status_state() if callable(status_state) else status_state
    if _status:
        mo.callout(mo.md(f"**Status:** {_status}"), kind="info")
    return


# ---------------------------------------------------------------------------
# Cell: Results table
# ---------------------------------------------------------------------------

@app.cell
def _results_table(mo, results_state, results_to_dataframe):
    _results = results_state() if callable(results_state) else results_state
    if not _results:
        mo.stop(True, mo.callout(
            mo.md("⏳ Waiting for benchmark results..."),
            kind="info",
        ))

    mo.vstack([
        mo.md("## 📊 Results Table"),
        mo.ui.table(results_to_dataframe(_results)),
    ])
    return


# ---------------------------------------------------------------------------
# Cell: Comparison bar charts
# ---------------------------------------------------------------------------

@app.cell
def _charts(METHOD_CATEGORIES, mo, np, plt, results_state):
    _results = results_state() if callable(results_state) else results_state
    if not _results:
        mo.stop(True)

    # Category → color mapping
    _cat_colors = {"local": "#3b82f6", "cluster": "#f97316", "advanced": "#22c55e"}

    _observables = [
        ("Energy ⟨E⟩", "energy"),
        ("|Magnetization| |⟨M⟩|", "abs_magnetization"),
        ("Specific Heat C", "specific_heat"),
        ("Susceptibility χ", "susceptibility"),
        ("Wall Time (s)", "wall_time"),
    ]

    _methods = [r.method for r in _results]
    _colors = [_cat_colors.get(METHOD_CATEGORIES.get(m, "local"), "#888") for m in _methods]
    _y_pos = np.arange(len(_methods))

    _figs = []
    for _title, _attr in _observables:
        _values = [getattr(r, _attr) for r in _results]

        _fig, _ax = plt.subplots(figsize=(10, max(3, 0.6 * len(_methods))))
        _ax.barh(_y_pos, _values, color=_colors, edgecolor="white", linewidth=0.5)
        _ax.set_yticks(_y_pos)
        _ax.set_yticklabels(_methods, fontsize=9)
        _ax.set_xlabel(_title)
        _ax.set_title(_title)
        _ax.invert_yaxis()
        _ax.grid(axis="x", alpha=0.3)
        _fig.tight_layout()
        _figs.append(_fig)

    # Legend figure
    _legend_items = [
        mo.md(
            f'<span style="color:{_cat_colors["local"]}">■</span> Local &nbsp; '
            f'<span style="color:{_cat_colors["cluster"]}">■</span> Cluster &nbsp; '
            f'<span style="color:{_cat_colors["advanced"]}">■</span> Advanced'
        )
    ]

    # Arrange charts in a 2-column grid with the 5th chart spanning full width
    mo.vstack([
        mo.md("## 📈 Comparison Charts"),
        mo.hstack(_legend_items, justify="center"),
        mo.hstack([mo.as_html(_figs[0]), mo.as_html(_figs[1])], gap=1),
        mo.hstack([mo.as_html(_figs[2]), mo.as_html(_figs[3])], gap=1),
        mo.as_html(_figs[4]),
    ])

    # Close all figures to free memory
    for _f in _figs:
        plt.close(_f)
    return


# ---------------------------------------------------------------------------
# Cell: Lattice snapshots
# ---------------------------------------------------------------------------

@app.cell
def _snapshots(mo, np, plt, results_state):
    _results = results_state() if callable(results_state) else results_state
    if not _results:
        mo.stop(True)

    # Filter out methods with zero lattices (Wang-Landau, Parallel Tempering)
    _snapshots = [
        (r.method, r.final_lattice)
        for r in _results
        if r.final_lattice is not None and np.any(r.final_lattice != 0)
    ]

    if not _snapshots:
        mo.stop(True, mo.callout(
            mo.md("No lattice snapshots available yet."),
            kind="info",
        ))

    _n = len(_snapshots)
    _ncols_grid = min(_n, 4)
    _nrows_grid = (_n + _ncols_grid - 1) // _ncols_grid

    _fig, _axes = plt.subplots(
        _nrows_grid, _ncols_grid,
        figsize=(4 * _ncols_grid, 4 * _nrows_grid),
        squeeze=False,
    )

    for _i, (_method, _lattice) in enumerate(_snapshots):
        _r = _i // _ncols_grid
        _c = _i % _ncols_grid
        _ax = _axes[_r][_c]
        _ax.imshow(_lattice, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        _ax.set_title(_method, fontsize=10)
        _ax.axis("off")

    # Hide unused subplots
    for _i in range(len(_snapshots), _nrows_grid * _ncols_grid):
        _r = _i // _ncols_grid
        _c = _i % _ncols_grid
        _axes[_r][_c].axis("off")

    _fig.suptitle("Final Lattice Snapshots", fontsize=14)
    _fig.tight_layout()

    mo.vstack([
        mo.md("## 🔬 Final Lattice Snapshots"),
        mo.md("*Wang-Landau and Parallel Tempering do not produce lattice snapshots.*"),
        mo.as_html(_fig),
    ])
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
