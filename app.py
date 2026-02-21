"""
PyIsing — Marimo interactive app.

Run locally:
    marimo run app.py

Edit/develop:
    marimo edit app.py

The app has three sections:
  1. Lattice Viewer  — watch the spin lattice evolve as an animation at a chosen β and h
  2. Phase Diagram   — sweep β across a range and plot E, |M|, C vs T/Tc
  3. Method Benchmark — compare all 9 Monte Carlo methods side-by-side

Hardware info is available in the sidebar under Settings & Info.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full", app_title="PyIsing")


@app.cell
def _imports():
    import io
    import tempfile
    import threading
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_animation
    import marimo as mo
    from pyising import IsingModel, BETA_CRIT, detect_hardware, print_hardware_summary
    from pyising.benchmark import (
        ALL_METHODS,
        METHOD_CATEGORIES,
        BenchmarkConfig,
        BenchmarkResult,
        results_to_dataframe,
        run_benchmark,
    )
    return (
        ALL_METHODS, BETA_CRIT, BenchmarkConfig, BenchmarkResult,
        IsingModel, METHOD_CATEGORIES, detect_hardware, io, mo,
        mpl_animation, np, plt, print_hardware_summary,
        results_to_dataframe, run_benchmark, tempfile, threading,
    )


@app.cell
def _hardware(detect_hardware, mo, print_hardware_summary):
    """Detect hardware once at startup."""
    import io as _io
    hw = detect_hardware()

    # Capture print_summary output as a string for display
    buf = _io.StringIO()
    import sys
    _old = sys.stdout
    sys.stdout = buf
    print_hardware_summary(hw)
    sys.stdout = _old
    hw_text = buf.getvalue()

    use_gpu = hw.recommended_backend == "gpu"
    gpu_badge = mo.callout(
        mo.md(f"```\n{hw_text}\n```"),
        kind="success" if use_gpu else "warn",
    )
    return gpu_badge, hw, hw_text, use_gpu


# ---------------------------------------------------------------------------
# Sidebar — Settings & Info (hardware summary moved here)
# ---------------------------------------------------------------------------

@app.cell
def _sidebar(gpu_badge, mo):
    mo.sidebar([
        mo.md("## ⚙️ Settings & Info"),
        mo.accordion({
            "🖥️ Hardware Summary": gpu_badge,
            "ℹ️ About": mo.md(
                "**PyIsing** — 2D Ising Model Simulation\n\n"
                "Interactive simulation with GPU acceleration via CuPy.\n\n"
                "**Local methods:** Metropolis-Hastings, Glauber, Overrelaxation, Kinetic MC\n\n"
                "**Cluster methods:** Wolff, Swendsen-Wang, Invaded Cluster\n\n"
                "**Advanced methods:** Wang-Landau, Parallel Tempering\n"
            ),
        }),
    ])
    return


@app.cell
def _header(mo):
    mo.vstack([
        mo.md("# 🧲 PyIsing — 2D Ising Model"),
        mo.md("Interactive simulation with GPU acceleration via CuPy."),
    ])
    return


# ---------------------------------------------------------------------------
# Shared parameter controls
# ---------------------------------------------------------------------------

@app.cell
def _controls(BETA_CRIT, mo, np):
    beta_slider = mo.ui.slider(
        start=round(0.1 * BETA_CRIT, 4),
        stop=round(5.0 * BETA_CRIT, 4),
        step=round(0.01 * BETA_CRIT, 4),
        value=round(BETA_CRIT, 4),
        label="β (inverse temperature)",
        show_value=True,
    )
    h_slider = mo.ui.slider(
        start=-1.0, stop=1.0, step=0.01, value=0.0,
        label="h (external field)",
        show_value=True,
    )
    nrows_slider = mo.ui.slider(
        start=10, stop=200, step=10, value=50,
        label="Lattice rows",
        show_value=True,
    )
    ncols_slider = mo.ui.slider(
        start=10, stop=200, step=10, value=50,
        label="Lattice cols",
        show_value=True,
    )
    method_dropdown = mo.ui.dropdown(
        options=[
            "metropolis-hastings", "glauber", "overrelaxation",
            "wolff", "swendsen-wang", "invaded-cluster", "kinetic-mc",
        ],
        value="metropolis-hastings",
        label="Algorithm",
    )
    frames_slider = mo.ui.slider(
        start=10, stop=500, step=10, value=100,
        label="Frames (iterations)",
        show_value=True,
    )
    return (
        beta_slider,
        frames_slider,
        h_slider,
        method_dropdown,
        ncols_slider,
        nrows_slider,
    )


# ---------------------------------------------------------------------------
# Tab 1 — Lattice Viewer
# ---------------------------------------------------------------------------

@app.cell
def _lattice_controls(beta_slider, frames_slider, h_slider, method_dropdown, mo, ncols_slider, nrows_slider):
    mo.md("## 🔬 Lattice Viewer")
    return


@app.cell
def _lattice_param_row(beta_slider, frames_slider, h_slider, method_dropdown, mo, ncols_slider, nrows_slider):
    mo.hstack([
        mo.vstack([beta_slider, h_slider]),
        mo.vstack([nrows_slider, ncols_slider]),
        mo.vstack([method_dropdown, frames_slider]),
    ], justify="start", gap=2)
    return


@app.cell
def _lattice_view(
    IsingModel,
    beta_slider,
    frames_slider,
    h_slider,
    io,
    method_dropdown,
    mo,
    mpl_animation,
    ncols_slider,
    nrows_slider,
    np,
    plt,
    use_gpu,
):
    """
    This cell re-runs whenever any slider changes.
    The IsingModel is created fresh each time (small lattices are fast).
    Produces an animated MP4 of the lattice evolution plus observable plots.
    """
    _nrows = nrows_slider.value
    _ncols = ncols_slider.value
    _fps = 10
    _tf = max(1, frames_slider.value // _fps)

    _ising = IsingModel(
        nrows=_nrows,
        ncols=_ncols,
        tf=_tf,
        fps=_fps,
        beta=beta_slider.value,
        h=h_slider.value,
        method=method_dropdown.value,
        use_gpu=use_gpu,
    )

    with mo.status.spinner(title="Running lattice simulation…"):
        _ising.quench(verbose=False)
        _ising.gather_data(verbose=False)

    # --- Collect all frames as numpy arrays ---
    _all_frames = []
    for _t in range(_ising.frames):
        _frame = _ising.at[_t]
        if hasattr(_frame, "get"):          # CuPy array
            _frame = _frame.get()
        else:
            _frame = np.asarray(_frame)
        _all_frames.append(_frame)

    # --- Animated lattice video ---
    _fig_anim, _ax_anim = plt.subplots(figsize=(5, 5))
    _im = _ax_anim.imshow(
        _all_frames[0], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"
    )
    _ax_anim.axis("off")
    _title = _ax_anim.set_title(
        f"Frame 0/{_ising.frames - 1}  |  "
        f"β={beta_slider.value:.3f}  h={h_slider.value:.2f}  "
        f"T/Tc={1/(beta_slider.value * 2.269):.2f}"
    )

    def _update_frame(_t_idx):
        _im.set_data(_all_frames[_t_idx])
        _title.set_text(
            f"Frame {_t_idx}/{_ising.frames - 1}  |  "
            f"β={beta_slider.value:.3f}  h={h_slider.value:.2f}  "
            f"T/Tc={1/(beta_slider.value * 2.269):.2f}"
        )
        return [_im, _title]

    _anim = mpl_animation.FuncAnimation(
        _fig_anim, _update_frame,
        frames=len(_all_frames),
        interval=1000 // _fps,
        blit=True,
    )

    # Try MP4 first (requires ffmpeg), fall back to GIF (pillow).
    # Animation.save() infers format from filename extension, so use a temp file.
    import tempfile as _tempfile
    import os as _os
    _video_buf = io.BytesIO()
    _video_mimetype = "video/mp4"
    try:
        _tmp_fd, _tmp_path = _tempfile.mkstemp(suffix=".mp4")
        _os.close(_tmp_fd)
        _anim.save(_tmp_path, writer="ffmpeg", fps=_fps, dpi=100)
        with open(_tmp_path, "rb") as _f:
            _video_buf.write(_f.read())
        _os.unlink(_tmp_path)
    except Exception:
        try:
            _os.unlink(_tmp_path)
        except OSError:
            pass
        _video_buf = io.BytesIO()
        _video_mimetype = "image/gif"
        _tmp_fd, _tmp_path = _tempfile.mkstemp(suffix=".gif")
        _os.close(_tmp_fd)
        _anim.save(_tmp_path, writer="pillow", fps=_fps, dpi=100)
        with open(_tmp_path, "rb") as _f:
            _video_buf.write(_f.read())
        _os.unlink(_tmp_path)
    plt.close(_fig_anim)
    _video_buf.seek(0)

    # --- Observables over time ---
    _fig_obs, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, layout="constrained")
    _t_arr = np.arange(_ising.frames)
    _ax1.plot(_t_arr, _ising.energy, color="darkgreen", lw=1.5, label=r"$\langle E \rangle$")
    _ax1.set_ylabel("Energy per spin")
    _ax1.legend()
    _ax1.grid(alpha=0.3)
    _ax2.plot(_t_arr, _ising.magnetization, color="darkorange", lw=1.5, label=r"$\langle M \rangle$")
    _ax2.set_ylabel("Magnetization per spin")
    _ax2.set_xlabel("Frame")
    _ax2.legend()
    _ax2.grid(alpha=0.3)
    _fig_obs.suptitle("Observables over time")
    _buf_obs = io.BytesIO()
    _fig_obs.savefig(_buf_obs, format="png", bbox_inches="tight", dpi=120)
    plt.close(_fig_obs)
    _buf_obs.seek(0)

    # --- Display animation + observables side by side ---
    if _video_mimetype == "video/mp4":
        _lattice_display = mo.video(
            _video_buf.read(),
            controls=True,
            autoplay=True,
            loop=True,
            width=480,
        )
    else:
        # GIF fallback — display as image
        _lattice_display = mo.image(_video_buf.read(), width=480)

    mo.hstack([
        _lattice_display,
        mo.image(_buf_obs.read(), width=600),
    ], justify="start", gap=2)
    return


# ---------------------------------------------------------------------------
# Tab 2 — Phase Diagram
# ---------------------------------------------------------------------------

@app.cell
def _phase_header(mo):
    mo.md("---\n## 📈 Phase Diagram Sweep")
    return


@app.cell
def _phase_controls(BETA_CRIT, mo, np):
    n_beta_slider = mo.ui.slider(
        start=20, stop=300, step=10, value=80,
        label="Number of β points",
        show_value=True,
    )
    sample_size_slider = mo.ui.slider(
        start=5, stop=100, step=5, value=20,
        label="Samples per point",
        show_value=True,
    )
    phase_nrows = mo.ui.slider(start=10, stop=100, step=10, value=20, label="Lattice rows", show_value=True)
    phase_ncols = mo.ui.slider(start=10, stop=100, step=10, value=20, label="Lattice cols", show_value=True)
    phase_method = mo.ui.dropdown(
        options=[
            "metropolis-hastings", "glauber", "overrelaxation",
            "wolff", "swendsen-wang", "invaded-cluster", "kinetic-mc",
        ],
        value="metropolis-hastings",
        label="Algorithm",
    )
    run_button = mo.ui.run_button(label="▶ Run Phase Diagram")
    return (
        n_beta_slider,
        phase_method,
        phase_ncols,
        phase_nrows,
        run_button,
        sample_size_slider,
    )


@app.cell
def _phase_param_row(mo, n_beta_slider, phase_method, phase_ncols, phase_nrows, run_button, sample_size_slider):
    mo.hstack([
        mo.vstack([n_beta_slider, sample_size_slider]),
        mo.vstack([phase_nrows, phase_ncols]),
        mo.vstack([phase_method, run_button]),
    ], justify="start", gap=2)
    return


@app.cell
def _phase_diagram(
    BETA_CRIT,
    IsingModel,
    io,
    mo,
    n_beta_slider,
    np,
    phase_method,
    phase_ncols,
    phase_nrows,
    plt,
    run_button,
    sample_size_slider,
    use_gpu,
):
    mo.stop(not run_button.value, mo.callout(
        mo.md("Configure parameters above and click **▶ Run Phase Diagram** to start."),
        kind="info",
    ))

    _beta_range = np.linspace(0.5 * BETA_CRIT, 10 * BETA_CRIT, n_beta_slider.value)

    _ising_phase = IsingModel(
        nrows=phase_nrows.value,
        ncols=phase_ncols.value,
        tf=10,
        fps=10,
        method=phase_method.value,
        use_gpu=use_gpu,
    )

    with mo.status.spinner(title="Running phase diagram simulation…"):
        _results = _ising_phase.simulate(
            beta_range=_beta_range,
            h_range=[0.0],
            sample_size=sample_size_slider.value,
            method=phase_method.value,
        )

    _results["NetMagnetization"] = _results["Magnetization"].abs()
    _results["ScaledTemperature"] = 1.0 / (BETA_CRIT * _results["ThermodynamicBeta"])

    _fig, (_a1, _a2, _a3) = plt.subplots(3, 1, sharex=True, layout="constrained", figsize=(10, 9))
    _fig.suptitle(
        f"Phase Diagram — {phase_nrows.value}×{phase_ncols.value} lattice, "
        f"{sample_size_slider.value} samples/point, {phase_method.value}"
    )
    _fig.supxlabel(r"$T / T_c$")
    _a1.scatter(_results["ScaledTemperature"], _results["Energy"], color="darkgreen", s=8, label=r"$\langle E \rangle$")
    _a1.axvline(1.0, color="gray", ls="--", lw=0.8, label=r"$T_c$")
    _a1.set_ylabel(r"Energy $\langle E \rangle$")
    _a1.legend(); _a1.grid(alpha=0.3)
    _a2.scatter(_results["ScaledTemperature"], _results["NetMagnetization"], color="darkorange", s=8, label=r"$|\langle M \rangle|$")
    _a2.axvline(1.0, color="gray", ls="--", lw=0.8)
    _a2.set_ylabel(r"Net Magnetization $|\langle M \rangle|$")
    _a2.legend(); _a2.grid(alpha=0.3)
    _a3.scatter(_results["ScaledTemperature"], _results["SpecificHeatCapacity"], color="darkred", s=8, label=r"$\langle C \rangle$")
    _a3.axvline(1.0, color="gray", ls="--", lw=0.8)
    _a3.set_ylabel(r"Specific Heat $\langle C \rangle$")
    _a3.legend(); _a3.grid(alpha=0.3)

    # --- Save plot as SVG first (vector), then PNG (raster) from same figure ---
    _buf_svg = io.BytesIO()
    _fig.savefig(_buf_svg, format="svg", bbox_inches="tight")
    _buf_svg.seek(0)
    _svg_bytes = _buf_svg.read()

    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(_fig)
    _buf.seek(0)
    _plot_bytes = _buf.read()

    # --- Prepare CSV download ---
    _csv_bytes = _results.to_csv(index=False).encode("utf-8")

    # --- Build download buttons ---
    _dl_png = mo.download(
        _plot_bytes,
        filename="pyising_phase_diagram.png",
        mimetype="image/png",
        label="📥 Plot (PNG)",
    )
    _dl_svg = mo.download(
        _svg_bytes,
        filename="pyising_phase_diagram.svg",
        mimetype="image/svg+xml",
        label="📥 Plot (SVG)",
    )
    _dl_csv = mo.download(
        _csv_bytes,
        filename="pyising_phase_diagram.csv",
        mimetype="text/csv",
        label="📥 Data (CSV)",
    )

    mo.vstack([
        mo.image(_plot_bytes),
        mo.hstack([_dl_png, _dl_svg, _dl_csv], justify="start", gap=1),
        mo.ui.table(_results.round(4)),
    ])
    return


# ---------------------------------------------------------------------------
# Tab 3 — Method Benchmark
# ---------------------------------------------------------------------------

@app.cell
def _bench_header(mo):
    mo.md("---\n## 🏁 Method Benchmark\n\n"
           "Compare all 9 Monte Carlo methods side-by-side under identical conditions.  "
           "Results stream in as each method completes.")
    return


@app.cell
def _bench_controls(mo):
    bench_rows = mo.ui.number(start=4, stop=100, value=20, step=1, label="Lattice rows")
    bench_cols = mo.ui.number(start=4, stop=100, value=20, step=1, label="Lattice cols")
    bench_tf = mo.ui.number(start=1, stop=50, value=5, step=1, label="tf (sim time)")
    bench_fps = mo.ui.number(start=1, stop=100, value=20, step=1, label="fps")
    bench_beta = mo.ui.slider(start=0.01, stop=2.0, step=0.01, value=0.6,
                               label="β (inverse temperature)", show_value=True)
    bench_seed = mo.ui.number(start=0, stop=999999, value=42, step=1, label="Seed")
    bench_run = mo.ui.run_button(label="▶ Run Benchmark")
    return bench_beta, bench_cols, bench_fps, bench_rows, bench_run, bench_seed, bench_tf


@app.cell
def _bench_controls_layout(bench_beta, bench_cols, bench_fps, bench_rows, bench_run, bench_seed, bench_tf, mo):
    mo.hstack([
        mo.vstack([bench_rows, bench_cols]),
        mo.vstack([bench_tf, bench_fps]),
        mo.vstack([bench_beta, bench_seed]),
        mo.vstack([bench_run]),
    ], justify="start", gap=2)
    return


@app.cell
def _bench_state(mo):
    bench_results_state, bench_set_results = mo.state([])
    bench_status_state, bench_set_status = mo.state("")
    return bench_results_state, bench_set_results, bench_set_status, bench_status_state


@app.cell
def _bench_runner(
    BenchmarkConfig,
    bench_beta,
    bench_cols,
    bench_fps,
    bench_rows,
    bench_run,
    bench_seed,
    bench_set_results,
    bench_set_status,
    bench_tf,
    mo,
    run_benchmark,
    threading,
    use_gpu,
):
    mo.stop(not bench_run.value)

    bench_set_results([])
    bench_set_status("Starting benchmark...")

    _config = BenchmarkConfig(
        nrows=bench_rows.value,
        ncols=bench_cols.value,
        tf=bench_tf.value,
        fps=bench_fps.value,
        beta=bench_beta.value,
        seed=bench_seed.value,
        use_gpu=use_gpu,
    )

    def _run():
        run_benchmark(
            config=_config,
            on_result=lambda r: bench_set_results(lambda prev: prev + [r]),
            on_status=lambda m, s: bench_set_status(f"Running: {m}..."),
            on_complete=lambda _: bench_set_status("✅ Benchmark complete!"),
        )

    _thread = threading.Thread(target=_run, daemon=True)
    _thread.start()
    return


@app.cell
def _bench_status(bench_status_state, mo):
    _status = bench_status_state() if callable(bench_status_state) else bench_status_state
    if _status:
        mo.callout(mo.md(f"**Status:** {_status}"), kind="info")
    return


@app.cell
def _bench_table(bench_results_state, mo, results_to_dataframe):
    _results = bench_results_state() if callable(bench_results_state) else bench_results_state
    if not _results:
        mo.stop(True, mo.callout(
            mo.md("⏳ Configure parameters above and click **▶ Run Benchmark** to start."),
            kind="info",
        ))

    mo.vstack([
        mo.md("### 📊 Results Table"),
        mo.ui.table(results_to_dataframe(_results)),
    ])
    return


@app.cell
def _bench_charts(METHOD_CATEGORIES, bench_results_state, mo, np, plt):
    _results = bench_results_state() if callable(bench_results_state) else bench_results_state
    if not _results:
        mo.stop(True)

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

    _legend = mo.md(
        f'<span style="color:{_cat_colors["local"]}">■</span> Local &nbsp; '
        f'<span style="color:{_cat_colors["cluster"]}">■</span> Cluster &nbsp; '
        f'<span style="color:{_cat_colors["advanced"]}">■</span> Advanced'
    )

    mo.vstack([
        mo.md("### 📈 Comparison Charts"),
        mo.hstack([_legend], justify="center"),
        mo.hstack([mo.as_html(_figs[0]), mo.as_html(_figs[1])], gap=1),
        mo.hstack([mo.as_html(_figs[2]), mo.as_html(_figs[3])], gap=1),
        mo.as_html(_figs[4]),
    ])

    for _f in _figs:
        plt.close(_f)
    return


@app.cell
def _bench_snapshots(bench_results_state, mo, np, plt):
    _results = bench_results_state() if callable(bench_results_state) else bench_results_state
    if not _results:
        mo.stop(True)

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

    for _i in range(len(_snapshots), _nrows_grid * _ncols_grid):
        _r = _i // _ncols_grid
        _c = _i % _ncols_grid
        _axes[_r][_c].axis("off")

    _fig.suptitle("Final Lattice Snapshots", fontsize=14)
    _fig.tight_layout()

    mo.vstack([
        mo.md("### 🔬 Final Lattice Snapshots"),
        mo.md("*Wang-Landau and Parallel Tempering do not produce lattice snapshots.*"),
        mo.as_html(_fig),
    ])
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
