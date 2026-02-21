"""
PyIsing — Marimo interactive app.

Run locally:
    marimo run app.py

Edit/develop:
    marimo edit app.py

The app has two tabs:
  1. Lattice Viewer  — watch the spin lattice evolve in real time at a chosen β and h
  2. Phase Diagram   — sweep β across a range and plot E, |M|, C vs T/Tc
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full", app_title="PyIsing")


@app.cell
def _imports():
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import marimo as mo
    from pyising import IsingModel, BETA_CRIT, detect_hardware, print_hardware_summary
    return BETA_CRIT, IsingModel, detect_hardware, io, mo, np, plt, print_hardware_summary


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
    return gpu_badge, hw, use_gpu


@app.cell
def _header(gpu_badge, mo):
    mo.vstack([
        mo.md("# 🧲 PyIsing — 2D Ising Model"),
        mo.md("Interactive simulation with GPU acceleration via CuPy."),
        gpu_badge,
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
        options=["metropolis-hastings", "wolff"],
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
    ncols_slider,
    nrows_slider,
    plt,
    use_gpu,
):
    """
    This cell re-runs whenever any slider changes.
    The IsingModel is created fresh each time (small lattices are fast).
    For large lattices, consider caching the model and only calling quench().
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
    _ising.quench(verbose=False)
    _ising.gather_data(verbose=False)

    # --- Lattice image (final frame) ---
    import numpy as _np
    _frame = _ising.at[-1]
    if hasattr(_frame, "get"):          # CuPy array
        _frame = _frame.get()
    else:
        _frame = _np.asarray(_frame)

    _fig_lat, _ax_lat = plt.subplots(figsize=(5, 5))
    _ax_lat.imshow(_frame, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    _ax_lat.set_title(
        f"Spin lattice (final frame)\n"
        f"β={beta_slider.value:.3f}  h={h_slider.value:.2f}  "
        f"T/Tc={1/(beta_slider.value * 2.269):.2f}"
    )
    _ax_lat.axis("off")
    _buf_lat = io.BytesIO()
    _fig_lat.savefig(_buf_lat, format="png", bbox_inches="tight", dpi=120)
    plt.close(_fig_lat)
    _buf_lat.seek(0)

    # --- Observables over time ---
    _fig_obs, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, layout="constrained")
    _t = _np.arange(_ising.frames)
    _ax1.plot(_t, _ising.energy, color="darkgreen", lw=1.5, label=r"$\langle E \rangle$")
    _ax1.set_ylabel("Energy per spin")
    _ax1.legend()
    _ax1.grid(alpha=0.3)
    _ax2.plot(_t, _ising.magnetization, color="darkorange", lw=1.5, label=r"$\langle M \rangle$")
    _ax2.set_ylabel("Magnetization per spin")
    _ax2.set_xlabel("Frame")
    _ax2.legend()
    _ax2.grid(alpha=0.3)
    _fig_obs.suptitle("Observables over time")
    _buf_obs = io.BytesIO()
    _fig_obs.savefig(_buf_obs, format="png", bbox_inches="tight", dpi=120)
    plt.close(_fig_obs)
    _buf_obs.seek(0)

    mo.hstack([
        mo.image(_buf_lat.read(), width=480),
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
        options=["metropolis-hastings", "wolff"],
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

    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(_fig)
    _buf.seek(0)

    mo.vstack([
        mo.image(_buf.read()),
        mo.ui.table(_results.round(4)),
    ])
    return


if __name__ == "__main__":
    app.run()
