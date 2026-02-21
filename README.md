# PyIsing

A Python 2D Ising model simulation package with optional GPU acceleration via CuPy.

## Features

- **Two Monte Carlo algorithms**: Metropolis-Hastings and Wolff cluster
- **GPU acceleration** (RTX 3090 / any CUDA device) via CuPy — automatic fallback to NumPy on CPU
- **Checkerboard Metropolis-Hastings**: fully parallel sublattice updates on GPU
- **Batched simulation**: all samples for a given (β, h) point run simultaneously on GPU
- **Hardware detection**: auto-detects CUDA devices, VRAM, and recommends backend + batch size
- **Video output**: renders spin lattice evolution to AVI (requires OpenCV)

## Installation

```bash
# CPU only
pip install -e .

# With GPU support (RTX 3090 / CUDA 12.x)
pip install -e ".[gpu]"

# With GPU + video output
pip install -e ".[all]"

# With Marimo interactive app
pip install -e ".[app]"

# Everything
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

> **Note**: Choose the CuPy variant matching your CUDA version.
> Check with `nvidia-smi` — look for `CUDA Version: XX.X`.
> - CUDA 11.x → `cupy-cuda11x`
> - CUDA 12.x → `cupy-cuda12x`
>
> Edit `pyproject.toml` `[project.optional-dependencies]` if needed.

## Quick Start

```python
import numpy as np
from pyising import IsingModel, BETA_CRIT, detect_hardware, print_hardware_summary

# Check what hardware is available
hw = detect_hardware()
print_hardware_summary(hw)

# Build model — auto-selects GPU if available
ising = IsingModel(
    nrows=50, ncols=50,
    tf=20, fps=20,
    use_gpu=(hw.recommended_backend == 'gpu'),
)

# Sweep β and h, average over samples
beta_range = np.linspace(0.5 * BETA_CRIT, 10 * BETA_CRIT, 150)
results = ising.simulate(beta_range=beta_range, h_range=[0.0], sample_size=30)

# Save results plot
ising.save_results(results)
print(results)
```

## Marimo Interactive App

```bash
# Install with app support
pip install -e ".[app,gpu]"

# Launch interactive app (opens browser automatically)
marimo run app.py

# Open in edit/development mode
marimo edit app.py
```

The app has two panels:

### 🔬 Lattice Viewer
- Sliders for **β**, **h**, **lattice size**, **algorithm**, **frames**
- Displays the final spin lattice frame as a colour image (blue = spin down, red = spin up)
- Plots **energy** and **magnetization** over time
- Re-runs automatically on every slider change (GPU quench happens in the background)

### 📈 Phase Diagram
- Configure β sweep range, samples per point, lattice size, algorithm
- Click **▶ Run Phase Diagram** to launch a full β sweep
- Plots **⟨E⟩**, **|⟨M⟩|**, **⟨C⟩** vs T/Tc with the critical temperature marked
- Results table shown below the plot

---

## Hardware Info CLI

```bash
pyising-info
```

Example output:
```
────────────────────────────────────────────────────────
  PyIsing Hardware Summary
────────────────────────────────────────────────────────
  Platform   : Linux-6.12-x86_64
  Python     : 3.11.8
  CPU cores  : 16
  CPU model  : AMD Ryzen 9 5950X
────────────────────────────────────────────────────────
  CuPy       : 13.0.0
  CUDA       : 12.3
  GPU [0]    : NVIDIA GeForce RTX 3090
               VRAM  22.4 GB free / 24.0 GB total
               Compute capability 8.6
────────────────────────────────────────────────────────
  Recommended backend    : GPU
  Recommended batch size : 256
────────────────────────────────────────────────────────
```

## Package Structure

```
PyIsing/
├── pyproject.toml
├── example.py
└── src/
    └── pyising/
        ├── __init__.py       ← public API
        ├── constants.py      ← KB, T_CRIT, BETA_CRIT
        ├── hardware.py       ← detect(), print_summary()
        └── model.py          ← IsingModel
```

## Physics

The 2D Ising Hamiltonian implemented is:

$$H = -J \sum_{\langle i,j \rangle} s_i s_j - \mu h \sum_i s_i$$

with periodic boundary conditions. The exact critical temperature (Onsager 1944) is:

$$T_c = \frac{2}{\ln(1 + \sqrt{2})} \approx 2.269$$

## License

MIT
