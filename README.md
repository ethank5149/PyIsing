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

## Available Methods

PyIsing implements a comprehensive suite of Monte Carlo algorithms for simulating the Ising model. Each method has distinct computational characteristics, GPU support, and optimal use cases.

### Methods Reference

| Method | String ID | Type | GPU Support |
|--------|-----------|------|-------------|
| Metropolis-Hastings | `'metropolis-hastings'` | Single-spin flip | Checkerboard ✓ |
| Wolff | `'wolff'` | Cluster | No |
| Glauber (Heat Bath) | `'glauber'` | Single-spin flip | Checkerboard ✓ |
| Overrelaxation | `'overrelaxation'` | Single-spin flip | Checkerboard ✓ |
| Swendsen-Wang | `'swendsen-wang'` | Cluster | No |
| Invaded Cluster | `'invaded-cluster'` | Cluster | No |
| Kinetic Monte Carlo | `'kinetic-mc'` | Event-driven | No |
| Wang-Landau | `'wang-landau'` | Advanced (flat histogram) | No |
| Parallel Tempering | `'parallel-tempering'` | Advanced (replica exchange) | Partial (batch updates) |

### Using Frame-by-Frame Methods

Frame-by-frame methods are used via [`quench()`](./src/pyising/model.py) or [`simulate()`](./src/pyising/model.py) to evolve the spin lattice step-by-step.

#### Metropolis-Hastings (Sequential & Checkerboard)

Classic single-spin-flip algorithm. On GPU, uses checkerboard sublattice parallelization for fully parallel updates. Ideal for equilibration and general-purpose phase diagram studies.

```python
# CPU single-spin flip or GPU checkerboard variant
ising = IsingModel(nrows=50, ncols=50, method='metropolis-hastings', use_gpu=True)
ising.quench(beta=0.6, n_steps=1000)
```

#### Wolff (Cluster Algorithm)

Single-cluster flipping with automatic cluster size tuning. Fast critical slowing-down reduction near phase transitions. CPU-only.

```python
ising = IsingModel(nrows=50, ncols=50, method='wolff', use_gpu=False)
results = ising.simulate(beta_range=np.linspace(0.4, 1.0, 50), sample_size=20)
```

#### Glauber (Heat Bath)

Heat bath dynamics with local field probability $P(+1) = 1/(1+\exp(-\beta h_{\text{local}}))$. Smooth acceptance probabilities reduce correlations. GPU checkerboard variant available.

```python
ising = IsingModel(nrows=100, ncols=100, method='glauber', use_gpu=True)
ising.quench(beta=1.2, n_steps=500)
```

#### Overrelaxation

Microcanonical energy-preserving algorithm that flips only sites where the local field is zero. Useful for studying dynamics without thermal noise. GPU checkerboard variant available.

```python
ising = IsingModel(nrows=50, ncols=50, method='overrelaxation')
results = ising.simulate(beta_range=[0.8], h_range=[0.0], sample_size=10)
```

#### Swendsen-Wang (Cluster Algorithm)

Identifies all clusters simultaneously via union-find and flips each with probability 1/2. Excellent critical scaling behavior. CPU-only.

```python
ising = IsingModel(nrows=64, ncols=64, method='swendsen-wang')
ising.quench(beta=0.6, n_steps=100)  # Fast convergence near Tc
```

#### Invaded Cluster (Self-Tuning)

Self-tuning cluster algorithm that grows clusters by sorted bond weights until percolation. Adapts cluster size automatically. CPU-only.

```python
ising = IsingModel(nrows=50, ncols=50, method='invaded-cluster')
results = ising.simulate(beta_range=[0.44, 0.50], sample_size=15)
```

#### Kinetic Monte Carlo (Event-Driven)

Event-driven continuous-time algorithm with Glauber rates. Selects sites proportional to flip rate rather than uniformly. Provides true continuous-time dynamics. CPU-only.

```python
ising = IsingModel(nrows=50, ncols=50, method='kinetic-mc')
ising.quench(beta=0.8, n_steps=200)
```

### Standalone Advanced Methods

Advanced methods are invoked directly via dedicated methods and return specialized results.

#### Wang-Landau Density of States

Flat histogram method that estimates the density of states $g(E)$ across the entire energy range. Useful for any-temperature observables without re-running simulations.

```python
ising = IsingModel(nrows=32, ncols=32)

# Perform Wang-Landau sampling
wl_result = ising.simulate_wang_landau(
    flatness=0.8,      # Flatness criterion (0.0-1.0)
    f_min=1e-8,        # Final histogram scaling factor
    n_sweeps=50        # Sweeps per iteration
)

# Compute observables at arbitrary temperatures
beta_range = np.linspace(0.1, 2.0, 100)
observables_df = ising.wang_landau_observables(wl_result, beta_range=beta_range)
print(observables_df)
```

The returned DataFrame contains columns: `beta`, `energy`, `magnetization`, `heat_capacity`, `susceptibility`.

#### Parallel Tempering (Replica Exchange)

Multiple replicas at different temperatures exchange configurations via Metropolis criterion. Overcomes free energy barriers and explores phase space efficiently.

```python
ising = IsingModel(nrows=50, ncols=50)

# Run parallel tempering across temperature range
results_df = ising.simulate_parallel_tempering(
    beta_range=np.linspace(0.2, 1.0, 20),  # Replicas at these β values
    h=0.0,                                   # External field
    n_sweeps=200,                            # Sweeps per swap interval
    swap_interval=10                         # Swap every N sweeps
)

print(results_df)
```

The returned DataFrame contains ensemble-averaged observables: `beta`, `energy`, `magnetization`, `heat_capacity`, `susceptibility`.

### Choosing a Method

| Goal | Recommended Methods |
|------|---------------------|
| Fast phase diagram mapping | Metropolis-Hastings (GPU) or Glauber (GPU) |
| Near critical temperature | Wolff, Swendsen-Wang, or Invaded Cluster |
| Density of states / any-T observables | Wang-Landau |
| Free energy barriers | Parallel Tempering |
| Dynamics with no thermal noise | Overrelaxation |
| True continuous-time dynamics | Kinetic Monte Carlo |

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
