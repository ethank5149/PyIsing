"""
PyIsing example — demonstrates hardware detection and GPU-accelerated simulation.

Install the package first:
    pip install -e ".[gpu,video]"   # with GPU + video support
    pip install -e "."              # CPU only
"""

import numpy as np
from pyising import IsingModel, BETA_CRIT, detect_hardware, print_hardware_summary

# ---------------------------------------------------------------------------
# 1. Detect hardware and pick the best backend automatically
# ---------------------------------------------------------------------------
hw = detect_hardware()
print_hardware_summary(hw)

USE_GPU = hw.recommended_backend == 'gpu'
# For batched simulate(), use the recommended batch size as sample_size
SAMPLE_SIZE = hw.recommended_batch_size if USE_GPU else 30

# ---------------------------------------------------------------------------
# 2. Simulation parameters
# ---------------------------------------------------------------------------
N = 150          # number of beta points
NROWS = 10
NCOLS = 10
TF = 15
FPS = 15
METHOD = 'metropolis-hastings'   # uses checkerboard on GPU, sequential on CPU
                                  # switch to 'wolff' for Wolff cluster algorithm

beta_range = np.linspace(0.5 * BETA_CRIT, 10 * BETA_CRIT, N)
h_range = [0.0]  # np.linspace(-0.01, 0.01, N)

# ---------------------------------------------------------------------------
# 3. Run simulation
# ---------------------------------------------------------------------------
ising = IsingModel(
    tf=TF,
    fps=FPS,
    ncols=NCOLS,
    nrows=NROWS,
    sample_size=SAMPLE_SIZE,
    method=METHOD,
    use_gpu=USE_GPU,
)

results = ising.simulate(
    beta_range=beta_range,
    h_range=h_range,
    sample_size=SAMPLE_SIZE,
    method=METHOD,
)

# ---------------------------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------------------------
plot_path = ising.save_results(results)
print(f'\nResults plot saved to: {plot_path}')
print(results)
