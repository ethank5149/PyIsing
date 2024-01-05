import numpy as np
from pyising import IsingModel


KB = 1
T_CRIT = 2 / np.log(1 + np.sqrt(2))
BETA_CRIT = 1 / (KB * T_CRIT)

N = 150
NROWS = 15
NCOLS = 15
TF = 15
FPS = 15
SAMPLE_SIZE = 30
METHOD = 'wolff'

beta_range = np.linspace(0.5 * BETA_CRIT, 10 * BETA_CRIT, N)
h_range = [0,]  # np.linspace(-0.01, 0.01, N)

ising = IsingModel(
    tf=TF,
    fps=FPS,
    ncols=NCOLS,
    nrows=NROWS,
    sample_size=SAMPLE_SIZE,
    method=METHOD
    )

results = ising.simulate(beta_range=beta_range, h_range=h_range)
ising.save_results(results)
print(results)
