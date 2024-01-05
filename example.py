import numpy as np
from pyising import IsingModel


N = 100
NROWS = 50
NCOLS = 50
TF = 10
FPS = 30
METHOD = 'wolff'

KB = 1
DELTA_BETA = 0.15
T_CRIT = 2 / np.log(1 + np.sqrt(2))

BETA_CRITICAL = 1 / (KB * T_CRIT)
SIM_ID = f'pyising_results_N={N}_{NCOLS}x{NROWS}_{TF * FPS}iters_{METHOD}_alg'

beta_range = np.linspace((1 - DELTA_BETA) * BETA_CRITICAL, (1 + DELTA_BETA) * BETA_CRITICAL, N)
h_range = [0,]  # np.linspace(-0.01, 0.01, N)

ising = IsingModel(
    tf=TF,
    fps=FPS,
    kB=KB,
    ncols=NCOLS,
    nrows=NROWS
    )

res = ising.simulate(beta_range=beta_range, h_range=h_range)
res_plot = res.plot(x='ThermodynamicBeta', y=['Energy', 'Magnetization', 'SpecificHeatCapacity', 'MagneticSusceptibility'], title='Ising Model Simulation Results', grid=True, legend=True)

print(res)
res_plot.figure.savefig(SIM_ID + '.pdf')
