"""
PyIsing — 2D Ising Model simulation with optional GPU acceleration via CuPy.

GPU features:
  - Step 1: CuPy drop-in for all array operations (metric, E, M, gather_data)
  - Step 2: Checkerboard Metropolis-Hastings — fully parallel sublattice updates
  - Step 3: Batched simulate() — all samples run simultaneously on GPU

Falls back to NumPy automatically if CuPy / CUDA is unavailable.
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

# ---------------------------------------------------------------------------
# GPU / CPU backend selection (Step 1)
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    _GPU_AVAILABLE = True
except ImportError:
    cp = None
    _GPU_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


KB = 1
T_CRIT = 2 / np.log(1 + np.sqrt(2))
BETA_CRIT = 1 / (KB * T_CRIT)


def _xp(use_gpu: bool):
    """Return cupy if GPU is requested and available, else numpy."""
    return cp if (use_gpu and _GPU_AVAILABLE) else np


def _to_numpy(arr):
    """Move an array to CPU numpy regardless of its origin."""
    if _GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Checkerboard mask helpers (Step 2)
# ---------------------------------------------------------------------------

def _checkerboard_masks(nrows: int, ncols: int, xp):
    """Return (black_mask, white_mask) boolean arrays for checkerboard updates."""
    rows = xp.arange(nrows, dtype=xp.int32)[:, None]
    cols = xp.arange(ncols, dtype=xp.int32)[None, :]
    black = (rows + cols) % 2 == 0
    return black, ~black


# ---------------------------------------------------------------------------
# IsingModel
# ---------------------------------------------------------------------------

class IsingModel:
    def __init__(
            self,
            tf=10,
            fps=10,
            ncols=10,
            nrows=10,
            J=1,
            mu=1,
            beta=1.5 * BETA_CRIT,
            h=0,
            method='metropolis-hastings',
            sample_size=30,
            h_range=[0,],
            beta_range=np.linspace(0.5 * BETA_CRIT, 10 * BETA_CRIT, 100),
            use_gpu=True,
            ):

        self.use_gpu = use_gpu and _GPU_AVAILABLE
        self.xp = _xp(self.use_gpu)

        self.rng = np.random.default_rng()

        self.nrows, self.ncols = nrows, ncols
        self.size = nrows * ncols
        self.J, self.mu = J, mu
        self.beta, self.h = beta, h
        self.T = 1 / (KB * self.beta)

        self.tf, self.fps = tf, fps
        self.frames = self.tf * self.fps
        self.sample_size, self.method = sample_size, method
        self.beta_range = np.asarray(beta_range)
        self.h_range = np.asarray(h_range)

        # Observables stored on CPU (small arrays)
        self.specific_heat_capacity = np.zeros(self.frames)
        self.magnetic_susceptibility = np.zeros(self.frames)
        self.energy = np.zeros(self.frames)
        self.magnetization = np.zeros(self.frames)
        self.energy_sqr = np.zeros(self.frames)
        self.magnetization_sqr = np.zeros(self.frames)

        # Spin lattice — stored on GPU if available
        self._init_lattice()

        # Checkerboard masks (Step 2)
        self._black_mask, self._white_mask = _checkerboard_masks(nrows, ncols, self.xp)

    # ------------------------------------------------------------------
    # Lattice initialisation
    # ------------------------------------------------------------------

    def _init_lattice(self):
        raw = 2 * self.rng.integers(low=0, high=2, size=(self.frames, self.nrows, self.ncols)).astype(np.int8) - 1
        if self.use_gpu:
            self.at = cp.asarray(raw)
        else:
            self.at = raw

    def __repr__(self):
        backend = 'gpu' if self.use_gpu else 'cpu'
        return (f'{self.ncols}x{self.nrows}_{self.sample_size}samples_'
                f'{self.tf * self.fps}iters_nbeta={len(self.beta_range)}_'
                f'nh={len(self.h_range)}_{self.method}_alg_{backend}')

    def __str__(self):
        return (f'{self.ncols}x{self.nrows}_tf={self.tf}_fps={self.fps}_'
                f'b={self.beta:0.4f}_h={self.h:0.4f}_{self.method}_alg')

    def reset(self):
        """Re-randomise frame 0 of the spin lattice."""
        raw = (2 * self.rng.integers(low=0, high=2, size=(self.nrows, self.ncols)).astype(np.int8) - 1)
        if self.use_gpu:
            self.at[0, :, :] = cp.asarray(raw)
        else:
            self.at[0, :, :] = raw

    # ------------------------------------------------------------------
    # Neighbour sums
    # ------------------------------------------------------------------

    def metric(self, t):
        """Vectorised sum of 4 nearest neighbours for the full lattice at frame t."""
        xp = self.xp
        s = self.at[t]
        return xp.roll(s, -1, 0) + xp.roll(s, 1, 0) + xp.roll(s, -1, 1) + xp.roll(s, 1, 1)

    def metric_at(self, t, i, j):
        """Scalar neighbour sum at site (i, j) — used only by Wolff."""
        return (self.at[t, (i + 1) % self.nrows, j] +
                self.at[t, (i - 1) % self.nrows, j] +
                self.at[t, i, (j + 1) % self.ncols] +
                self.at[t, i, (j - 1) % self.ncols])

    def neighbors_at(self, t, i, j):
        return [
            ((i + 1) % self.nrows, j),
            ((i - 1) % self.nrows, j),
            (i, (j + 1) % self.ncols),
            (i, (j - 1) % self.ncols),
        ]

    # ------------------------------------------------------------------
    # Thermodynamic observables (all vectorised, GPU-aware)
    # ------------------------------------------------------------------

    def E(self, t):
        """Mean energy per spin at frame t."""
        xp = self.xp
        s = self.at[t]
        nb = self.metric(t)
        val = -0.5 * self.J * xp.sum(s * nb) - self.mu * self.h * xp.sum(s)
        return float(val) / self.size

    def E_sqr(self, t):
        """Mean squared per-site energy at frame t."""
        xp = self.xp
        s = self.at[t]
        nb = self.metric(t)
        per_site = -0.5 * self.J * s * nb - self.mu * self.h * s
        return float(xp.sum(per_site ** 2)) / self.size

    def M(self, t):
        """Mean magnetisation per spin at frame t."""
        return float(self.xp.sum(self.at[t])) / self.size

    def M_sqr(self, t):
        """Mean squared magnetisation per spin at frame t."""
        return float(self.xp.sum(self.at[t].astype(self.xp.float32) ** 2)) / self.size

    def C(self, t):
        """Specific heat capacity at frame t (requires gather_data to have run)."""
        return self.beta * (self.energy_sqr[t] - self.energy[t] ** 2) / self.T

    def X(self, t):
        """Magnetic susceptibility at frame t (requires gather_data to have run)."""
        return self.beta * (self.magnetization_sqr[t] - self.magnetization[t] ** 2)

    # ------------------------------------------------------------------
    # Monte Carlo update steps
    # ------------------------------------------------------------------

    def update_metropolis_hastings(self, t):
        """
        Sequential (site-by-site) Metropolis-Hastings update.
        Kept for reference / Wolff fallback; use update_checkerboard for GPU speed.
        """
        i_s = self.rng.integers(low=0, high=self.nrows, size=self.size)
        j_s = self.rng.integers(low=0, high=self.ncols, size=self.size)
        self.at[t + 1, :, :] = self.at[t, :, :]
        for obv in range(self.size):
            i, j = i_s[obv], j_s[obv]
            nb = int(self.metric_at(t + 1, i, j))
            delta_e = -0.5 * (self.J * nb + self.mu * self.h) * int(self.at[t + 1, i, j])
            p = 1.0 if delta_e < 0.0 else float(np.exp(-delta_e * self.beta))
            if self.rng.random() < p:
                self.at[t + 1, i, j] = -self.at[t + 1, i, j]

    def update_checkerboard(self, t):
        """
        Step 2 — Fully parallel checkerboard Metropolis-Hastings.

        Black and white sublattices are updated alternately; spins within
        each sublattice share no neighbours so all can flip simultaneously.
        This is the GPU-friendly replacement for update_metropolis_hastings.
        """
        xp = self.xp
        self.at[t + 1, :, :] = self.at[t, :, :]

        for mask in (self._black_mask, self._white_mask):
            s = self.at[t + 1].astype(xp.float32)
            nb = (xp.roll(s, -1, 0) + xp.roll(s, 1, 0) +
                  xp.roll(s, -1, 1) + xp.roll(s, 1, 1))
            delta_e = -0.5 * (self.J * nb + self.mu * self.h) * s

            # Acceptance probability
            rand = xp.asarray(self.rng.random((self.nrows, self.ncols)).astype(np.float32))
            accept = (delta_e < 0) | (rand < xp.exp(-delta_e * self.beta))

            # Only flip spins in this sublattice that were accepted
            flip_mask = mask & accept
            self.at[t + 1] = xp.where(flip_mask, -self.at[t + 1], self.at[t + 1])

    def update_wolff(self, t):
        """Wolff cluster algorithm — sequential by nature, runs on CPU."""
        # Pull frame to CPU for set-based cluster growth
        frame_cpu = _to_numpy(self.at[t + 1])
        frame_cpu[:] = _to_numpy(self.at[t])

        p = 1 - np.exp(-2 * self.beta * self.J)
        seed_i = int(self.rng.integers(low=0, high=self.nrows))
        seed_j = int(self.rng.integers(low=0, high=self.ncols))
        cluster = {(seed_i, seed_j)}
        perimeter = {(seed_i, seed_j)}

        while perimeter:
            idx = self.rng.integers(low=0, high=len(perimeter))
            seed_spin = list(perimeter)[idx]
            perimeter.remove(seed_spin)
            si, sj = seed_spin
            for ni, nj in [
                ((si + 1) % self.nrows, sj),
                ((si - 1) % self.nrows, sj),
                (si, (sj + 1) % self.ncols),
                (si, (sj - 1) % self.ncols),
            ]:
                spin = (ni, nj)
                if (frame_cpu[ni, nj] == frame_cpu[si, sj] and
                        spin not in cluster and
                        self.rng.random() < p):
                    cluster.add(spin)
                    perimeter.add(spin)

        for ci, cj in cluster:
            frame_cpu[ci, cj] = -frame_cpu[ci, cj]

        if self.use_gpu:
            self.at[t + 1] = cp.asarray(frame_cpu)
        else:
            self.at[t + 1] = frame_cpu

    # ------------------------------------------------------------------
    # Quench / gather
    # ------------------------------------------------------------------

    def quench(self, beta=None, h=None, method=None, verbose=True):
        if beta is not None:
            self.beta = beta
            self.T = 1 / (KB * self.beta)
        if h is not None:
            self.h = h
        if method is not None:
            self.method = method

        _range = trange(self.frames - 1, desc='Quenching System', leave=False) if verbose else range(self.frames - 1)

        if self.method == 'wolff':
            for _ in _range:
                self.update_wolff(_)
        else:
            # Use checkerboard on GPU, sequential MH on CPU
            update_fn = self.update_checkerboard if self.use_gpu else self.update_metropolis_hastings
            for _ in _range:
                update_fn(_)

    def gather_data(self, verbose=True):
        _range = trange(self.frames, desc='Gathering Data', leave=False) if verbose else range(self.frames)
        for _ in _range:
            self.energy[_] = self.E(_)
            self.energy_sqr[_] = self.E_sqr(_)
            self.magnetization[_] = self.M(_)
            self.magnetization_sqr[_] = self.M_sqr(_)
            self.specific_heat_capacity[_] = self.C(_)
            self.magnetic_susceptibility[_] = self.X(_)

    # ------------------------------------------------------------------
    # Step 3 — Batched simulate()
    # ------------------------------------------------------------------

    def simulate(self, beta_range, h_range, sample_size=30, method='wolff'):
        """
        Sweep over beta_range × h_range, averaging over sample_size independent
        realisations.

        On GPU with method='metropolis-hastings', all samples for a given
        (beta, h) are stacked into a batch dimension and updated simultaneously,
        giving up to sample_size× throughput improvement.
        """
        self.sample_size = sample_size
        beta_range = np.asarray(beta_range)
        h_range = np.asarray(h_range)

        data_size = len(beta_range) * len(h_range)
        h_results = np.zeros(data_size)
        beta_results = np.zeros(data_size)
        energy_results = np.zeros(data_size)
        specific_heat_capacity_results = np.zeros(data_size)
        magnetization_results = np.zeros(data_size)
        magnetic_susceptibility_results = np.zeros(data_size)

        use_batched = self.use_gpu and method != 'wolff'

        for idx_h, h in tqdm(enumerate(h_range), desc='Sweeping H', total=len(h_range), position=0, leave=False):
            for idx_beta, beta in tqdm(enumerate(beta_range), desc='Sweeping Beta', total=len(beta_range), position=1, leave=False):
                idx = idx_h * len(beta_range) + idx_beta

                if use_batched:
                    e_mean, c_mean, m_mean, x_mean = self._simulate_batch(
                        beta=float(beta), h=float(h), method=method, sample_size=sample_size)
                else:
                    e_mean, c_mean, m_mean, x_mean = self._simulate_sequential(
                        beta=float(beta), h=float(h), method=method, sample_size=sample_size)

                beta_results[idx] = beta
                h_results[idx] = h
                energy_results[idx] = e_mean
                specific_heat_capacity_results[idx] = c_mean
                magnetization_results[idx] = m_mean
                magnetic_susceptibility_results[idx] = x_mean

        return pd.DataFrame(
            np.vstack((
                h_results, beta_results,
                energy_results, specific_heat_capacity_results,
                magnetization_results, magnetic_susceptibility_results,
            )).T,
            columns=[
                'ExternalMagneticField', 'ThermodynamicBeta',
                'Energy', 'SpecificHeatCapacity',
                'Magnetization', 'MagneticSusceptibility',
            ])

    def _simulate_sequential(self, beta, h, method, sample_size):
        """Run sample_size independent simulations one at a time (CPU or Wolff)."""
        e_s = np.zeros(sample_size)
        c_s = np.zeros(sample_size)
        m_s = np.zeros(sample_size)
        x_s = np.zeros(sample_size)

        for idx in range(sample_size):
            self.reset()
            self.quench(beta=beta, h=h, method=method, verbose=False)
            self.gather_data(verbose=False)
            e_s[idx] = self.energy[-1]
            c_s[idx] = self.specific_heat_capacity[-1]
            m_s[idx] = self.magnetization[-1]
            x_s[idx] = self.magnetic_susceptibility[-1]

        return e_s.mean(), c_s.mean(), m_s.mean(), x_s.mean()

    def _simulate_batch(self, beta, h, method, sample_size):
        """
        Step 3 — Run sample_size simulations in parallel on GPU.

        A batch lattice of shape (sample_size, nrows, ncols) is evolved
        simultaneously using the checkerboard algorithm.
        """
        xp = self.xp
        T_local = 1.0 / (KB * beta)

        # Initialise batch of random lattices
        raw = (2 * self.rng.integers(0, 2, size=(sample_size, self.nrows, self.ncols)).astype(np.int8) - 1)
        batch = xp.asarray(raw)  # (S, R, C)

        black, white = _checkerboard_masks(self.nrows, self.ncols, xp)

        def _batch_checkerboard_step(batch):
            for mask in (black, white):
                s = batch.astype(xp.float32)
                nb = (xp.roll(s, -1, 1) + xp.roll(s, 1, 1) +
                      xp.roll(s, -1, 2) + xp.roll(s, 1, 2))
                delta_e = -0.5 * (self.J * nb + self.mu * h) * s
                rand = xp.asarray(
                    self.rng.random((sample_size, self.nrows, self.ncols)).astype(np.float32))
                accept = (delta_e < 0) | (rand < xp.exp(-delta_e * beta))
                flip = mask[None, :, :] & accept
                batch = xp.where(flip, -batch, batch)
            return batch

        # Quench
        for _ in range(self.frames - 1):
            batch = _batch_checkerboard_step(batch)

        # Gather observables from final frame
        s = batch.astype(xp.float32)
        nb = (xp.roll(s, -1, 1) + xp.roll(s, 1, 1) +
              xp.roll(s, -1, 2) + xp.roll(s, 1, 2))

        per_site = -0.5 * self.J * s * nb - self.mu * h * s  # (S, R, C)
        e_per_sample = xp.sum(per_site, axis=(1, 2)) / self.size          # (S,)
        e_sqr_per_sample = xp.sum(per_site ** 2, axis=(1, 2)) / self.size # (S,)
        m_per_sample = xp.sum(s, axis=(1, 2)) / self.size                 # (S,)
        m_sqr_per_sample = xp.sum(s ** 2, axis=(1, 2)) / self.size        # (S,)

        e_arr = _to_numpy(e_per_sample)
        e_sqr_arr = _to_numpy(e_sqr_per_sample)
        m_arr = _to_numpy(m_per_sample)
        m_sqr_arr = _to_numpy(m_sqr_per_sample)

        c_arr = beta * (e_sqr_arr - e_arr ** 2) / T_local
        x_arr = beta * (m_sqr_arr - m_arr ** 2)

        return e_arr.mean(), c_arr.mean(), m_arr.mean(), x_arr.mean()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(self, results):
        results = results.copy()
        results['NetMagnetization'] = np.abs(results['Magnetization'])
        results['ScaledTemperature'] = np.power(KB * T_CRIT * results['ThermodynamicBeta'], -1)

        os.makedirs('results', exist_ok=True)
        file_name = os.path.join('results', f'pyising_{self.__repr__()}.png')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, layout='constrained', figsize=(19.2, 10.8))
        fig.suptitle('PyIsing Simulation Results\n' + self.__repr__())
        fig.supxlabel(r'$T/T_c$')
        ax1.scatter(results['ScaledTemperature'], results['Energy'], color='darkgreen', label=r'$\left<E\right>$')
        ax1.set_ylabel(r'Energy, $\left<E\right>$')
        ax2.scatter(results['ScaledTemperature'], results['NetMagnetization'], color='darkorange', label=r'$\left|\left<M\right>\right|$')
        ax2.set_ylabel(r'Net Magnetization, $\left|\left<M\right>\right|$')
        ax3.scatter(results['ScaledTemperature'], results['SpecificHeatCapacity'], color='darkred', label=r'$\left<C\right>$')
        ax3.set_ylabel(r'Specific Heat Capacity, $\left<C\right>$')
        for ax in (ax1, ax2, ax3):
            ax.legend()
            ax.grid()
        fig.savefig(file_name)
        return file_name

    def save_video(self):
        if not _CV2_AVAILABLE:
            raise RuntimeError('cv2 (opencv-python) is required for save_video()')

        frames_dir = os.path.join(os.getcwd(), 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs('results', exist_ok=True)

        # Write frames
        for _ in trange(self.frames, desc='Gathering Frames'):
            frame_np = _to_numpy(self.at[_, :, :])
            file_name = os.path.join(frames_dir, f'pyising_{self.__str__()}_frame_{str(_).zfill(4)}.png')
            plt.imsave(file_name, frame_np, vmin=-1, vmax=1, cmap='coolwarm')

        # Write video
        file_name = os.path.join('results', f'pyising_{self.__str__()}.avi')
        images = sorted(img for img in os.listdir(frames_dir) if img.endswith('.png'))
        titleframe = cv2.imread(os.path.join(frames_dir, images[0]))
        height, width, _ = titleframe.shape
        video = cv2.VideoWriter(file_name, 0, self.fps, (width, height))
        for _, image in tqdm(enumerate(images), desc='Writing Video', total=self.frames):
            video.write(cv2.imread(os.path.join(frames_dir, image)))
        cv2.destroyAllWindows()
        video.release()

        # Clean up frames
        for _, img in tqdm(enumerate(images), desc='Releasing Frames', total=self.frames):
            os.remove(os.path.join(frames_dir, img))

        return file_name

    def visualize(self, h=None, beta=None, method=None):
        if beta is not None:
            self.beta = beta
            self.T = 1 / (KB * self.beta)
        if h is not None:
            self.h = h
        if method is not None:
            self.method = method
        self.quench(beta=self.beta, h=self.h, method=self.method)
        self.save_video()
