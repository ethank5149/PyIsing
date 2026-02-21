"""
Core IsingModel class.

GPU acceleration is provided transparently via CuPy when available.
Pass ``use_gpu=False`` to force CPU execution.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from pyising.constants import KB, T_CRIT, BETA_CRIT

# ---------------------------------------------------------------------------
# Backend selection
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


def _xp(use_gpu: bool):
    return cp if (use_gpu and _GPU_AVAILABLE) else np


def _to_numpy(arr) -> np.ndarray:
    if _GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Checkerboard helpers
# ---------------------------------------------------------------------------

def _checkerboard_masks(nrows: int, ncols: int, xp):
    rows = xp.arange(nrows, dtype=xp.int32)[:, None]
    cols = xp.arange(ncols, dtype=xp.int32)[None, :]
    black = (rows + cols) % 2 == 0
    return black, ~black


# ---------------------------------------------------------------------------
# IsingModel
# ---------------------------------------------------------------------------

class IsingModel:
    """
    2D Ising model on a periodic square lattice.

    Parameters
    ----------
    tf : int
        Simulation time (seconds equivalent; total frames = tf * fps).
    fps : int
        Frames per second for video output / iteration density.
    ncols, nrows : int
        Lattice dimensions.
    J : float
        Exchange coupling constant (J > 0 → ferromagnetic).
    mu : float
        Magnetic moment.
    beta : float
        Inverse temperature β = 1 / (k_B T).
    h : float
        External magnetic field.
    method : str
        Default update algorithm. Options: ``'metropolis-hastings'``, ``'glauber'``,
        ``'overrelaxation'``, ``'wolff'``, ``'swendsen-wang'``, ``'invaded-cluster'``,
        ``'kinetic-mc'``. For ``'wang-landau'`` and ``'parallel-tempering'``, use
        the dedicated :meth:`simulate_wang_landau` and :meth:`simulate_parallel_tempering`
        methods directly.
    sample_size : int
        Default number of independent samples per (β, h) point in simulate().
    h_range : array-like
        Default external field sweep range.
    beta_range : array-like
        Default inverse-temperature sweep range.
    use_gpu : bool
        If True (default), use CuPy/CUDA when available.
    """

    def __init__(
            self,
            tf: int = 10,
            fps: int = 10,
            ncols: int = 10,
            nrows: int = 10,
            J: float = 1.0,
            mu: float = 1.0,
            beta: float = 1.5 * BETA_CRIT,
            h: float = 0.0,
            method: str = 'metropolis-hastings',
            sample_size: int = 30,
            h_range=None,
            beta_range=None,
            use_gpu: bool = True,
    ):
        if h_range is None:
            h_range = [0.0]
        if beta_range is None:
            beta_range = np.linspace(0.5 * BETA_CRIT, 10 * BETA_CRIT, 100)

        self.use_gpu = use_gpu and _GPU_AVAILABLE
        self.xp = _xp(self.use_gpu)

        self.rng = np.random.default_rng()

        self.nrows, self.ncols = nrows, ncols
        self.size = nrows * ncols
        self.J, self.mu = J, mu
        self.beta, self.h = beta, h
        self.T = 1.0 / (KB * self.beta)

        self.tf, self.fps = tf, fps
        self.frames = self.tf * self.fps
        self.sample_size = sample_size
        self.method = method
        self.beta_range = np.asarray(beta_range)
        self.h_range = np.asarray(h_range)

        # Observables (always CPU)
        self.specific_heat_capacity = np.zeros(self.frames)
        self.magnetic_susceptibility = np.zeros(self.frames)
        self.energy = np.zeros(self.frames)
        self.magnetization = np.zeros(self.frames)
        self.energy_sqr = np.zeros(self.frames)
        self.magnetization_sqr = np.zeros(self.frames)

        # Spin lattice
        self._init_lattice()

        # Checkerboard masks (pre-computed once)
        self._black_mask, self._white_mask = _checkerboard_masks(nrows, ncols, self.xp)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_lattice(self) -> None:
        raw = (2 * self.rng.integers(0, 2, size=(self.frames, self.nrows, self.ncols)).astype(np.int8) - 1)
        self.at = self.xp.asarray(raw) if self.use_gpu else raw

    def __repr__(self) -> str:
        backend = 'gpu' if self.use_gpu else 'cpu'
        return (f'{self.ncols}x{self.nrows}_{self.sample_size}samples_'
                f'{self.frames}iters_nbeta={len(self.beta_range)}_'
                f'nh={len(self.h_range)}_{self.method}_alg_{backend}')

    def __str__(self) -> str:
        return (f'{self.ncols}x{self.nrows}_tf={self.tf}_fps={self.fps}_'
                f'b={self.beta:0.4f}_h={self.h:0.4f}_{self.method}_alg')

    def reset(self) -> None:
        """Re-randomise frame 0 of the spin lattice."""
        raw = (2 * self.rng.integers(0, 2, size=(self.nrows, self.ncols)).astype(np.int8) - 1)
        self.at[0] = self.xp.asarray(raw) if self.use_gpu else raw

    # ------------------------------------------------------------------
    # Neighbour sums
    # ------------------------------------------------------------------

    def metric(self, t: int):
        """Vectorised sum of 4 nearest neighbours for the full lattice at frame *t*."""
        xp, s = self.xp, self.at[t]
        return xp.roll(s, -1, 0) + xp.roll(s, 1, 0) + xp.roll(s, -1, 1) + xp.roll(s, 1, 1)

    def metric_at(self, t: int, i: int, j: int) -> int:
        """Scalar neighbour sum at site (i, j) — used only by the Wolff algorithm."""
        return (int(self.at[t, (i + 1) % self.nrows, j]) +
                int(self.at[t, (i - 1) % self.nrows, j]) +
                int(self.at[t, i, (j + 1) % self.ncols]) +
                int(self.at[t, i, (j - 1) % self.ncols]))

    def neighbors_at(self, t: int, i: int, j: int) -> List[tuple]:
        return [
            ((i + 1) % self.nrows, j),
            ((i - 1) % self.nrows, j),
            (i, (j + 1) % self.ncols),
            (i, (j - 1) % self.ncols),
        ]

    # ------------------------------------------------------------------
    # Thermodynamic observables
    # ------------------------------------------------------------------

    def E(self, t: int) -> float:
        """Mean energy per spin at frame *t*."""
        xp, s = self.xp, self.at[t]
        nb = self.metric(t)
        return float(-0.5 * self.J * xp.sum(s * nb) - self.mu * self.h * xp.sum(s)) / self.size

    def E_sqr(self, t: int) -> float:
        """Mean squared per-site energy at frame *t*."""
        xp, s = self.xp, self.at[t]
        nb = self.metric(t)
        per_site = -0.5 * self.J * s * nb - self.mu * self.h * s
        return float(xp.sum(per_site ** 2)) / self.size

    def M(self, t: int) -> float:
        """Mean magnetisation per spin at frame *t*."""
        return float(self.xp.sum(self.at[t])) / self.size

    def M_sqr(self, t: int) -> float:
        """Mean squared magnetisation per spin at frame *t*."""
        return float(self.xp.sum(self.at[t].astype(self.xp.float32) ** 2)) / self.size

    def C(self, t: int) -> float:
        """Specific heat capacity at frame *t* (requires :meth:`gather_data` first)."""
        return self.beta * (self.energy_sqr[t] - self.energy[t] ** 2) / self.T

    def X(self, t: int) -> float:
        """Magnetic susceptibility at frame *t* (requires :meth:`gather_data` first)."""
        return self.beta * (self.magnetization_sqr[t] - self.magnetization[t] ** 2)

    def _total_energy(self, lattice: np.ndarray) -> float:
        """Total energy of a lattice configuration (CPU-only).

        Parameters
        ----------
        lattice : np.ndarray
            Spin lattice of shape ``(nrows, ncols)`` with values ±1.

        Returns
        -------
        float
            Total energy including nearest-neighbour and external-field terms.
        """
        nb = (np.roll(lattice, -1, 0) + np.roll(lattice, 1, 0) +
              np.roll(lattice, -1, 1) + np.roll(lattice, 1, 1))
        return float(-0.5 * self.J * np.sum(lattice * nb) - self.mu * self.h * np.sum(lattice))

    # ------------------------------------------------------------------
    # Monte Carlo update steps
    # ------------------------------------------------------------------

    def update_metropolis_hastings(self, t: int) -> None:
        """Sequential (site-by-site) Metropolis-Hastings — CPU reference implementation."""
        i_s = self.rng.integers(0, self.nrows, size=self.size)
        j_s = self.rng.integers(0, self.ncols, size=self.size)
        self.at[t + 1] = self.at[t].copy()
        for k in range(self.size):
            i, j = int(i_s[k]), int(j_s[k])
            nb = self.metric_at(t + 1, i, j)
            delta_e = -0.5 * (self.J * nb + self.mu * self.h) * int(self.at[t + 1, i, j])
            p = 1.0 if delta_e < 0.0 else float(np.exp(-delta_e * self.beta))
            if self.rng.random() < p:
                self.at[t + 1, i, j] = -self.at[t + 1, i, j]

    def update_checkerboard(self, t: int) -> None:
        """
        Fully parallel checkerboard Metropolis-Hastings (GPU-optimised).

        Black and white sublattices are updated alternately; spins within
        each sublattice share no neighbours so all can flip simultaneously.
        """
        xp = self.xp
        self.at[t + 1] = self.at[t].copy()

        for mask in (self._black_mask, self._white_mask):
            s = self.at[t + 1].astype(xp.float32)
            nb = xp.roll(s, -1, 0) + xp.roll(s, 1, 0) + xp.roll(s, -1, 1) + xp.roll(s, 1, 1)
            delta_e = -0.5 * (self.J * nb + self.mu * self.h) * s
            rand = xp.asarray(self.rng.random((self.nrows, self.ncols)).astype(np.float32))
            accept = (delta_e < 0) | (rand < xp.exp(-delta_e * self.beta))
            self.at[t + 1] = xp.where(mask & accept, -self.at[t + 1], self.at[t + 1])

    def update_wolff(self, t: int) -> None:
        """Wolff cluster algorithm — sequential by nature, always runs on CPU."""
        frame = _to_numpy(self.at[t]).copy()

        p = 1.0 - np.exp(-2.0 * self.beta * self.J)
        si = int(self.rng.integers(0, self.nrows))
        sj = int(self.rng.integers(0, self.ncols))
        cluster = {(si, sj)}
        perimeter = {(si, sj)}

        while perimeter:
            idx = int(self.rng.integers(0, len(perimeter)))
            ci, cj = list(perimeter)[idx]
            perimeter.discard((ci, cj))
            for ni, nj in [
                ((ci + 1) % self.nrows, cj),
                ((ci - 1) % self.nrows, cj),
                (ci, (cj + 1) % self.ncols),
                (ci, (cj - 1) % self.ncols),
            ]:
                if (frame[ni, nj] == frame[ci, cj] and
                        (ni, nj) not in cluster and
                        self.rng.random() < p):
                    cluster.add((ni, nj))
                    perimeter.add((ni, nj))

        for ci, cj in cluster:
            frame[ci, cj] = -frame[ci, cj]

        self.at[t + 1] = self.xp.asarray(frame) if self.use_gpu else frame

    def update_glauber(self, t: int) -> None:
        """Sequential (site-by-site) Glauber heat bath dynamics."""
        i_s = self.rng.integers(0, self.nrows, size=self.size)
        j_s = self.rng.integers(0, self.ncols, size=self.size)
        self.at[t + 1] = self.at[t].copy()
        for k in range(self.size):
            i, j = int(i_s[k]), int(j_s[k])
            nb = self.metric_at(t + 1, i, j)
            local_field = self.J * nb + self.mu * self.h
            p_plus = 1.0 / (1.0 + float(np.exp(-self.beta * local_field)))
            self.at[t + 1, i, j] = 1 if self.rng.random() < p_plus else -1

    def update_glauber_checkerboard(self, t: int) -> None:
        """Fully parallel checkerboard Glauber heat bath dynamics (GPU-optimised)."""
        xp = self.xp
        self.at[t + 1] = self.at[t].copy()

        for mask in (self._black_mask, self._white_mask):
            s = self.at[t + 1].astype(xp.float32)
            nb = xp.roll(s, -1, 0) + xp.roll(s, 1, 0) + xp.roll(s, -1, 1) + xp.roll(s, 1, 1)
            local_field = self.J * nb + self.mu * self.h
            p_plus = 1.0 / (1.0 + xp.exp(-self.beta * local_field))
            rand = xp.asarray(self.rng.random((self.nrows, self.ncols)).astype(np.float32))
            new_spins = xp.where(rand < p_plus, xp.int8(1), xp.int8(-1))
            self.at[t + 1] = xp.where(mask, new_spins, self.at[t + 1])

    def update_overrelaxation(self, t: int) -> None:
        """Sequential (site-by-site) microcanonical overrelaxation."""
        i_s = self.rng.integers(0, self.nrows, size=self.size)
        j_s = self.rng.integers(0, self.ncols, size=self.size)
        self.at[t + 1] = self.at[t].copy()
        for k in range(self.size):
            i, j = int(i_s[k]), int(j_s[k])
            nb = self.metric_at(t + 1, i, j)
            h_local = self.J * nb + self.mu * self.h
            if abs(h_local) < 1e-10:
                self.at[t + 1, i, j] = -self.at[t + 1, i, j]

    def update_overrelaxation_checkerboard(self, t: int) -> None:
        """Fully parallel checkerboard microcanonical overrelaxation (GPU-optimised)."""
        xp = self.xp
        self.at[t + 1] = self.at[t].copy()

        for mask in (self._black_mask, self._white_mask):
            s = self.at[t + 1].astype(xp.float32)
            nb = xp.roll(s, -1, 0) + xp.roll(s, 1, 0) + xp.roll(s, -1, 1) + xp.roll(s, 1, 1)
            local_field = self.J * nb + self.mu * self.h
            flip = xp.abs(local_field) < 1e-10
            self.at[t + 1] = xp.where(mask & flip, -self.at[t + 1], self.at[t + 1])

    def update_swendsen_wang(self, t: int) -> None:
        """Swendsen-Wang cluster algorithm — identifies and flips all clusters simultaneously."""
        frame = _to_numpy(self.at[t]).copy()
        nrows, ncols = self.nrows, self.ncols

        p = 1.0 - np.exp(-2.0 * self.beta * self.J)

        # Union-Find data structure (inline)
        parent = np.arange(nrows * ncols)
        rank = np.zeros(nrows * ncols, dtype=np.int32)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        # Activate bonds between aligned neighbours
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                # Right neighbour
                rj = (j + 1) % ncols
                if frame[i, j] == frame[i, rj] and self.rng.random() < p:
                    union(idx, i * ncols + rj)
                # Bottom neighbour
                bi = (i + 1) % nrows
                if frame[i, j] == frame[bi, j] and self.rng.random() < p:
                    union(idx, bi * ncols + j)

        # Find root for every site
        roots = np.array([find(k) for k in range(nrows * ncols)])

        # Assign a random flip decision to each unique cluster
        unique_roots = np.unique(roots)
        flip_decision = self.rng.random(len(unique_roots)) < 0.5
        flip_map = dict(zip(unique_roots, flip_decision))

        # Apply flips
        for k in range(nrows * ncols):
            if flip_map[roots[k]]:
                i, j = divmod(k, ncols)
                frame[i, j] = -frame[i, j]

        self.at[t + 1] = self.xp.asarray(frame) if self.use_gpu else frame

    def update_invaded_cluster(self, t: int) -> None:
        """Invaded cluster algorithm — self-tuning cluster method that finds the critical point."""
        frame = _to_numpy(self.at[t]).copy()
        nrows, ncols = self.nrows, self.ncols
        total = nrows * ncols
        half = self.size // 2

        # Union-Find data structure (inline)
        parent = np.arange(total)
        uf_rank = np.zeros(total, dtype=np.int32)
        cluster_size = np.ones(total, dtype=np.int32)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression (pointer halving)
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if uf_rank[ra] < uf_rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            cluster_size[ra] += cluster_size[rb]
            if uf_rank[ra] == uf_rank[rb]:
                uf_rank[ra] += 1

        # Build list of bonds between aligned nearest neighbours
        bonds = []
        for i in range(nrows):
            for j in range(ncols):
                # Right neighbour
                rj = (j + 1) % ncols
                if frame[i, j] == frame[i, rj]:
                    bonds.append((self.rng.random(), i * ncols + j, i * ncols + rj))
                # Bottom neighbour
                bi = (i + 1) % nrows
                if frame[i, j] == frame[bi, j]:
                    bonds.append((self.rng.random(), i * ncols + j, bi * ncols + j))

        # Sort bonds by weight (ascending)
        bonds.sort(key=lambda b: b[0])

        # Add bonds in order until a cluster spans the lattice
        for weight, idx1, idx2 in bonds:
            union(idx1, idx2)
            root = find(idx1)
            if cluster_size[root] >= half:
                break

        # Find root for every site
        roots = np.array([find(k) for k in range(total)])

        # Assign a random flip decision to each unique cluster
        unique_roots = np.unique(roots)
        flip_decision = self.rng.random(len(unique_roots)) < 0.5
        flip_map = dict(zip(unique_roots, flip_decision))

        # Apply flips
        for k in range(total):
            if flip_map[roots[k]]:
                i, j = divmod(k, ncols)
                frame[i, j] = -frame[i, j]

        self.at[t + 1] = self.xp.asarray(frame) if self.use_gpu else frame

    def update_kinetic_mc(self, t: int) -> None:
        """Kinetic Monte Carlo — event-driven continuous-time algorithm with Glauber rates."""
        frame = _to_numpy(self.at[t]).copy()
        nrows, ncols = self.nrows, self.ncols

        # Compute flip rates for all sites using Glauber dynamics (vectorized)
        nb = (np.roll(frame, -1, 0) + np.roll(frame, 1, 0) +
              np.roll(frame, -1, 1) + np.roll(frame, 1, 1))
        delta_E = 2.0 * self.J * frame * nb + 2.0 * self.mu * self.h * frame
        rates = 1.0 / (1.0 + np.exp(self.beta * delta_E.astype(np.float64)))

        # Perform self.size events (one sweep equivalent)
        for _ in range(self.size):
            R_total = rates.sum()
            if R_total < 1e-15:
                break  # system is frozen

            # Select site with probability proportional to its rate
            flat_rates = rates.ravel()
            cumulative = np.cumsum(flat_rates)
            r = self.rng.random() * R_total
            idx = np.searchsorted(cumulative, r)
            i, j = divmod(idx, ncols)

            # Flip the selected spin
            frame[i, j] = -frame[i, j]

            # Update rates for the flipped site and its 4 neighbours
            for (ni, nj) in [(i, j), ((i + 1) % nrows, j), ((i - 1) % nrows, j),
                             (i, (j + 1) % ncols), (i, (j - 1) % ncols)]:
                nb_val = (frame[(ni + 1) % nrows, nj] + frame[(ni - 1) % nrows, nj] +
                          frame[ni, (nj + 1) % ncols] + frame[ni, (nj - 1) % ncols])
                dE = 2.0 * self.J * frame[ni, nj] * nb_val + 2.0 * self.mu * self.h * frame[ni, nj]
                rates[ni, nj] = 1.0 / (1.0 + np.exp(self.beta * float(dE)))

        self.at[t + 1] = self.xp.asarray(frame) if self.use_gpu else frame

    # ------------------------------------------------------------------
    # Quench / gather
    # ------------------------------------------------------------------

    def quench(self, beta: Optional[float] = None, h: Optional[float] = None,
               method: Optional[str] = None, verbose: bool = True) -> None:
        """Evolve the lattice for ``self.frames - 1`` steps."""
        if beta is not None:
            self.beta = beta
            self.T = 1.0 / (KB * self.beta)
        if h is not None:
            self.h = h
        if method is not None:
            self.method = method

        _range = (trange(self.frames - 1, desc='Quenching System', leave=False)
                  if verbose else range(self.frames - 1))

        _DISPATCH = {
            'metropolis-hastings': (self.update_metropolis_hastings, self.update_checkerboard),
            'glauber': (self.update_glauber, self.update_glauber_checkerboard),
            'overrelaxation': (self.update_overrelaxation, self.update_overrelaxation_checkerboard),
            'wolff': (self.update_wolff, self.update_wolff),
            'swendsen-wang': (self.update_swendsen_wang, self.update_swendsen_wang),
            'invaded-cluster': (self.update_invaded_cluster, self.update_invaded_cluster),
            'kinetic-mc': (self.update_kinetic_mc, self.update_kinetic_mc),
        }

        if self.method not in _DISPATCH:
            raise ValueError(f"Unknown method '{self.method}'. Choose from: {list(_DISPATCH.keys())}")

        cpu_fn, gpu_fn = _DISPATCH[self.method]
        update_fn = gpu_fn if self.use_gpu else cpu_fn

        for _ in _range:
            update_fn(_)

    def gather_data(self, verbose: bool = True) -> None:
        """Compute and store all observables for every frame."""
        _range = (trange(self.frames, desc='Gathering Data', leave=False)
                  if verbose else range(self.frames))
        for _ in _range:
            self.energy[_] = self.E(_)
            self.energy_sqr[_] = self.E_sqr(_)
            self.magnetization[_] = self.M(_)
            self.magnetization_sqr[_] = self.M_sqr(_)
            self.specific_heat_capacity[_] = self.C(_)
            self.magnetic_susceptibility[_] = self.X(_)

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------

    def simulate(
            self,
            beta_range=None,
            h_range=None,
            sample_size: int = 30,
            method: str = 'wolff',
    ) -> pd.DataFrame:
        """
        Sweep over *beta_range* × *h_range*, averaging over *sample_size*
        independent realisations.

        On GPU with ``method='metropolis-hastings'``, all samples for a given
        (β, h) are stacked into a batch dimension and updated simultaneously.
        """
        if beta_range is None:
            beta_range = self.beta_range
        if h_range is None:
            h_range = self.h_range
        beta_range = np.asarray(beta_range)
        h_range = np.asarray(h_range)
        self.sample_size = sample_size

        data_size = len(beta_range) * len(h_range)
        h_res = np.zeros(data_size)
        b_res = np.zeros(data_size)
        e_res = np.zeros(data_size)
        c_res = np.zeros(data_size)
        m_res = np.zeros(data_size)
        x_res = np.zeros(data_size)

        use_batched = self.use_gpu and method in ('metropolis-hastings', 'glauber')

        for idx_h, h in tqdm(enumerate(h_range), desc='Sweeping H',
                              total=len(h_range), position=0, leave=False):
            for idx_b, beta in tqdm(enumerate(beta_range), desc='Sweeping Beta',
                                    total=len(beta_range), position=1, leave=False):
                idx = idx_h * len(beta_range) + idx_b
                fn = self._simulate_batch if use_batched else self._simulate_sequential
                e_mean, c_mean, m_mean, x_mean = fn(
                    beta=float(beta), h=float(h), method=method, sample_size=sample_size)
                b_res[idx] = beta
                h_res[idx] = h
                e_res[idx] = e_mean
                c_res[idx] = c_mean
                m_res[idx] = m_mean
                x_res[idx] = x_mean

        return pd.DataFrame(
            np.vstack((h_res, b_res, e_res, c_res, m_res, x_res)).T,
            columns=['ExternalMagneticField', 'ThermodynamicBeta',
                     'Energy', 'SpecificHeatCapacity',
                     'Magnetization', 'MagneticSusceptibility'])

    def _simulate_sequential(self, beta, h, method, sample_size):
        e_s = np.zeros(sample_size)
        c_s = np.zeros(sample_size)
        m_s = np.zeros(sample_size)
        x_s = np.zeros(sample_size)
        for i in range(sample_size):
            self.reset()
            self.quench(beta=beta, h=h, method=method, verbose=False)
            self.gather_data(verbose=False)
            e_s[i] = self.energy[-1]
            c_s[i] = self.specific_heat_capacity[-1]
            m_s[i] = self.magnetization[-1]
            x_s[i] = self.magnetic_susceptibility[-1]
        return e_s.mean(), c_s.mean(), m_s.mean(), x_s.mean()

    def _simulate_batch(self, beta, h, method, sample_size):
        """Run *sample_size* simulations in parallel on GPU (checkerboard MH)."""
        xp = self.xp
        T_local = 1.0 / (KB * beta)

        raw = (2 * self.rng.integers(0, 2, size=(sample_size, self.nrows, self.ncols)).astype(np.int8) - 1)
        batch = xp.asarray(raw)  # (S, R, C)

        black, white = _checkerboard_masks(self.nrows, self.ncols, xp)

        def _step(b):
            for mask in (black, white):
                s = b.astype(xp.float32)
                nb = (xp.roll(s, -1, 1) + xp.roll(s, 1, 1) +
                      xp.roll(s, -1, 2) + xp.roll(s, 1, 2))
                delta_e = -0.5 * (self.J * nb + self.mu * h) * s
                rand = xp.asarray(
                    self.rng.random((sample_size, self.nrows, self.ncols)).astype(np.float32))
                accept = (delta_e < 0) | (rand < xp.exp(-delta_e * beta))
                b = xp.where(mask[None, :, :] & accept, -b, b)
            return b

        for _ in range(self.frames - 1):
            batch = _step(batch)

        # Observables from final frame
        s = batch.astype(xp.float32)
        nb = (xp.roll(s, -1, 1) + xp.roll(s, 1, 1) +
              xp.roll(s, -1, 2) + xp.roll(s, 1, 2))
        per_site = -0.5 * self.J * s * nb - self.mu * h * s
        e_arr = _to_numpy(xp.sum(per_site, axis=(1, 2)) / self.size)
        e_sqr_arr = _to_numpy(xp.sum(per_site ** 2, axis=(1, 2)) / self.size)
        m_arr = _to_numpy(xp.sum(s, axis=(1, 2)) / self.size)
        m_sqr_arr = _to_numpy(xp.sum(s ** 2, axis=(1, 2)) / self.size)

        c_arr = beta * (e_sqr_arr - e_arr ** 2) / T_local
        x_arr = beta * (m_sqr_arr - m_arr ** 2)
        return e_arr.mean(), c_arr.mean(), m_arr.mean(), x_arr.mean()

    # ------------------------------------------------------------------
    # Wang-Landau
    # ------------------------------------------------------------------

    def simulate_wang_landau(
            self,
            flatness: float = 0.8,
            f_min: float = 1e-8,
            max_mc_sweeps: int = 10000,
    ) -> dict:
        """Wang-Landau flat histogram estimation of the density of states.

        This is a standalone simulation that does **not** use the frame-by-frame
        ``self.at`` storage.  It operates on a single lattice configuration and
        iteratively builds up ``log g(E)`` until the modification factor is
        smaller than ``ln(f_min)``.

        Parameters
        ----------
        flatness : float
            Fraction of the mean histogram count that the minimum must exceed
            for the histogram to be considered flat (default 0.8).
        f_min : float
            Stopping criterion — iteration ends when the multiplicative
            modification factor *f* drops below this value (default 1e-8).
        max_mc_sweeps : int
            Maximum number of MC sweeps per flatness iteration (default 10000).

        Returns
        -------
        dict
            ``{'log_g': {E: ln g(E), ...}, 'energies': [sorted energies]}``
        """
        # --- initialise lattice (CPU / numpy only) -----------------------
        lattice = (2 * self.rng.integers(0, 2,
                                         size=(self.nrows, self.ncols)).astype(np.int8) - 1)
        E_total = self._total_energy(lattice)

        log_g: defaultdict[float, float] = defaultdict(float)
        histogram: defaultdict[float, int] = defaultdict(int)
        ln_f = 1.0  # ln(e) ≈ 1

        # --- main Wang-Landau loop ----------------------------------------
        while ln_f > f_min:
            histogram = defaultdict(int)  # reset histogram each stage

            for _sweep in range(max_mc_sweeps):
                # One MC sweep = self.size single-spin-flip trials
                for _trial in range(self.size):
                    i = int(self.rng.integers(0, self.nrows))
                    j = int(self.rng.integers(0, self.ncols))

                    # Neighbour sum with periodic boundaries
                    nb_sum = int(
                        lattice[(i + 1) % self.nrows, j]
                        + lattice[(i - 1) % self.nrows, j]
                        + lattice[i, (j + 1) % self.ncols]
                        + lattice[i, (j - 1) % self.ncols]
                    )
                    delta_E = 2.0 * self.J * lattice[i, j] * nb_sum \
                              + 2.0 * self.mu * self.h * lattice[i, j]
                    E_new = E_total + delta_E

                    # Accept based on density-of-states ratio
                    if log_g[E_total] - log_g[E_new] >= 0:
                        accept = True
                    else:
                        accept = self.rng.random() < np.exp(log_g[E_total] - log_g[E_new])

                    if accept:
                        lattice[i, j] = -lattice[i, j]
                        E_total = E_new

                    log_g[E_total] += ln_f
                    histogram[E_total] += 1

                # Flatness check after each sweep
                if len(histogram) > 1:
                    counts = np.array(list(histogram.values()), dtype=np.float64)
                    if counts.min() > flatness * counts.mean():
                        break  # histogram is flat — proceed to next stage

            # Reduce modification factor
            ln_f /= 2.0

        return {
            'log_g': dict(log_g),
            'energies': sorted(log_g.keys()),
        }

    def wang_landau_observables(
            self,
            wl_result: dict,
            beta_range=None,
    ) -> pd.DataFrame:
        """Compute canonical observables from a Wang-Landau density of states.

        For each inverse temperature β in *beta_range*, the canonical average
        energy ⟨E⟩, specific heat *C*, and placeholders for magnetisation
        quantities are computed analytically from ``log g(E)``.

        Parameters
        ----------
        wl_result : dict
            Output of :meth:`simulate_wang_landau`.
        beta_range : array-like or None
            Inverse temperatures at which to evaluate observables.  Defaults to
            ``self.beta_range`` when *None*.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns matching :meth:`simulate`:
            ``['ExternalMagneticField', 'ThermodynamicBeta', 'Energy',
            'SpecificHeatCapacity', 'Magnetization',
            'MagneticSusceptibility']``.
        """
        if beta_range is None:
            beta_range = self.beta_range
        beta_range = np.asarray(beta_range, dtype=np.float64)

        log_g = wl_result['log_g']
        energies = np.array(wl_result['energies'], dtype=np.float64)
        log_g_arr = np.array([log_g[e] for e in energies], dtype=np.float64)

        n_beta = len(beta_range)
        h_res = np.full(n_beta, self.h)
        b_res = np.empty(n_beta)
        e_res = np.empty(n_beta)
        c_res = np.empty(n_beta)
        m_res = np.zeros(n_beta)
        x_res = np.zeros(n_beta)

        for idx, beta in enumerate(beta_range):
            log_weights = log_g_arr - beta * energies
            log_weights -= log_weights.max()  # numerical stability
            weights = np.exp(log_weights)
            Z = weights.sum()

            E_mean = np.sum(energies * weights) / Z
            E2_mean = np.sum(energies ** 2 * weights) / Z
            C_val = beta ** 2 * (E2_mean - E_mean ** 2) / self.size

            b_res[idx] = beta
            e_res[idx] = E_mean / self.size  # per-spin energy
            c_res[idx] = C_val

        return pd.DataFrame(
            np.vstack((h_res, b_res, e_res, c_res, m_res, x_res)).T,
            columns=['ExternalMagneticField', 'ThermodynamicBeta',
                     'Energy', 'SpecificHeatCapacity',
                     'Magnetization', 'MagneticSusceptibility'],
        )

    # ------------------------------------------------------------------
    # Parallel Tempering
    # ------------------------------------------------------------------

    def simulate_parallel_tempering(
            self,
            beta_range=None,
            h: float = 0.0,
            n_sweeps: int = 100,
            swap_interval: int = 1,
            method: str = 'metropolis-hastings',
    ) -> pd.DataFrame:
        """Parallel Tempering (Replica Exchange Monte Carlo) simulation.

        Runs multiple replicas at different inverse temperatures simultaneously,
        periodically attempting to swap configurations between adjacent-temperature
        replicas.  This is a standalone simulation method that returns per-replica
        observables from the final configurations.

        Parameters
        ----------
        beta_range : array-like or None
            Inverse temperatures for the replica ladder.  If *None*, uses
            ``self.beta_range``.  Will be sorted ascending internally.
        h : float
            External magnetic field applied to all replicas (default 0.0).
        n_sweeps : int
            Number of full Monte Carlo sweeps to perform (default 100).
        swap_interval : int
            Attempt replica swaps every *swap_interval* sweeps (default 1).
        method : str
            Single-replica update algorithm.  ``'metropolis-hastings'`` (default)
            enables GPU-accelerated checkerboard updates when a GPU is available.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per replica and columns:
            ``['ExternalMagneticField', 'ThermodynamicBeta', 'Energy',
            'SpecificHeatCapacity', 'Magnetization',
            'MagneticSusceptibility']``.
        """
        # ---- 1. Initialise ------------------------------------------------
        if beta_range is None:
            beta_range = self.beta_range
        beta_arr = np.sort(np.asarray(beta_range, dtype=np.float64))
        n_replicas = len(beta_arr)

        # Create independent random spin lattices (CPU numpy)
        replicas = [
            (2 * self.rng.integers(0, 2, size=(self.nrows, self.ncols)).astype(np.int8) - 1)
            for _ in range(n_replicas)
        ]

        use_gpu_batch = self.use_gpu and method != 'wolff'

        if use_gpu_batch:
            xp = self.xp
            # Stack replicas into a single GPU array: (n_replicas, nrows, ncols)
            batch = xp.asarray(np.stack(replicas, axis=0))
            black, white = _checkerboard_masks(self.nrows, self.ncols, xp)
            beta_broadcast = xp.asarray(
                beta_arr[:, None, None].astype(np.float32))  # (n_replicas, 1, 1)

        # ---- 2. Main loop -------------------------------------------------
        for sweep in trange(n_sweeps, desc='Parallel Tempering', leave=False):

            # -- 2a. Update each replica with one MC sweep ------------------
            if use_gpu_batch:
                # GPU checkerboard update for all replicas simultaneously
                for mask in (black, white):
                    s = batch.astype(xp.float32)
                    nb = (xp.roll(s, -1, 1) + xp.roll(s, 1, 1) +
                          xp.roll(s, -1, 2) + xp.roll(s, 1, 2))
                    delta_e = 2.0 * (self.J * s * nb + self.mu * h * s)
                    rand = xp.asarray(
                        self.rng.random(
                            (n_replicas, self.nrows, self.ncols)
                        ).astype(np.float32))
                    accept = (delta_e <= 0) | (rand < xp.exp(-beta_broadcast * delta_e))
                    batch = xp.where(mask[None, :, :] & accept, -batch, batch)
            else:
                # CPU path: iterate over replicas, each gets self.size MH trials
                for r in range(n_replicas):
                    lattice = replicas[r]
                    beta_r = beta_arr[r]
                    for _trial in range(self.size):
                        i = int(self.rng.integers(0, self.nrows))
                        j = int(self.rng.integers(0, self.ncols))
                        nb_sum = int(
                            lattice[(i + 1) % self.nrows, j]
                            + lattice[(i - 1) % self.nrows, j]
                            + lattice[i, (j + 1) % self.ncols]
                            + lattice[i, (j - 1) % self.ncols]
                        )
                        delta_E = (2.0 * self.J * lattice[i, j] * nb_sum
                                   + 2.0 * self.mu * h * lattice[i, j])
                        if delta_E <= 0 or self.rng.random() < np.exp(-beta_r * delta_E):
                            lattice[i, j] = -lattice[i, j]

            # -- 2b. Replica exchange ---------------------------------------
            if sweep % swap_interval == 0:
                # Transfer GPU replicas to CPU list for energy computation
                if use_gpu_batch:
                    replicas = [_to_numpy(batch[r]).copy() for r in range(n_replicas)]

                # Even/odd alternating swap pattern
                start = (sweep // swap_interval) % 2
                for r in range(start, n_replicas - 1, 2):
                    E_r = self._total_energy(replicas[r])
                    E_r1 = self._total_energy(replicas[r + 1])
                    delta = (beta_arr[r] - beta_arr[r + 1]) * (E_r - E_r1)
                    if delta >= 0 or self.rng.random() < np.exp(delta):
                        replicas[r], replicas[r + 1] = replicas[r + 1], replicas[r]

                # Push swapped replicas back to GPU
                if use_gpu_batch:
                    batch = xp.asarray(np.stack(replicas, axis=0))

        # ---- 3. Collect observables from final configurations -------------
        if use_gpu_batch:
            replicas = [_to_numpy(batch[r]).copy() for r in range(n_replicas)]

        h_res = np.full(n_replicas, h)
        b_res = beta_arr.copy()
        e_res = np.empty(n_replicas)
        c_res = np.zeros(n_replicas)
        m_res = np.empty(n_replicas)
        x_res = np.zeros(n_replicas)

        for r in range(n_replicas):
            e_res[r] = self._total_energy(replicas[r]) / self.size
            m_res[r] = np.sum(replicas[r]) / self.size

        # ---- 4. Return DataFrame ------------------------------------------
        return pd.DataFrame(
            np.vstack((h_res, b_res, e_res, c_res, m_res, x_res)).T,
            columns=['ExternalMagneticField', 'ThermodynamicBeta',
                     'Energy', 'SpecificHeatCapacity',
                     'Magnetization', 'MagneticSusceptibility'],
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(self, results: pd.DataFrame) -> str:
        """Save a three-panel scatter plot of simulation results and return the file path."""
        results = results.copy()
        results['NetMagnetization'] = np.abs(results['Magnetization'])
        results['ScaledTemperature'] = 1.0 / (KB * T_CRIT * results['ThermodynamicBeta'])

        os.makedirs('results', exist_ok=True)
        file_name = os.path.join('results', f'pyising_{self.__repr__()}.png')

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, layout='constrained', figsize=(19.2, 10.8))
        fig.suptitle('PyIsing Simulation Results\n' + self.__repr__())
        fig.supxlabel(r'$T/T_c$')
        ax1.scatter(results['ScaledTemperature'], results['Energy'],
                    color='darkgreen', label=r'$\left<E\right>$')
        ax1.set_ylabel(r'Energy, $\left<E\right>$')
        ax2.scatter(results['ScaledTemperature'], results['NetMagnetization'],
                    color='darkorange', label=r'$\left|\left<M\right>\right|$')
        ax2.set_ylabel(r'Net Magnetization, $\left|\left<M\right>\right|$')
        ax3.scatter(results['ScaledTemperature'], results['SpecificHeatCapacity'],
                    color='darkred', label=r'$\left<C\right>$')
        ax3.set_ylabel(r'Specific Heat Capacity, $\left<C\right>$')
        for ax in (ax1, ax2, ax3):
            ax.legend()
            ax.grid()
        fig.savefig(file_name)
        plt.close(fig)
        return file_name

    def save_video(self) -> str:
        """Render all frames to an AVI video and return the file path."""
        if not _CV2_AVAILABLE:
            raise RuntimeError('opencv-python (cv2) is required for save_video(). '
                               'Install it with: pip install opencv-python')

        frames_dir = os.path.join(os.getcwd(), 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs('results', exist_ok=True)

        for _ in trange(self.frames, desc='Gathering Frames'):
            frame_np = _to_numpy(self.at[_])
            fname = os.path.join(frames_dir, f'pyising_{self.__str__()}_frame_{str(_).zfill(4)}.png')
            plt.imsave(fname, frame_np, vmin=-1, vmax=1, cmap='coolwarm')

        file_name = os.path.join('results', f'pyising_{self.__str__()}.avi')
        images = sorted(img for img in os.listdir(frames_dir) if img.endswith('.png'))
        titleframe = cv2.imread(os.path.join(frames_dir, images[0]))
        height, width, _ = titleframe.shape
        video = cv2.VideoWriter(file_name, 0, self.fps, (width, height))
        for _, image in tqdm(enumerate(images), desc='Writing Video', total=self.frames):
            video.write(cv2.imread(os.path.join(frames_dir, image)))
        cv2.destroyAllWindows()
        video.release()

        for _, img in tqdm(enumerate(images), desc='Releasing Frames', total=self.frames):
            os.remove(os.path.join(frames_dir, img))

        return file_name

    def visualize(self, h: Optional[float] = None, beta: Optional[float] = None,
                  method: Optional[str] = None) -> str:
        """Quench the system and save a video. Returns the video file path."""
        if beta is not None:
            self.beta = beta
            self.T = 1.0 / (KB * self.beta)
        if h is not None:
            self.h = h
        if method is not None:
            self.method = method
        self.quench(beta=self.beta, h=self.h, method=self.method)
        return self.save_video()
