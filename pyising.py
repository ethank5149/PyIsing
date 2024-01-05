import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import os, cv2


KB = 1
T_CRIT = 2 / np.log(1 + np.sqrt(2))
BETA_CRIT = 1 / (KB * T_CRIT)


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
            beta_range=np.linspace(0.5 * BETA_CRIT, 10 * BETA_CRIT, 100)
            ):
        
        self.rng = np.random.default_rng()
        self.nrows, self.ncols = nrows, ncols
        self.size, self.scale = nrows * ncols, 1 / (nrows * ncols)
        self.J, self.mu = J, mu
        self.beta, self.h = beta, h
        self.T = 1 / (KB * self.beta)

        self.tf, self.fps = tf, fps
        self.frames = self.tf * self.fps
        self.sample_size, self.method = sample_size, method
        self.beta_range, self.h_range = beta_range, h_range
        
        self.specific_heat_capacity, self.magnetic_susceptibility = np.zeros(self.frames), np.zeros(self.frames)
        self.energy, self.magnetization = np.zeros(self.frames), np.zeros(self.frames)
        self.energy_sqr, self.magnetization_sqr = np.zeros(self.frames), np.zeros(self.frames)

        self.at = np.zeros(shape=(self.frames, self.nrows, self.ncols), dtype=np.int8)
        self.at[0, :, :] = 2 * self.rng.integers(low=0, high=2, size=(self.nrows, self.ncols)) - 1

    def __repr__(self):
        return f'{self.ncols}x{self.nrows}_{self.sample_size}samples_{self.tf * self.fps}iters_nbeta={len(self.beta_range)}_nh={len(self.h_range)}_{self.method}_alg'
        # return f'{self.ncols}x{self.nrows}_{self.sample_size}samples_{self.tf * self.fps}iters_{self.method}_alg'
    
    def __str__(self):
        return f'{self.ncols}x{self.nrows}_tf={self.tf}_fps={self.fps}_b={self.beta:0.4f}_h={self.h:0.4f}_{self.method}_alg'

    def reset(self):
        self.at[0, :, :] = 2 * self.rng.integers(low=0, high=2, size=(self.nrows, self.ncols)) - 1
        return None

    def metric(self, t):
        return np.roll(self.at[t, :, :], -1, 0) + \
            np.roll(self.at[t, :, :], 1, 0) + \
                np.roll(self.at[t, :, :], -1, 1) + \
                    np.roll(self.at[t, :, :], 1, 1)

    def metric_at(self, t, i, j):
        return self.at[t,  (i + 1)               % self.nrows,   j                                ] + \
               self.at[t, ((i - 1) + self.nrows) % self.nrows,   j                                ] + \
               self.at[t,   i                                ,  (j + 1)               % self.ncols] + \
               self.at[t,   i                                , ((j - 1) + self.ncols) % self.ncols]

    def neighbors_at(self, t, i, j):
        return [
            ((i + 1) % self.nrows, j),
            (((i - 1) + self.nrows) % self.nrows, j),
            (i, (j + 1) % self.ncols),
            (i, ((j - 1) + self.ncols) % self.ncols)
        ]

    def dE(self, t, i, j):
        return 0.5 * (self.J * self.metric_at(t, i, j) - self.mu * self.h) * self.at[t, i, j]

    def E(self, t):
        return np.sum(np.asarray([[self.dE(t, i, j) for i in range(self.nrows)] for j in range(self.ncols)])) / self.size

    def E_sqr(self, t):
        return np.sum(np.asarray([[self.dE(t, i, j) ** 2 for i in range(self.nrows)] for j in range(self.ncols)])) / self.size

    def M(self, t):
        return np.sum(np.asarray([[self.at[t, i, j] for i in range(self.nrows)] for j in range(self.ncols)])) / self.size

    def M_sqr(self, t):
        return np.sum(np.asarray([[self.at[t, i, j] ** 2 for i in range(self.nrows)] for j in range(self.ncols)])) / self.size

    def C(self, t):
        # if self.energy is not None and self.energy_sqr is not None:
        #     return self.beta * (self.energy_sqr[t] - self.energy[t] ** 2) / self.T
        # else:
        #     return self.beta * (self.E_sqr(t) - np.power(self.E(t), 2)) / self.T
        return self.beta * (self.energy_sqr[t] - self.energy[t] ** 2) / self.T

    def X(self, t):
        # if self.magnetization is not None and self.magnetization_sqr is not None:
        #     return self.beta * (self.magnetization_sqr[t] - self.magnetization[t] ** 2)
        # else:
        #     return self.beta * (self.M_sqr(t) - np.power(self.M(t), 2))
        return self.beta * (self.magnetization_sqr[t] - self.magnetization[t] ** 2)

    def update_metropolis_hastings(self, t):
        i_s = self.rng.integers(low=0, high=self.nrows, size=self.size)
        j_s = self.rng.integers(low=0, high=self.ncols, size=self.size)
        self.at[t + 1, :, :] = self.at[t, :, :]  # Copy last state onto next frame
        # Now use next frame
        for obv in range(self.size):
            i, j = i_s[obv], j_s[obv]
            dE = self.dE(t + 1, i, j)
            p = 1. if dE < 0.0 else np.exp(-dE * self.beta)
            if self.rng.random() < p:
                self.at[t + 1, i, j] = -self.at[t + 1, i, j]
        return None

    def update_wolff(self, t):
        self.at[t + 1, :, :] = self.at[t, :, :]  # Copy last state onto next frame
        # Now use next frame
        p = 1 - np.exp(-2 * self.beta * self.J)
        seed_i = self.rng.integers(low=0, high=self.nrows)
        seed_j = self.rng.integers(low=0, high=self.ncols)
        cluster = [(seed_i, seed_j),]
        perimeter = [cluster[0]]
        
        while len(perimeter) > 0:
            seed_spin = perimeter[self.rng.integers(low=0, high=len(perimeter))]
            perimeter.remove(seed_spin)
        
            for spin in self.neighbors_at(t + 1, *seed_spin):
                if self.at[t + 1, *spin] == self.at[t + 1, *seed_spin] and \
                    spin not in cluster and \
                        self.rng.random() < p:
                    cluster.append(spin)
                    perimeter.append(spin)
        
        for spin in cluster:
            self.at[t + 1, *spin] = -self.at[t + 1, *spin]

        return len(cluster) / self.size

    def quench(self, beta=None, h=None, method=None, verbose=True):
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
            self.T = 1 / (KB * self.beta)
        if h is None:
            h = self.h
        else:
            self.h = h
        if method is None:
            method = self.method
        else:
            self.method = method

        # To prevent unnecessary checks in a loop, we duplicate code
        if self.method == 'wolff':
            for _ in trange(self.frames - 1, desc='Quenching System', leave=False) if verbose else range(self.frames - 1):
                self.update_wolff(_)
        else:
            for _ in trange(self.frames - 1, desc='Quenching System', leave=False) if verbose else range(self.frames - 1):
                self.update_metropolis_hastings(_)
    
    def gather_data(self, verbose=True):
        for _ in trange(self.frames, desc='Gathering Data', leave=False) if verbose else range(self.frames):
            self.energy[_] = self.E(_)
            self.energy_sqr[_] = self.E_sqr(_)
            self.magnetization[_] = self.M(_)
            self.magnetization_sqr[_] = self.M_sqr(_)
            self.specific_heat_capacity[_] = self.C(_)
            self.magnetic_susceptibility[_] = self.X(_)
        return None

    def save_results(self, results):
        results['NetMagnetization'] = np.abs(results['Magnetization'])
        results['ScaledTemperature'] = np.power(KB * T_CRIT * results['ThermodynamicBeta'], -1)
        
        file_name = f'results/pyising_{self.__repr__()}.png'
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, layout="constrained", figsize=(19.2, 10.8))
        fig.suptitle('PyIsing Simulation Results' + '\n' +  self.__repr__())
        fig.supxlabel(r'$T/T_c$')
        net_energy_plot = ax1.scatter(results['ScaledTemperature'], results['Energy'], color='darkgreen', label=r'$\left<E\right>$')
        ax1.set_ylabel(r'Energy, $\left<E\right>$')
        net_magnetization_plot = ax2.scatter(results['ScaledTemperature'], results['NetMagnetization'], color='darkorange', label=r'$\left|\left<M\right>\right|$')
        ax2.set_ylabel(r'Net Magnetization, $\left|\left<M\right>\right|$')
        specific_heat_capacity_plot = ax3.scatter(results['ScaledTemperature'], results['SpecificHeatCapacity'], color='darkred', label=r'$\left<C\right>$')
        ax3.set_ylabel(r'Specific Heat Capacity, $\left<C\right>$')
        ax1.legend()
        ax1.grid()
        ax2.legend()
        ax2.grid()
        ax3.legend()
        ax3.grid()
        fig.savefig(file_name)
        return file_name

    def save_video(self):
        # Write Frames
        for _ in trange(self.frames, desc='Gathering Frames'):
            file_name = f"./frames/pyising_{self.__str__()}_frame_{str(_).zfill(4)}.png"
            plt.imsave(
                file_name,
                self.at[_, :, :], 
                vmin=-1, vmax=1, 
                cmap='coolwarm')
        
        # Write Video
        file_dir = os.path.join(os.getcwd(), ".\\frames")  # Windows
        file_name = f"results/pyising_{self.__str__()}.avi"
        images = [img for img in os.listdir(file_dir) if img.endswith(".png")]
        titleframe = cv2.imread(os.path.join(file_dir, images[0])) 
        height, width, _ = titleframe.shape
        video = cv2.VideoWriter(file_name, 0, self.fps, (width, height))
        for _,image in tqdm(enumerate(images), desc='Writing Video', total=self.frames):
            video.write(cv2.imread(os.path.join(file_dir, image)))  

        cv2.destroyAllWindows()  
        video.release()

        # Delete Unneeded Frames
        for _,img in tqdm(enumerate(images), desc='Releasing Frames', total=self.frames):
            os.remove(os.path.join(file_dir, img))

        return file_name
    
    def visualize(self, h=None, beta=None, method=None):
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
            self.T = 1 / (KB * self.beta)
        if h is None:
            h = self.h
        else:
            self.h = h
        if method is None:
            method = self.method
        else:
            self.method = method
        
        self.quench(beta=beta, h=h, method=method)
        self.save_video()

    def simulate(self, beta_range, h_range, sample_size=30, method='wolff'):
        self.sample_size = sample_size

        energy_sample, specific_heat_capacity_sample = np.zeros(sample_size), np.zeros(sample_size)
        magnetization_sample, magnetic_susceptibility_sample = np.zeros(sample_size), np.zeros(sample_size)

        data_size = len(beta_range) * len(h_range)
        h_results, beta_results = np.zeros(data_size), np.zeros(data_size)
        energy_results, specific_heat_capacity_results = np.zeros(data_size), np.zeros(data_size)
        magnetization_results, magnetic_susceptibility_results = np.zeros(data_size), np.zeros(data_size)

        for idx_h,h in tqdm(enumerate(h_range), desc='Sweeping H', total=len(h_range), position=0, leave=False):
            for idx_beta,beta in tqdm(enumerate(beta_range), desc='Sweeping Beta', total=len(beta_range), position=1, leave=False):
                for idx_sample in tqdm(range(sample_size), desc='Sampling Parameters', total=sample_size, position=2, leave=False):
                    idx = idx_h * len(beta_range) + idx_beta

                    self.reset()
                    self.quench(beta=beta, h=h, method=method)  #, verbose=False)
                    self.gather_data()  # verbose=False)

                    energy_sample[idx_sample] = self.energy[-1]
                    specific_heat_capacity_sample[idx_sample] = self.specific_heat_capacity[-1]
                    magnetization_sample[idx_sample] = self.magnetization[-1]
                    magnetic_susceptibility_sample[idx_sample] = self.magnetic_susceptibility[-1]

                beta_results[idx] = beta
                h_results[idx] = h
                energy_results[idx] = np.mean(energy_sample)
                specific_heat_capacity_results[idx] = np.mean(specific_heat_capacity_sample)
                magnetization_results[idx] = np.mean(magnetization_sample)
                magnetic_susceptibility_results[idx] = np.mean(magnetic_susceptibility_sample)

        return pd.DataFrame(
            np.vstack(
                (h_results, beta_results, energy_results, specific_heat_capacity_results, magnetization_results, magnetic_susceptibility_results)).T, 
                columns=['ExternalMagneticField','ThermodynamicBeta', 'Energy', 'SpecificHeatCapacity', 'Magnetization', 'MagneticSusceptibility'])