import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
import os, cv2


class IsingModel:
    def __init__(
            self,
            tf=30,
            fps=30,
            ncols=854, 
            nrows=480,
            J=1., 
            mu=1,
            kB=1.,
            beta=10,
            h=0,
            method='metropolis-hastings'
            ):
        
        self.rng = np.random.default_rng()
        self.nrows, self.ncols = nrows, ncols
        self.size, self.scale = nrows * ncols, 1 / (nrows * ncols)
        self.J, self.mu, self.kB = J, mu, kB
        self.beta, self.h = beta, h
        self.T = 1 / (self.kB * self.beta)

        self.tf, self.fps = tf, fps
        self.frames = self.tf * self.fps
        self.method = method
        self.id = f"{self.ncols}x{self.nrows}_b={self.beta:0.4f}_h={self.h:0.4f}_tf={self.tf}_fps={self.fps}_method={self.method}"

        self.specific_heat_capacity, self.magnetic_susceptibility = np.zeros(self.frames), np.zeros(self.frames)
        self.energy, self.magnetization,  = np.zeros(self.frames), np.zeros(self.frames)
        self.energy_sqr, self.magnetization_sqr = np.zeros(self.frames), np.zeros(self.frames)
        self.dEdt = np.zeros(self.frames)

        self.at = np.zeros(shape=(self.frames, self.nrows, self.ncols), dtype=np.int8)
        self.at[0, :, :] = 2 * self.rng.integers(low=0, high=2, size=(self.nrows, self.ncols)) - 1

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

    def update(self, t):
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
                    # self.at[t + 1, *spin] = -self.at[t + 1, *spin]
        
        for spin in cluster:
            self.at[t + 1, *spin] = -self.at[t + 1, *spin]

        return len(cluster) / self.size

    def quench(self, iterations=None, beta=None, h=None, method=None):
        if iterations is None:
            iterations = self.frames
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
            self.T = 1 / (self.kB * self.beta)
        if h is None:
            h = self.h
        else:
            self.h = h
        if method is None:
            method = self.method
        else:
            self.method = method

        # Update id
        self.id = f"{self.ncols}x{self.nrows}_b={self.beta:0.4f}_h={self.h:0.4f}_tf={self.tf}_fps={self.fps}_method={self.method}"

        file_name = f"./frames/pyising_{self.id}_frame_{str(0).zfill(4)}.png"
        plt.imsave(file_name, self.at[0, :, :], vmin=-1, vmax=1, cmap='coolwarm')

        # To prevent unnecessary checks in a loop, we duplicate code
        if self.method == 'wolff':
            for _ in trange(iterations - 1, desc='Quenching System'):
                cluster_size = self.update_wolff(_)
                file_name = f"./frames/pyising_{self.id}_frame_{str(_ + 1).zfill(4)}.png"
                plt.imsave(file_name, self.at[_ + 1, :, :], vmin=-1, vmax=1, cmap='coolwarm')
                if cluster_size > 0.75:  # DEBUG
                    print("Warning: Potential Oscillatory Solution in Wolff Algorithm")
                    self.frames = _ + 1
                    self.tf = self.frames / self.fps
                    self.energy = self.energy[:self.frames]
                    self.energy_sqr = self.energy_sqr[:self.frames]
                    self.magnetization = self.magnetization[:self.frames]
                    self.magnetization_sqr = self.magnetization_sqr[:self.frames]
                    self.specific_heat_capacity = self.specific_heat_capacity[:self.frames]
                    self.magnetic_susceptibility = self.magnetic_susceptibility[:self.frames]
                    break
        else:
            for _ in trange(iterations - 1, desc='Quenching System'):
                self.update(_)
                file_name = f"./frames/pyising_{self.id}_frame_{str(_ + 1).zfill(4)}.png"
                plt.imsave(file_name, self.at[_ + 1, :, :], vmin=-1, vmax=1, cmap='coolwarm')
    
    def dE(self, t, i, j):
        # return 2 * (self.J * self.metric_at(t, i, j) - self.mu * self.h) * self.at[t, i, j]
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
        if self.energy is not None and self.energy_sqr is not None:
            return self.beta * (self.energy_sqr[t] - self.energy[t] ** 2) / self.T
        else:
            return self.beta * (self.E_sqr(t) - np.power(self.E(t), 2)) / self.T

    def X(self, t):
        if self.magnetization is not None and self.magnetization_sqr is not None:
            return self.beta * (self.magnetization_sqr[t] - self.magnetization[t] ** 2)
        else:
            return self.beta * (self.M_sqr(t) - np.power(self.M(t), 2))

    def gather_data(self):
        for _ in trange(self.frames, desc='Gathering Data'):
            self.energy[_] = self.E(_)
            self.energy_sqr[_] = self.E_sqr(_)
            self.magnetization[_] = self.M(_)
            self.magnetization_sqr[_] = self.M_sqr(_)
            self.specific_heat_capacity[_] = self.C(_)
            self.magnetic_susceptibility[_] = self.X(_)

        self.dEdt = np.diff(self.energy)
        return None

    def save_results(self):
        self.gather_data()
        t, file_name = np.linspace(0, self.tf, self.frames), f'results/pyising_{self.id}.png'
        fig, (energy_ax, magnetization_ax, diff_ax) = plt.subplots(3, 1, sharex=True, layout="constrained", figsize=(19.2, 10.8))
        fig.suptitle('PyIsing Simulation Results' + 
                     '\n' +  # self.id 
                     rf'$\beta={self.beta:0.4f},h={self.h:0.4f},t_f={self.tf:0.4f},\text{{fps}}={self.fps},${self.method} algorithm')
        fig.supxlabel(r'$t$')

        energy_plot = energy_ax.plot(t, self.energy, color='darkgreen', label=r'$\left<E\right>$')
        energy_ax.set_ylabel(r'Energy, $\left<E\right>$')
        specific_heat_capacity_ax = energy_ax.twinx()
        specific_heat_capacity_ax.set_ylabel(r'Specific Heat Capacity, $\left<C\right>$')
        specific_heat_capacity_plot = specific_heat_capacity_ax.plot(t, self.specific_heat_capacity, color='darkred', label=r'$\left<C\right>$')

        plots1 = energy_plot + specific_heat_capacity_plot
        labels1 = [l.get_label() for l in plots1]
        energy_ax.legend(plots1, labels1)
        energy_ax.grid()

        magnetization_plot = magnetization_ax.plot(t, self.magnetization, color='darkorange', label=r'$\left<M\right>$')
        magnetization_ax.set_ylabel(r'Magnetization, $\left<M\right>$')
        magnetic_susceptibility_ax = magnetization_ax.twinx()
        magnetic_susceptibility_ax.set_ylabel(r'Magnetic Susceptibility, $\left<\chi\right>$')
        magnetic_susceptibility_plot = magnetic_susceptibility_ax.plot(t, self.magnetic_susceptibility, color='darkblue', label=r'$\left<\chi\right>$')

        plots2 = magnetization_plot + magnetic_susceptibility_plot
        labels2 = [l.get_label() for l in plots2]
        magnetization_ax.legend(plots2, labels2)
        magnetization_ax.grid()

        diff_plot = diff_ax.plot(t[:-1], self.dEdt, color='purple', label=r'$\frac{d\left<E\right>}{dt}$')
        diff_ax.set_ylabel(r'Time Derivative of Energy, $\frac{d\left<E\right>}{dt}$')
        diff_ax.legend()
        diff_ax.grid()

        fig.savefig(file_name)
        return file_name

    def save_video(self):
        # Write Frames
        for _ in trange(self.frames, desc='Gathering Frames'):
            file_name = f"./frames/pyising_{self.id}_frame_{str(_).zfill(4)}.png"
            plt.imsave(
                file_name,
                self.at[_, :, :], 
                vmin=-1, vmax=1, 
                cmap='coolwarm')
        
        # Write Video
        file_dir = os.path.join(os.getcwd(), ".\\frames")  # Windows
        file_name = f"results/pyising_{self.id}.avi"
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


    # def simulate(self):
    #     for i_h, H in enumerate(tqdm(self.Hrange, desc='Sweeping Magnetic Fields', leave=False)):
    #         for i_t, T in enumerate(tqdm(self.Trange, desc='Sweeping Temperatures', leave=False)):
    #             line = i_h * self.nT + i_t                
    #             self.reset()
    #             self.relax(H, T)
    #             self.Hs[line], self.Ts[line], self.Es[line], self.Ms[line], self.Cs[line], self.Xs[line] = self.measure_at(H, T)
    #     return pd.DataFrame(np.vstack((self.Hs, self.Ts, self.Es, self.Cs, self.Ms, self.Xs)).T, 
    #                         columns=['MagneticField','Temperature', 'Energy', 'SpecificHeatCapacity', 'Magnetization', 'MagneticSusceptibility'])