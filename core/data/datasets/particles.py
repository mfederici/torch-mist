from typing import Optional
import os
import numpy as np

from torch.utils.data import Dataset

PARTICLES_FILE = 'particles.npy'
IID_CLUSTER_FILE = 'iid_clusters.npy'
NDIM = 2
PARTICLE_MI = 2.112134


class ParticleTrajectories(Dataset):
    def __init__(self, data_dir: str, n_particles: int = 5, attribute_file: Optional[str] = None):
        super().__init__()
        with open(os.path.join(data_dir, PARTICLES_FILE), 'rb') as file:
            self.trajectories = np.load(file)
        if attribute_file is None:
            self.attributes = None
        else:
            with open(os.path.join(data_dir, attribute_file), 'rb') as file:
                self.attributes = np.load(file)
        self.n_particles = n_particles

    def __getitem__(self, idx: int):
        data = dict()
        data['x'] = self.trajectories[idx, :self.n_particles*NDIM]
        data['y'] = self.trajectories[idx+1, :self.n_particles*NDIM]
        if self.attributes is not None:
            data['a'] = self.attributes[idx]

        data['idx'] = idx

        return data

    def __len__(self):
        return self.trajectories.shape[0]-1
