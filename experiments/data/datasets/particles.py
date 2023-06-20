from typing import Optional
import os
import numpy as np

from torch.utils.data import Dataset

PARTICLES_FILE = 'particles.npy'
CORRELATED_PARTICLES_FILE = 'correlated_particles.npy'
IID_CLUSTER_FILE = 'iid_clusters.npy'
CLUSTER_FILE = 'clusters.npy'
CORRELATED_CLUSTER_FILE = 'correlated_clusters.npy'

NDIM = 2
PARTICLE_MI = 2.112134
PARTICLE_H = 2.562166


class ParticleTrajectories(Dataset):
    def __init__(self,
                 data_dir: str,
                 n_particles: int = 5,
                 traj_file: Optional[str] = None,
                 attribute_file: Optional[str] = None,
                 split: str = 'train'
                 ):
        super().__init__()

        self.split = split

        if self.split == 'train':
            ids = np.arange(80000)
        elif self.split == 'val':
            ids = np.arange(80000, 90000)
        elif self.split == 'test':
            ids = np.arange(90000, 100000)
        elif self.split == 'train+val':
            ids = np.arange(90000)
        elif self.split == 'all':
            ids = np.arange(100000)
        else:
            raise ValueError(f'Unknown split {self.split}')

        if traj_file is None or traj_file == PARTICLES_FILE:
            traj_file = PARTICLES_FILE
            self.total_dims = n_particles*NDIM
        else:
            assert traj_file == CORRELATED_PARTICLES_FILE
            traj_file = CORRELATED_PARTICLES_FILE
            self.total_dims = None

        with open(os.path.join(data_dir, traj_file), 'rb') as file:
            trajectories = np.load(file)

        self.trajectories = trajectories[ids, :self.total_dims]

        if attribute_file is None:
            self.attributes = None
        else:
            with open(os.path.join(data_dir, attribute_file), 'rb') as file:
                attributes = np.load(file)
            self.attributes = attributes[ids[:-1]]
            assert len(self.attributes) == len(self)

        if self.total_dims is None:
            self.total_dims = self.trajectories.shape[1]

    def __getitem__(self, idx: int):
        data = dict()
        data['x'] = self.trajectories[idx, :self.total_dims]
        data['y'] = self.trajectories[idx+1, :self.total_dims]
        if self.attributes is not None:
            data['a'] = self.attributes[idx]

        data['idx'] = idx

        return data

    def __len__(self):
        return self.trajectories.shape[0]-1
