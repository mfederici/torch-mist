from torchvision.datasets.omniglot import Omniglot, list_dir, list_files
import numpy as np
from os.path import join
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.utils import make_grid
from PIL import Image


class MultiOmniglot(Omniglot):
    def __init__(self, *args, n_images: int = 1, n_samples: int = 50000, split="train", **kwargs):
        super().__init__(*args, **kwargs, transform=None)
        self.n_images = n_images
        self.n_samples = n_samples


        lookup = {}
        n_characters = {}
        for alphabet in self._alphabets:
            lookup[alphabet] = {}
            n_characters[alphabet] = 0
            for character in list_dir(join(self.target_folder, alphabet)):
                c_id = int(character.replace('character', ''))
                lookup[alphabet][c_id] = []
                n_characters[alphabet] += 1
                for version in sorted(list_files(join(self.target_folder, alphabet, character), ".png")):
                    lookup[alphabet][c_id].append(version)

        self.used_alphabets = []
        for alphabet, _ in sorted(n_characters.items(), key=lambda item: -item[1]):
            self.used_alphabets.append(alphabet)
            if len(self.used_alphabets) == self.n_images**2:
                break
        self.lookup = lookup
        self.n_characters = n_characters
        self.split = split

        if split == "train":
            versions = np.arange(16)
        elif split == "val":
            versions = np.arange(2)+16
        elif split == "train+val":
            versions = np.arange(18)
        elif split == "test":
            versions = np.arange(2)+18
        else:
            raise Exception("Only train, val, train+val and test splits are available")

        self.s_characters = {
            alphabet: np.random.choice(
                n_characters[alphabet],
                n_samples,
                replace=True
            ) for alphabet in self.used_alphabets
        }
        self.s_versions_x = {
            alphabet: np.random.choice(
                versions,
                n_samples,
                replace=True
            ) for alphabet in self.used_alphabets
        }
        self.s_versions_y = {
            alphabet: np.random.choice(
                versions,
                n_samples,
                replace=True
            ) for alphabet in self.used_alphabets
        }
        self.transform = Compose([
            Resize(28),
            ToTensor()
        ])

    def get_image(self, alphabet, character, version):
        filename = self.lookup[alphabet][character][version]
        full_path = join(self.target_folder, alphabet, "character{:02d}".format(character), filename)
        image = Image.open(full_path, mode="r").convert("L")
        return image

    def __getitem__(self, idx):
        data = {'x': [], 'y': [], 't': []}

        for alphabet in self.used_alphabets:
            character = self.s_characters[alphabet][idx]+1
            version_x = self.s_versions_x[alphabet][idx]
            img_x = self.get_image(alphabet, character, version_x)
            img_x = self.transform(img_x)
            data['x'].append(img_x)
            data['t'].append(character)

            character_y = 1 + (character % self.n_characters[alphabet])
            version_y = self.s_versions_y[alphabet][idx]
            img_y = self.get_image(alphabet, character_y, version_y)
            img_y = self.transform(img_y)
            data['y'].append(img_y)

        data['x'] = make_grid(data['x'], self.n_images, padding=0)[0].unsqueeze(0)
        data['y'] = make_grid(data['y'], self.n_images, padding=0)[0].unsqueeze(0)
        data['idx'] = idx

        return data

    def __len__(self):
        return self.n_samples
