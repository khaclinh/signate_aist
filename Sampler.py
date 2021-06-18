import torch
from torch.utils.data.sampler import Sampler
from numpy.random import shuffle


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


class ImbalancedDatasetSampler(Sampler):
    def __init__(self, sampler, indices: list, batch_size: int, ratio) -> None:
        self.sampler = sampler
        self.indices = indices
        self.batch_size = batch_size

        # get the label of each index
        data = {0: [], 1: []}
        labels = [self.sampler.get_label(idx) for idx in self.indices]
        # split to 2 label lists
        for idx in range(len(self.indices)):
            data[labels[idx]].append(idx)
        shuffle(data[0])
        shuffle(data[1])

        # split data to batches
        self.combined = []
        pick_idx = {0: 0, 1: 0}
        min_number = round(self.batch_size * ratio)
        maj_number = self.batch_size - min_number

        for batch in range(len(self.indices) // self.batch_size):
            batch_samples = []

            for i in range(maj_number):
                batch_samples.append(data[0][pick_idx[0]])
                pick_idx[0] += 1
                if pick_idx[0] == len(data[0]):
                    pick_idx[0] = 0
                    shuffle(data[0])

            for j in range(min_number):
                batch_samples.append(data[1][pick_idx[1]])
                pick_idx[1] += 1
                if pick_idx[1] == len(data[1]):
                    pick_idx[1] = 0
                    shuffle(data[1])

            shuffle(batch_samples)
            self.combined.extend(batch_samples)

    def __iter__(self):
        return iter(self.combined)

    def __len__(self):
        return len(self.indices)