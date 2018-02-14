import os
import subprocess
from tempfile import NamedTemporaryFile
from torch.utils.data.sampler import Sampler

import pdb

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class FeatDataset(Dataset):
    def __init__(self, manifest_filepath, maxval=400.0):
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.maxval = maxval
        super(FeatDataset, self).__init__()

    def __getitem__(self, index):
        sample = self.ids[index]
        feat_path, id = sample[0], sample[1]

        feat = np.load(feat_path, encoding="latin1")
        feat = feat.item()
        feat = feat['data_GP_smoothed_trial']
        feat = feat[:, :2]
        feat = torch.FloatTensor(feat)/self.maxval

        return feat, id

    def __len__(self):
        return self.size


def _collate_fn_feat(batch):

    nFrame = batch[0][0].size(0)
    nAxis = batch[0][0].size(1)
    minibatch_size = len(batch)
    input = torch.zeros(minibatch_size, nFrame, nAxis) # NxTx2
    target = torch.LongTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]

        tensor = sample[0]
        id = sample[1]

        input[x] = tensor
        target[x] = int(id)-1 # idx starts from 0

    return input, target



class FeatLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(FeatLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_feat


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        np.random.shuffle(self.bins)

