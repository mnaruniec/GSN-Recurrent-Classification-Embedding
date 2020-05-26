import torch

from torch.nn.utils import rnn as padding
from torch.utils.data import DataLoader, TensorDataset

from constants import *


class PreprocessDataLoader(DataLoader):
    def __iter__(self):
        batches = super().__iter__()
        for b in batches:
            yield self.preprocess(*b)

    def preprocess(self, x, lens, y):
        return x, lens, y


class RNNDataLoader(PreprocessDataLoader):
    def preprocess(self, x, lens, y):
        order, _ = zip(*sorted(enumerate(lens), key=lambda e: e[1], reverse=True))
        order = list(order)

        new_x = x[order].float().to(DEVICE)
        new_lens = lens[order]
        packed = padding.pack_padded_sequence(new_x, new_lens, batch_first=True)

        return packed, y[order].to(DEVICE)


def load_file(path: str, is_labels: bool) -> torch.Tensor:
    with open(path, mode='r') as f:
        if is_labels:
            return torch.tensor([int(line) for line in f])

        splits = torch.tensor([[
                [int(i) for i in col.split('-')]
                for col in line.split(',')
            ]
            for line in f
        ])

    return splits


def get_dataloader(
        x_path: str,
        y_path: str,
        shuffle=True,
        drop_last=True,
        mb_size=DEFAULT_MB_SIZE,
        truncation_p=0.,
        truncation_amount=DEFAULT_TRUNCATION_AMOUNT
) -> RNNDataLoader:
    xs = load_file(x_path, is_labels=False)
    ys = load_file(y_path, is_labels=True)

    lens = torch.tensor(xs.shape[0] * [xs.shape[1]])

    if truncation_p > 0.:
        rand = torch.rand(len(lens))
        for i in range(len(lens)):
            if rand[i] < truncation_p:
                lens[i] -= truncation_amount

    ds = TensorDataset(xs, lens, ys)
    dl = RNNDataLoader(ds, batch_size=mb_size, shuffle=shuffle, drop_last=drop_last, pin_memory=True)
    return dl


def get_dataloaders(truncation_p=0.):
    train_dl = get_dataloader(
        TRAIN_X_PATH, TRAIN_Y_PATH, shuffle=True, drop_last=True, truncation_p=truncation_p
    )
    test_dl = get_dataloader(TEST_X_PATH, TEST_Y_PATH, shuffle=False, drop_last=False, truncation_p=truncation_p)
    valid_dl = test_dl

    return train_dl, valid_dl, test_dl
