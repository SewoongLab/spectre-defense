import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
from typing import Callable, Iterable, Tuple
from pathlib import Path


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


CIFAR_PATH = Path("./data_cifar10")
CIFAR_TRANSFORM_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_TRANSFORM_NORMALIZE_STD = (0.2023, 0.1994, 0.2010)
CIFAR_TRANSFORM_NORMALIZE = transforms.Normalize(
    CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD
)
CIFAR_TRANSFORM_NORMALIZE_INV = NormalizeInverse(
    CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD
)
CIFAR_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)
CIFAR_TRANSFORM_TRAIN_XY = lambda xy: (CIFAR_TRANSFORM_TRAIN(xy[0]), xy[1])

CIFAR_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)
CIFAR_TRANSFORM_TEST_XY = lambda xy: (CIFAR_TRANSFORM_TEST(xy[0]), xy[1])


class LabelSortedDataset(ConcatDataset):
    def __init__(self, dataset: Dataset):
        self.orig_dataset = dataset
        self.by_label = {}
        for i, (_, y) in enumerate(dataset):
            self.by_label.setdefault(y, []).append(i)

        self.n = len(self.by_label)
        assert set(self.by_label.keys()) == set(range(self.n))
        self.by_label = [Subset(dataset, self.by_label[i]) for i in range(self.n)]
        super().__init__(self.by_label)

    def subset(self, labels: Iterable[int]) -> ConcatDataset:
        if isinstance(labels, int):
            labels = [labels]
        return ConcatDataset([self.by_label[i] for i in labels])


class FilterDataset(Subset):
    def __init__(self, dataset: Dataset, *, label: int):
        indices = []
        for i, (_, y) in enumerate(dataset):
            if y == label:
                indices.append(i)
        super().__init__(dataset, indices)


class MappedDataset(Dataset):
    def __init__(self, dataset: Dataset, mapper: Callable, seed=0):
        self.dataset = dataset
        self.mapper = mapper
        self.seed = seed

    def __getitem__(self, i: int):
        if hasattr(self.mapper, 'seed'):
            self.mapper.seed(i + self.seed)
        return self.mapper(self.dataset[i])

    def __len__(self):
        return len(self.dataset)


class PoisonedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        poisoner,
        *,
        label=None,
        indices=None,
        eps=500,
        seed=1,
        transform=None
    ):
        self.orig_dataset = dataset
        self.label = label
        if not indices and not eps:
            raise ValueError()

        if not indices:
            if label is not None:
                clean_inds = [i for i, (x, y) in enumerate(dataset) if y == label]
            else:
                clean_inds = range(len(dataset))

            rng = np.random.RandomState(seed)
            indices = rng.choice(clean_inds, eps, replace=False)

        self.indices = indices
        self.poison_dataset = MappedDataset(Subset(dataset, indices), poisoner, seed=seed)
        if transform:
            self.poison_dataset = MappedDataset(self.poison_dataset, transform)

        clean_indices = list(set(range(len(dataset))).difference(indices))
        self.clean_dataset = Subset(dataset, clean_indices)
        if transform:
            self.clean_dataset = MappedDataset(self.clean_dataset, transform)

        self.dataset = ConcatDataset([self.clean_dataset, self.poison_dataset])

    def __getitem__(self, i: int):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class Poisoner(object):
    def poison(self, x: Image.Image) -> Image.Image:
        raise NotImplementedError()

    def __call__(self, x: Image.Image) -> Image.Image:
        return self.poison(x)


class PixelPoisoner(Poisoner):
    def __init__(
        self,
        *,
        method="pixel",
        pos: Tuple[int, int] = (11, 16),
        col: Tuple[int, int, int] = (101, 0, 25)
    ):
        self.method = method
        self.pos = pos
        self.col = col

    def poison(self, x: Image.Image) -> Image.Image:
        ret_x = x.copy()
        pos, col = self.pos, self.col

        if self.method == "pixel":
            ret_x.putpixel(pos, col)
        elif self.method == "pattern":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] - 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] - 1, pos[1] + 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] + 1), col)
        elif self.method == "ell":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] + 1, pos[1]), col)
            ret_x.putpixel((pos[0], pos[1] + 1), col)

        return ret_x


class StripePoisoner(Poisoner):
    def __init__(self, *, horizontal=True, strength=6, freq=16):
        self.horizontal = horizontal
        self.strength = strength
        self.freq = freq

    def poison(self, x: Image.Image) -> Image.Image:
        arr = np.asarray(x)
        (w, h, d) = arr.shape
        assert w == h  # have not tested w != h
        mask = np.full(
            (d, w, h), np.sin(np.linspace(0, self.freq * np.pi, h))
        ).swapaxes(0, 2)
        if self.horizontal:
            mask = mask.swapaxes(0, 1)
        mix = np.asarray(x) + self.strength * mask
        return Image.fromarray(np.uint8(mix.clip(0, 255)))


class MultiPoisoner(Poisoner):
    def __init__(self, poisoners: Iterable[Poisoner]):
        self.poisoners = poisoners

    def poison(self, x):
        for poisoner in self.poisoners:
            x = poisoner.poison(x)
        return x


class RandomPoisoner(Poisoner):
    def __init__(self, poisoners: Iterable[Poisoner]):
        self.poisoners = poisoners
        self.rng = np.random.RandomState()

    def poison(self, x):
        poisoner = self.rng.choice(self.poisoners)
        return poisoner.poison(x)

    def seed(self, i):
        self.rng.seed(i)

class LabelPoisoner(Poisoner):
    def __init__(self, poisoner: Poisoner, target_label: int):
        self.poisoner = poisoner
        self.target_label = target_label

    def poison(self, xy):
        x, _ = xy
        return self.poisoner(x), self.target_label

    def seed(self, i):
        if hasattr(self.poisoner, 'seed'):
            self.poisoner.seed(i)


def load_cifar_dataset(train=True):
    dataset = datasets.CIFAR10(root=str(CIFAR_PATH), train=train, download=True)
    return dataset


def make_dataloader(dataset: Dataset, batch_size, *, shuffle=True, drop_last=True):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataloader


def load_cifar_train(batch_size=32):
    path = "./data_cifar10"
    kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True}
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    return trainloader


def load_cifar_test(batch_size=32):
    path = "./data_cifar10"
    kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True}
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return testloader
