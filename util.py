import sys
import numpy as np
import pandas as pd
import scipy
import scipy.sparse.linalg
import torch
import tqdm
from functools import partial
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Collection, Dict, List, Union
import torch.backends.cudnn as cudnn

import datasets

if torch.cuda.is_available():
    cudnn.benchmark = True

default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def custom_svd(A, k=None, *, backend="arpack"):
    assert len(A.shape) == 2
    assert k is None or 1 <= k <= max(A.shape)
    if backend == "arpack" or backend is None:
        if k is None:
            k = min(A.shape)
        if k == A.shape[0]:
            A = np.vstack([A, np.zeros((1, A.shape[1]))])
            U, S, V = custom_svd(A, k)
            return U[:-1, :], S, V

        elif k == A.shape[1]:
            A = np.hstack([A, np.zeros((A.shape[0], 1))])
            U, S, V = custom_svd(A, k)
            return U, S, V[:, :-1]

        U, S, V = scipy.sparse.linalg.svds(A, k=k)
        return np.copy(U[:, ::-1]), np.copy(S[::-1]), np.copy(V[::-1])

    elif backend == "lapack":
        U, S, V = scipy.linalg.svd(A, full_matrices=False)
        if k is None or k == min(A.shape):
            return U, S, V
        return U[:, :k], S[:k], V[:k, :]

    elif backend == "irlb":
        import irlb

        U, S, Vt, _, _ = irlb.irlb(A, k)
        return U, S, Vt.T

    raise ValueError(f"Invalid backend {backend}")


def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return "ipykernel" in sys.modules


if in_notebook():
    import tqdm.notebook


def make_pbar(*args, **kwargs):
    pbar_constructor = (
        tqdm.notebook.tqdm if in_notebook() else partial(tqdm.tqdm, dynamic_ncols=True)
    )
    return pbar_constructor(*args, **kwargs)


def get_module_device(module: torch.nn.Module, check=True):
    if check:
        assert len(set(param.device for param in module.parameters())) == 1
    return next(module.parameters()).device


# Just some helpers to inspect the parameter shapes of a network
def param_count(net: torch.nn.Module):
    return sum(p.numel() for p in net.parameters())


def param_layer_count(net: torch.nn.Module):
    return len(list(net.parameters()))


def param_size_max(net: torch.nn.Module):
    return max(p.numel() for p in net.parameters())


def param_shapes(net: torch.nn.Module):
    return [(k, p.shape) for k, p in net.named_parameters()]


def param_sizes(net: torch.nn.Module):
    return [(k, p.shape.numel()) for k, p in net.named_parameters()]


def either_dataloader_dataset_to_both(
    data: Union[DataLoader, Dataset], *, batch_size=None, eval=False, **kwargs
):
    if isinstance(data, DataLoader):
        dataloader = data
        dataset = data.dataset
    elif isinstance(data, Dataset):
        dataset = data
        dl_kwargs = {}

        if eval:
            dl_kwargs.update(dict(batch_size=1000, shuffle=False, drop_last=False))
        else:
            dl_kwargs.update(dict(batch_size=128, shuffle=True))

        if batch_size is not None:
            dl_kwargs["batch_size"] = batch_size

        dl_kwargs.update(kwargs)

        dataloader = datasets.make_dataloader(data, **dl_kwargs)
    else:
        raise NotImplementedError()
    return dataloader, dataset


clf_loss = torch.nn.CrossEntropyLoss()


def clf_correct(y_pred: torch.Tensor, y: torch.Tensor):
    y_hat = y_pred.data.max(1)[1]
    correct = (y_hat == y).long().cpu().sum()
    return correct


def clf_eval(model: torch.nn.Module, data: Union[DataLoader, Dataset]):
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(data, eval=True)
    total_correct, total_loss = 0.0, 0.0
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = clf_loss(y_pred, y)
            correct = clf_correct(y_pred, y)

            total_correct += correct.item()
            total_loss += loss.item()

    n = len(dataloader.dataset)
    total_correct /= n
    total_loss /= n
    return total_correct, total_loss


def get_mean_lr(opt: optim.Optimizer):
    return np.mean([group["lr"] for group in opt.param_groups])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, *, n=1, weight=1):
        self.val = val
        self.sum += val * weight
        self.count += n * weight

    @property
    def avg(self):
        if self.count == 0:
            if self.sum == 0:
                return 0
            else:
                return np.sign(self.sum) * np.inf

        else:
            return self.sum / self.count


class ExtremaMeter(object):
    def __init__(self, maximum=False):
        self.maximum = maximum
        self.reset()

    def reset(self):
        self.val = 0

        if self.maximum:
            self.extrema = -np.inf
        else:
            self.extrema = np.inf

    def update(self, val):
        self.val = val

        if self.maximum:
            if val > self.extrema:
                self.extrema = val
                return True

        else:
            if val < self.extrema:
                self.extrema = val
                return True

        return False


class MaxMeter(ExtremaMeter):
    def __init__(self):
        super().__init__(True)

    @property
    def max(self):
        return self.extrema


class MinMeter(ExtremaMeter):
    def __init__(self):
        super().__init__()

    @property
    def min(self):
        return self.extrema


class FlatThenCosineAnnealingLR(object):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, flat_ratio=0.7):
        self.last_epoch = last_epoch
        self.flat_ratio = flat_ratio
        self.T_max = T_max
        self.inner = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            int(T_max * (1 - flat_ratio)),
            eta_min,
            max(-1, last_epoch - flat_ratio * T_max - 1),
        )

    def step(self):
        self.last_epoch += 1
        if self.last_epoch >= self.flat_ratio * self.T_max:
            self.inner.step()

    def state_dict(self):
        result = {
            "inner." + key: value for key, value in self.inner.state_dict().items()
        }
        result.update(
            {key: value for key, value in self.__dict__.items() if key != "inner"}
        )
        return result

    def load_state_dict(self, state_dict):
        self.inner.load_state_dict(
            {k[6:]: v for k, v in state_dict.items() if k.startswith("inner.")}
        )
        self.__dict__.update(
            {k: v for k, v in state_dict.items() if not k.startswith("inner.")}
        )


def mini_train(
    *,
    model: torch.nn.Module,
    train_data: Union[DataLoader, Dataset],
    test_data: Union[DataLoader, Dataset] = None,
    batch_size=32,
    opt: optim.Optimizer,
    scheduler,
    epochs: int,
):
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(train_data, batch_size=batch_size)
    n = len(dataloader.dataset)
    total_examples = epochs * n
    with make_pbar(total=total_examples) as pbar:
        for _ in range(1, epochs + 1):
            train_epoch_loss, train_epoch_correct = 0, 0
            model.train()
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                minibatch_size = len(x)
                model.zero_grad()
                y_pred = model(x)
                loss = clf_loss(y_pred, y)
                correct = clf_correct(y_pred, y)
                loss.backward()
                opt.step()
                train_epoch_correct += int(correct.item())
                train_epoch_loss += float(loss.item())
                pbar.update(minibatch_size)

            lr = get_mean_lr(opt)
            if scheduler:
                scheduler.step()

            pbar_postfix = {
                "acc": "%.2f" % (train_epoch_correct / n * 100),
                "loss": "%.4g" % (train_epoch_loss / n),
                "lr": "%.3g" % lr,
            }
            if test_data:
                test_epoch_acc, test_epoch_loss = clf_eval(model, test_data)
                pbar_postfix.update(
                    {
                        "tacc": "%.2f" % (test_epoch_acc * 100),
                        "tloss": "%.4g" % test_epoch_loss,
                    }
                )

            pbar.set_postfix(**pbar_postfix)

    return model


def compute_reps(model: torch.nn.Module, data: Union[DataLoader, Dataset]):
    device = get_module_device(model)
    dataloader, dataset = either_dataloader_dataset_to_both(data, eval=True)
    n = len(dataset)
    inner_shape = model(dataset[0][0][None, ...].to(device)).shape[1:]
    reps = torch.empty(n, *inner_shape)

    with torch.no_grad():
        model.eval()
        start_index = 0
        for x, _ in dataloader:
            x = x.to(device)
            minibatch_size = len(x)
            rep = model(x)
            reps[start_index : start_index + minibatch_size] = rep
            start_index += minibatch_size

    return reps


def compute_all_reps(
    model: torch.nn.Sequential,
    data: Union[DataLoader, Dataset],
    *,
    layers: Collection[int],
    flat=False,
) -> Dict[int, np.ndarray]:
    device = get_module_device(model)
    dataloader, dataset = either_dataloader_dataset_to_both(data, eval=True)
    n = len(dataset)
    max_layer = max(layers)
    assert max_layer < len(model)

    reps = {}
    x = dataset[0][0][None, ...].to(device)
    for i, layer in enumerate(model):
        if i > max_layer:
            break
        x = layer(x)
        if i in layers:
            inner_shape = x.shape[1:]
            reps[i] = torch.empty(n, *inner_shape)

    with torch.no_grad():
        model.eval()
        start_index = 0
        for x, _ in dataloader:
            x = x.to(device)
            minibatch_size = len(x)
            for i, layer in enumerate(model):
                if i > max_layer:
                    break
                x = layer(x)
                if i in layers:
                    reps[i][start_index : start_index + minibatch_size] = x.cpu()

            start_index += minibatch_size

    if flat:
        for layer in reps:
            layer_reps = reps[layer]
            reps[layer] = layer_reps.reshape(layer_reps.shape[0], -1)

    return reps


def compute_spectral_df(
    model: torch.nn.Module,
    dataset: Union[DataLoader, Dataset],
    *,
    layers: Collection[int],
    labels=range(10),
    k=64,
    svd_backend=None,
):
    eps = len(dataset.poison_dataset)
    target_label = dataset.poison_dataset.dataset.mapper.target_label
    lsd = datasets.LabelSortedDataset(dataset)
    dfs = []
    with make_pbar(total=len(labels) * len(layers)) as pbar:
        for label in labels:
            if isinstance(label, int):
                lbs = [label]
            else:
                lbs = label
            all_reps = compute_all_reps(
                model, lsd.subset(lbs), layers=layers, flat=True
            )
            for layer, reps in all_reps.items():
                reps = reps.numpy()
                reps_centered = reps - reps.mean(axis=0)
                U, S, V = custom_svd(reps_centered, k=k, backend=svd_backend)
                df = pd.DataFrame(reps @ V.T)
                df["layer"] = layer
                df["poison"] = False
                df["label"] = -1
                label_index = 0
                for lb in lbs:
                    l = len(lsd.by_label[lb])
                    df.iloc[
                        label_index : label_index + l, df.columns.get_loc("label")
                    ] = lb
                    if lb == target_label:
                        df.iloc[
                            label_index + l - eps : label_index + l,
                            df.columns.get_loc("poison"),
                        ] = True
                    label_index += l
                df["norm"] = np.linalg.norm(reps_centered, axis=1)
                dfs.append(df)
                pbar.update(1)

    return pd.concat(dfs)
