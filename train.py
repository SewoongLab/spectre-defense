import torch
from torch import nn, optim
from pathlib import Path
from ranger_opt.ranger import ranger2020 as ranger

from model import SequentialImageNetwork, SequentialImageNetworkMod
from util import *
from datasets import *
import re

import sys

name = sys.argv[1]

model_flag = name.split("-")[0]
train_flag = name.split("-")[1]

if model_flag == "r32p":
    import resnet

    model = SequentialImageNetworkMod(resnet.resnet32()).cuda()
elif model_flag == "r18":
    from pytorch_cifar.models import resnet

    model = SequentialImageNetwork(resnet.ResNet18()).cuda()
else:
    raise NotImplementedError

eps = int(re.search(r"[0-9]+$", name).group())
poisoner_flag = name.split("-")[3][:3]
clean_label = int(name.split("-")[2][0])
target_label = int(name.split("-")[2][1])

print(f"{model_flag=} {clean_label=} {target_label=} {poisoner_flag=} {eps=}")

if len(sys.argv) > 2:
    retrain = sys.argv[2]
    target_mask = np.load(Path("output") / name / f"{retrain}.npy")
    assert len(target_mask) == 5000 + eps
    target_mask_ind = [i for i in range(5000 + eps) if not target_mask[i]]
    poison_removed = np.sum(target_mask[-eps:])
    clean_removed = np.sum(target_mask) - poison_removed
    print(f"{poison_removed=} {clean_removed=}")
else:
    retrain = None

print("Building datasets...")

if poisoner_flag == "1xp":
    x_poisoner = PixelPoisoner()
    all_x_poisoner = PixelPoisoner()

elif poisoner_flag == "2xp":
    x_poisoner = RandomPoisoner(
        [
            PixelPoisoner(),
            PixelPoisoner(pos=(5, 27), col=(101, 123, 121)),
        ]
    )
    all_x_poisoner = MultiPoisoner(
        [
            PixelPoisoner(),
            PixelPoisoner(pos=(5, 27), col=(101, 123, 121)),
        ]
    )

elif poisoner_flag == "3xp":
    x_poisoner = RandomPoisoner(
        [
            PixelPoisoner(),
            PixelPoisoner(pos=(5, 27), col=(101, 123, 121)),
            PixelPoisoner(pos=(30, 7), col=(0, 36, 54)),
        ]
    )
    all_x_poisoner = MultiPoisoner(
        [
            PixelPoisoner(),
            PixelPoisoner(pos=(5, 27), col=(101, 123, 121)),
            PixelPoisoner(pos=(30, 7), col=(0, 36, 54)),
        ]
    )

elif poisoner_flag == "1xs":
    x_poisoner = StripePoisoner(strength=6, freq=16)
    all_x_poisoner = StripePoisoner(strength=6, freq=16)

elif poisoner_flag == "2xs":
    x_poisoner = RandomPoisoner(
        [
            StripePoisoner(strength=6, freq=16),
            StripePoisoner(strength=6, freq=16, horizontal=False),
        ]
    )
    all_x_poisoner = MultiPoisoner(
        [
            StripePoisoner(strength=6, freq=16),
            StripePoisoner(strength=6, freq=16, horizontal=False),
        ]
    )

else:
    raise NotImplementedError

poisoner = LabelPoisoner(x_poisoner, target_label=target_label)
all_poisoner = LabelPoisoner(all_x_poisoner, target_label=target_label)

cifar_train_dataset = load_cifar_dataset()
cifar_test_dataset = load_cifar_dataset(train=False)

poison_cifar_train = PoisonedDataset(
    cifar_train_dataset,
    poisoner,
    eps=eps,
    label=clean_label,
    transform=CIFAR_TRANSFORM_TRAIN_XY,
)

if retrain:
    lsd = LabelSortedDataset(poison_cifar_train)
    target_subset = lsd.subset(target_label)
    poison_cifar_train = ConcatDataset(
        [lsd.subset(label) for label in range(10) if label != target_label]
        + [Subset(target_subset, target_mask_ind)]
    )

cifar_test = MappedDataset(cifar_test_dataset, CIFAR_TRANSFORM_TEST_XY)

poison_cifar_test = PoisonedDataset(
    cifar_test_dataset,
    poisoner,
    eps=1000,
    label=clean_label,
    transform=CIFAR_TRANSFORM_TEST_XY,
)

all_poison_cifar_test = PoisonedDataset(
    cifar_test_dataset,
    all_poisoner,
    eps=1000,
    label=clean_label,
    transform=CIFAR_TRANSFORM_TEST_XY,
)

if train_flag == "sgd":
    batch_size = 128
    epochs = 200
    opt = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=2e-4
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[75, 150], gamma=0.1)

elif train_flag == "ranger":
    batch_size = 128
    epochs = 60
    opt = ranger.Ranger(
        model.parameters(),
        lr=0.001 * (batch_size / 32),
        weight_decay=1e-1,
        betas=(0.9, 0.999),
        eps=1e-1,
    )
    lr_scheduler = FlatThenCosineAnnealingLR(opt, T_max=epochs)

if __name__ == "__main__":
    print("Training...")

    mini_train(
        model=model,
        train_data=poison_cifar_train,
        test_data=cifar_test,
        batch_size=batch_size,
        opt=opt,
        scheduler=lr_scheduler,
        epochs=epochs,
    )

    print("Evaluating...")

    if not retrain:
        clean_train_acc = clf_eval(model, poison_cifar_train.clean_dataset)[0]
        poison_train_acc = clf_eval(model, poison_cifar_train.poison_dataset)[0]
        print(f"{clean_train_acc=}")
        print(f"{poison_train_acc=}")

    clean_test_acc = clf_eval(model, cifar_test)[0]
    poison_test_acc = clf_eval(model, poison_cifar_test.poison_dataset)[0]
    all_poison_test_acc = clf_eval(model, all_poison_cifar_test.poison_dataset)[0]

    print(f"{clean_test_acc=}")
    print(f"{poison_test_acc=}")
    print(f"{all_poison_test_acc=}")

    print("Saving model...")
    output_dir = Path('output') / name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{retrain}-model.pth" if retrain else "model.pth"
    torch.save(model.state_dict(), output_dir / output_name)
