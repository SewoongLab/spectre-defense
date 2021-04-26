from pathlib import Path
from tqdm import trange
from train import *

assert not retrain

model_file = Path("output") / name / "model.pth"
model.load_state_dict(torch.load(model_file))


print("Evaluating...")

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

lsd = LabelSortedDataset(poison_cifar_train)

if model_flag == "r32p":
    layer = 14
elif model_flag == "r18":
    layer = 13

for i in trange(lsd.n, dynamic_ncols=True):
    target_reps = compute_all_reps(model, lsd.subset(i), layers=[layer], flat=True)[
        layer
    ]
    np.save(Path("output") / name / f"label_{i}_reps.npy", target_reps.numpy())
