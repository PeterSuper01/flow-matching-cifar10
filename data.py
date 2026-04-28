import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_dataset(root="./data", train=True):
    transforms = [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    if train:
        transforms = [T.RandomHorizontalFlip()] + transforms
    return torchvision.datasets.CIFAR10(
        root=root, train=train, download=True, transform=T.Compose(transforms)
    )


def get_loader(dataset, batch_size=128, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)
