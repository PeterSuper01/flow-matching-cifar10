import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_dataset(root="./data"):
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)


def get_loader(dataset, batch_size=128, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)
