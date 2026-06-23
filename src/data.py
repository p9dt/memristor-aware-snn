"""Dataset loaders.

Phase 1 uses MNIST (flattened, normalized). The ``load_dataset`` indirection is
intentional: Phase 1.5 will add an event-based loader (N-MNIST / DVS-Gesture via
``tonic``) behind the same interface.
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for MNIST.

    Images are normalized and left as (1, 28, 28); the training loop flattens
    them to 784. MNIST is downloaded to ``data_dir`` on first run.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def load_dataset(name: str = "mnist", **kwargs) -> tuple[DataLoader, DataLoader]:
    """Dispatch by dataset name. (N-MNIST hook lands in Phase 1.5.)"""
    name = name.lower()
    if name == "mnist":
        return load_mnist(**kwargs)
    raise ValueError(f"Unknown dataset {name!r}. Available: 'mnist'.")
