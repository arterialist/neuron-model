"""Vision dataset loading and selection for activity dataset building."""

from typing import Any

from torchvision import datasets, transforms

from snn_classification_realtime.core.config import DatasetConfig
from snn_classification_realtime.activity_dataset_builder.prompts import prompt_float


DATASET_NAMES = (
    "mnist",
    "cifar10",
    "cifar10_color",
    "cifar100",
    "usps",
    "svhn",
    "fashionmnist",
)


_ROOT_CANDIDATES = [
    "./data",
    "./data/mnist",
    "./data/cifar",
    "./data/usps",
    "./data/svhn",
    "./data/fashionmnist",
]


def _load_with_fallback(
    dataset_cls: type,
    root_candidates: list[str],
    **kwargs: Any,
) -> Any:
    """Try loading a dataset from multiple root candidates."""
    last_err: Exception | None = None
    for root in root_candidates:
        try:
            return dataset_cls(root=root, **kwargs)
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise RuntimeError(f"Failed to load dataset: {last_err}") from last_err
    raise RuntimeError("Failed to load dataset")


def load_mnist(train: bool) -> datasets.MNIST:
    """Load MNIST dataset with normalization to [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return _load_with_fallback(
        datasets.MNIST,
        ["./data/mnist", "./data"],
        train=train,
        download=True,
        transform=transform,
    )


def load_dataset_by_name(
    dataset_name: str,
    train: bool = True,
    cifar10_color_normalization_factor: float = 0.165,
) -> DatasetConfig:
    """Load dataset by name (non-interactive). For CLI use."""
    name = dataset_name.lower()
    if name == "mnist":
        ds = _load_with_fallback(
            datasets.MNIST,
            _ROOT_CANDIDATES,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=10,
            dataset_name="mnist",
        )
    if name == "cifar10":
        ds = _load_with_fallback(
            datasets.CIFAR10,
            _ROOT_CANDIDATES,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=10,
            dataset_name="cifar10",
        )
    if name == "cifar10_color":
        ds = _load_with_fallback(
            datasets.CIFAR10,
            _ROOT_CANDIDATES,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.shape[1] * img0.shape[2] * 3),
            num_classes=10,
            dataset_name="cifar10_color",
            is_colored_cifar10=True,
            cifar10_color_normalization_factor=cifar10_color_normalization_factor,
        )
    if name == "cifar100":
        ds = _load_with_fallback(
            datasets.CIFAR100,
            _ROOT_CANDIDATES,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=100,
            dataset_name="cifar100",
        )
    if name == "usps":
        ds = _load_with_fallback(
            datasets.USPS,
            _ROOT_CANDIDATES,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=10,
            dataset_name="usps",
        )
    if name == "svhn":
        ds = _load_with_fallback(
            datasets.SVHN,
            _ROOT_CANDIDATES,
            split="train",
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=10,
            dataset_name="svhn",
        )
    if name == "fashionmnist":
        ds = _load_with_fallback(
            datasets.FashionMNIST,
            _ROOT_CANDIDATES,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=10,
            dataset_name="fashionmnist",
        )
    raise ValueError(
        f"Unknown dataset: {dataset_name}. "
        f"Choose from: {', '.join(DATASET_NAMES)}"
    )


def select_and_load_dataset() -> DatasetConfig:
    """Select dataset (MNIST, CIFAR10, CIFAR100, etc.) and load training split.

    Returns DatasetConfig with dataset, vector size, num classes, and metadata.
    """
    print("Select dataset:")
    print("  1) MNIST")
    print("  2) CIFAR10")
    print("  3) CIFAR10 (color)")
    print("  4) CIFAR100")
    print("  5) USPS")
    print("  6) SVHN")
    print("  7) FashionMNIST")
    choice = input("Enter choice [1]: ").strip() or "1"

    if choice == "1":
        ds = _load_with_fallback(
            datasets.MNIST,
            _ROOT_CANDIDATES,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=10,
            dataset_name="mnist",
        )

    if choice == "2":
        ds = _load_with_fallback(
            datasets.CIFAR10,
            _ROOT_CANDIDATES,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=10,
            dataset_name="cifar10",
        )

    if choice == "3":
        ds = _load_with_fallback(
            datasets.CIFAR10,
            _ROOT_CANDIDATES,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
        img0, _ = ds[0]
        norm_factor = prompt_float(
            "Normalization factor for each color channel [0, X] (default 0.33 for [0, 0.33] range): ",
            0.33,
        )
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.shape[1] * img0.shape[2] * 3),
            num_classes=10,
            dataset_name="cifar10_color",
            is_colored_cifar10=True,
            cifar10_color_normalization_factor=norm_factor / 2.0,
        )

    if choice == "4":
        ds = _load_with_fallback(
            datasets.CIFAR100,
            _ROOT_CANDIDATES,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=100,
            dataset_name="cifar100",
        )

    if choice == "5":
        ds = _load_with_fallback(
            datasets.USPS,
            _ROOT_CANDIDATES,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=10,
            dataset_name="usps",
        )

    if choice == "6":
        ds = _load_with_fallback(
            datasets.SVHN,
            _ROOT_CANDIDATES,
            split="train",
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=10,
            dataset_name="svhn",
        )

    if choice == "7":
        ds = _load_with_fallback(
            datasets.FashionMNIST,
            _ROOT_CANDIDATES,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
        img0, _ = ds[0]
        return DatasetConfig(
            dataset=ds,
            image_vector_size=int(img0.numel()),
            num_classes=10,
            dataset_name="fashionmnist",
        )

    raise ValueError("Invalid dataset choice.")
