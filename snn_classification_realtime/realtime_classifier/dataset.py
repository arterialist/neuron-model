"""Dataset loading for real-time classification."""

from torchvision import datasets, transforms

from activity_dataset_builder.config import DatasetConfig


def select_and_load_dataset(
    dataset_name: str,
    cifar10_color_upper_bound: float = 1.0,
) -> DatasetConfig:
    """Load a specified dataset and return DatasetConfig."""
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset_map = {
        "mnist": (datasets.MNIST, transform_mnist, 10, False),
        "fashionmnist": (datasets.FashionMNIST, transform_mnist, 10, False),
        "cifar10": (datasets.CIFAR10, transform_cifar, 10, False),
        "cifar10_color": (datasets.CIFAR10, transform_cifar, 10, True),
        "cifar100": (datasets.CIFAR100, transform_cifar, 100, False),
    }

    if dataset_name.lower() not in dataset_map:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Choose from {list(dataset_map.keys())}."
        )

    loader, transform, num_classes, is_colored = dataset_map[dataset_name.lower()]
    cifar10_norm_factor = cifar10_color_upper_bound / 2.0

    ds = loader(root="./data", train=False, download=True, transform=transform)
    img0, _ = ds[0]

    if is_colored:
        image_vector_size = int(img0.shape[1] * img0.shape[2] * 3)
        print(
            f"Successfully loaded {dataset_name} dataset (colored, "
            f"{img0.shape[1]}x{img0.shape[2]} pixels × 3 channels = {image_vector_size} synapses)."
        )
        print(
            f"Each color channel normalized to [0, {cifar10_color_upper_bound:.3f}] range"
        )
        return DatasetConfig(
            dataset=ds,
            image_vector_size=image_vector_size,
            num_classes=num_classes,
            dataset_name=dataset_name,
            is_colored_cifar10=True,
            cifar10_color_normalization_factor=cifar10_norm_factor,
        )

    image_vector_size = int(img0.numel())
    print(f"Successfully loaded {dataset_name} dataset.")
    return DatasetConfig(
        dataset=ds,
        image_vector_size=image_vector_size,
        num_classes=num_classes,
        dataset_name=dataset_name,
        is_colored_cifar10=False,
        cifar10_color_normalization_factor=cifar10_norm_factor,
    )
