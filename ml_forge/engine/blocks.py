"""
blocks.py
Block (node type) definitions and lookup helpers.
"""

from typing import Optional

SECTIONS: dict = {
    "Model Creation": {
        "Layers": [
            {
                "label": "Linear", "color": (100, 180, 255),
                "params": ["in_features", "out_features"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {},
            },
            {
                "label": "Conv2D", "color": (120, 220, 140),
                "params": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"kernel_size": "3", "stride": "1", "padding": "1"},
            },
            {
                "label": "ConvTranspose2D", "color": (120, 220, 140),
                "params": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"kernel_size": "3", "stride": "1", "padding": "0"},
            },
            {
                "label": "Flatten", "color": (100, 180, 255),
                "params": ["start_dim", "end_dim"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"start_dim": "1", "end_dim": "-1"},
            },
        ],
        "Activations": [
            {"label": "ReLU",      "color": (255, 180, 80), "params": [],                 "inputs": ["x"], "outputs": ["out"], "defaults": {}},
            {"label": "Sigmoid",   "color": (255, 180, 80), "params": [],                 "inputs": ["x"], "outputs": ["out"], "defaults": {}},
            {"label": "Tanh",      "color": (255, 180, 80), "params": [],                 "inputs": ["x"], "outputs": ["out"], "defaults": {}},
            {"label": "GELU",      "color": (255, 180, 80), "params": [],                 "inputs": ["x"], "outputs": ["out"], "defaults": {}},
            {
                "label": "Softmax", "color": (255, 180, 80),
                "params": ["dim"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"dim": "1"},
            },
            {
                "label": "LeakyReLU", "color": (255, 180, 80),
                "params": ["negative_slope"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"negative_slope": "0.01"},
            },
        ],
        "Normalization": [
            {
                "label": "BatchNorm2D", "color": (200, 130, 255),
                "params": ["num_features", "eps", "momentum"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"eps": "1e-5", "momentum": "0.1"},
            },
            {
                "label": "LayerNorm", "color": (200, 130, 255),
                "params": ["normalized_shape", "eps"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"eps": "1e-5"},
            },
            {
                "label": "GroupNorm", "color": (200, 130, 255),
                "params": ["num_groups", "num_channels"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {},
            },
            {
                "label": "Dropout", "color": (200, 130, 255),
                "params": ["p"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"p": "0.5"},
            },
        ],
        "Pooling": [
            {
                "label": "MaxPool2D", "color": (255, 120, 120),
                "params": ["kernel_size", "stride", "padding"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"kernel_size": "2", "stride": "2", "padding": "0"},
            },
            {
                "label": "AvgPool2D", "color": (255, 120, 120),
                "params": ["kernel_size", "stride", "padding"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"kernel_size": "2", "stride": "2", "padding": "0"},
            },
            {
                "label": "AdaptiveAvgPool2D", "color": (255, 120, 120),
                "params": ["output_size"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"output_size": "1"},
            },
        ],
        "I/O": [
            {"label": "Input",  "color": (80, 220, 200), "params": ["shape"], "inputs": [],    "outputs": ["out"], "defaults": {}},
            {"label": "Output", "color": (80, 220, 200), "params": ["shape"], "inputs": ["x"], "outputs": [],      "defaults": {}},
        ],
    },
    "Training": {
        "Pipeline Inputs": [
            {"label": "ModelBlock",      "color": (80,  180, 255), "params": [], "inputs": ["images"], "outputs": ["predictions"], "defaults": {}},
            {"label": "DataLoaderBlock", "color": (180, 100, 255), "params": [], "inputs": [],         "outputs": ["images", "labels"], "defaults": {}},
        ],
        "Loss Functions": [
            {"label": "CrossEntropyLoss", "color": (255, 160, 100), "params": ["weight", "ignore_index", "reduction"], "inputs": ["pred", "target"], "outputs": ["loss"], "defaults": {"reduction": "mean"}},
            {"label": "MSELoss",          "color": (255, 160, 100), "params": ["reduction"],                           "inputs": ["pred", "target"], "outputs": ["loss"], "defaults": {"reduction": "mean"}},
            {"label": "BCELoss",          "color": (255, 160, 100), "params": ["reduction"],                           "inputs": ["pred", "target"], "outputs": ["loss"], "defaults": {"reduction": "mean"}},
            {"label": "BCEWithLogits",    "color": (255, 160, 100), "params": ["reduction"],                           "inputs": ["pred", "target"], "outputs": ["loss"], "defaults": {"reduction": "mean"}},
            {"label": "NLLLoss",          "color": (255, 160, 100), "params": ["reduction"],                           "inputs": ["pred", "target"], "outputs": ["loss"], "defaults": {"reduction": "mean"}},
            {"label": "HuberLoss",        "color": (255, 160, 100), "params": ["delta", "reduction"],                  "inputs": ["pred", "target"], "outputs": ["loss"], "defaults": {"delta": "1.0", "reduction": "mean"}},
            {"label": "KLDivLoss",        "color": (255, 160, 100), "params": ["reduction"],                           "inputs": ["pred", "target"], "outputs": ["loss"], "defaults": {"reduction": "mean"}},
        ],
        "Optimizers": [
            {"label": "Adam",    "color": (100, 220, 180), "params": ["lr", "betas", "eps", "weight_decay"], "inputs": ["params"], "outputs": [], "defaults": {"lr": "0.001", "betas": "0.9, 0.999", "eps": "1e-8", "weight_decay": "0.0"}},
            {"label": "AdamW",   "color": (100, 220, 180), "params": ["lr", "betas", "eps", "weight_decay"], "inputs": ["params"], "outputs": [], "defaults": {"lr": "0.001", "betas": "0.9, 0.999", "eps": "1e-8", "weight_decay": "0.01"}},
            {"label": "SGD",     "color": (100, 220, 180), "params": ["lr", "momentum", "weight_decay"],     "inputs": ["params"], "outputs": [], "defaults": {"lr": "0.01", "momentum": "0.9", "weight_decay": "0.0"}},
            {"label": "RMSprop", "color": (100, 220, 180), "params": ["lr", "alpha", "eps", "weight_decay"], "inputs": ["params"], "outputs": [], "defaults": {"lr": "0.01", "alpha": "0.99", "eps": "1e-8", "weight_decay": "0.0"}},
            {"label": "Adagrad", "color": (100, 220, 180), "params": ["lr", "lr_decay", "weight_decay"],     "inputs": ["params"], "outputs": [], "defaults": {"lr": "0.01", "lr_decay": "0.0", "weight_decay": "0.0"}},
            {"label": "LBFGS",   "color": (100, 220, 180), "params": ["lr", "max_iter", "history_size"],     "inputs": ["params"], "outputs": [], "defaults": {"lr": "1.0", "max_iter": "20", "history_size": "100"}},
        ],
    },
    "Data Prep": {
        "Datasets": [
            {"label": "MNIST",        "color": (220, 180, 255), "params": ["root", "train", "download"], "inputs": [], "outputs": ["img"], "defaults": {"root": "./data", "train": "True", "download": "True"}},
            {"label": "CIFAR10",      "color": (220, 180, 255), "params": ["root", "train", "download"], "inputs": [], "outputs": ["img"], "defaults": {"root": "./data", "train": "True", "download": "True"}},
            {"label": "CIFAR100",     "color": (220, 180, 255), "params": ["root", "train", "download"], "inputs": [], "outputs": ["img"], "defaults": {"root": "./data", "train": "True", "download": "True"}},
            {"label": "FashionMNIST", "color": (220, 180, 255), "params": ["root", "train", "download"], "inputs": [], "outputs": ["img"], "defaults": {"root": "./data", "train": "True", "download": "True"}},
            {"label": "ImageFolder",  "color": (220, 180, 255), "params": ["root"],                      "inputs": [], "outputs": ["img"], "defaults": {"root": "./data"}},
        ],
        "Augmentation": [
            {"label": "Resize",         "color": (255, 200, 120), "params": ["size"],                                     "inputs": ["img"], "outputs": ["img"], "defaults": {}},
            {"label": "CenterCrop",     "color": (255, 200, 120), "params": ["size"],                                     "inputs": ["img"], "outputs": ["img"], "defaults": {}},
            {"label": "RandomCrop",     "color": (255, 200, 120), "params": ["size", "padding"],                          "inputs": ["img"], "outputs": ["img"], "defaults": {"padding": "4"}},
            {"label": "RandomHFlip",    "color": (255, 200, 120), "params": ["p"],                                        "inputs": ["img"], "outputs": ["img"], "defaults": {"p": "0.5"}},
            {"label": "RandomVFlip",    "color": (255, 200, 120), "params": ["p"],                                        "inputs": ["img"], "outputs": ["img"], "defaults": {"p": "0.5"}},
            {"label": "ColorJitter",    "color": (255, 200, 120), "params": ["brightness", "contrast", "saturation", "hue"], "inputs": ["img"], "outputs": ["img"], "defaults": {}},
            {"label": "RandomRotation", "color": (255, 200, 120), "params": ["degrees"],                                  "inputs": ["img"], "outputs": ["img"], "defaults": {}},
            {"label": "GaussianBlur",   "color": (255, 200, 120), "params": ["kernel_size", "sigma"],                     "inputs": ["img"], "outputs": ["img"], "defaults": {"kernel_size": "3", "sigma": "0.1, 2.0"}},
            {"label": "RandomErasing",  "color": (255, 200, 120), "params": ["p", "scale", "ratio"],                      "inputs": ["img"], "outputs": ["img"], "defaults": {"p": "0.5", "scale": "0.02, 0.33", "ratio": "0.3, 3.3"}},
            {"label": "Normalize",      "color": (255, 200, 120), "params": ["mean", "std"],                              "inputs": ["img"], "outputs": ["img"], "defaults": {"mean": "0.5", "std": "0.5"}},
            {"label": "ToTensor",       "color": (255, 200, 120), "params": [],                                           "inputs": ["img"], "outputs": ["img"], "defaults": {}},
            {"label": "Grayscale",      "color": (255, 200, 120), "params": ["num_output_channels"],                      "inputs": ["img"], "outputs": ["img"], "defaults": {"num_output_channels": "1"}},
        ],
        "DataLoader": [
            {
                "label": "DataLoader (train)", "color": (200, 160, 255),
                "params": ["batch_size", "shuffle", "num_workers", "pin_memory"],
                "inputs": ["img"], "outputs": [],
                "defaults": {"batch_size": "32", "shuffle": "True", "num_workers": "0", "pin_memory": "False"},
            },
            {
                "label": "DataLoader (val)", "color": (180, 140, 235),
                "params": ["batch_size", "num_workers", "pin_memory"],
                "inputs": ["img"], "outputs": [],
                "defaults": {"batch_size": "32", "num_workers": "0", "pin_memory": "False"},
            },
        ],
    },
}


def get_block_def(label: str) -> Optional[dict]:
    """Return the block definition dict for a given label, or None."""
    for section in SECTIONS.values():
        for block_list in section.values():
            for block in block_list:
                if block["label"] == label:
                    return block
    return None


def all_block_labels() -> list[str]:
    """Return a flat list of every block label across all sections."""
    labels = []
    for section in SECTIONS.values():
        for block_list in section.values():
            for block in block_list:
                labels.append(block["label"])
    return labels