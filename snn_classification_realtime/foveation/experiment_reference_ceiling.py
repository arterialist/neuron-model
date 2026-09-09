"""Reference ceilings on CIFAR-10 (32x32): what does 'cracking it' mean?

Before asking whether PAULA can crack CIFAR, establish honest yardsticks with
conventional models of escalating capacity. This separates 'genuine structure a
small model can grab' from 'capacity/memorization only big dense models buy':

  - kNN on raw pixels            (nonparametric, no learned features)
  - logistic regression          (linear decision boundary)
  - tiny CNN (~small-net scale)  (a few learned conv features)
  - larger CNN                   (more capacity)

Grayscale and RGB, so we know the color cost too. No PAULA here; this is the
ruler we hold PAULA up against.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def load_xy(train: bool, gray: bool, root="./data"):
    tf = [transforms.Grayscale(1)] if gray else []
    tf += [transforms.ToTensor()]
    ds = datasets.CIFAR10(root=root, train=train, download=True,
                          transform=transforms.Compose(tf))
    X = torch.stack([ds[i][0] for i in range(len(ds))]).numpy()
    y = np.array([ds[i][1] for i in range(len(ds))])
    return X, y


class SmallCNN(nn.Module):
    def __init__(self, in_ch, width=16, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(width * 2 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_cnn(Xtr, ytr, Xte, yte, in_ch, width, epochs, device):
    net = SmallCNN(in_ch, width).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    n = len(Xtr_t)
    bs = 256
    for _ep in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
    net.eval()
    correct = 0
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(Xte_t), 512):
            xb = Xte_t[i:i + 512].to(device)
            pred = net(xb).argmax(1).cpu().numpy()
            correct += (pred == yte[i:i + 512]).sum()
    n_params = sum(p.numel() for p in net.parameters())
    return float(correct / len(yte)), int(n_params)


def run(args):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = {}
    for gray in ([True, False] if args.both_color else [args.gray]):
        tag = "grayscale" if gray else "rgb"
        Xtr, ytr = load_xy(True, gray)
        Xte, yte = load_xy(False, gray)
        in_ch = 1 if gray else 3
        # subset for the nonparametric/linear probes (full set is slow for kNN)
        s = args.probe_subset
        Xtr_f = Xtr.reshape(len(Xtr), -1)
        Xte_f = Xte.reshape(len(Xte), -1)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xtr_f[:s], ytr[:s])
        knn_acc = float(knn.score(Xte_f[:args.test_subset], yte[:args.test_subset]))
        logr = LogisticRegression(max_iter=200, C=1.0)
        logr.fit(Xtr_f[:s], ytr[:s])
        lin_acc = float(logr.score(Xte_f[:args.test_subset], yte[:args.test_subset]))

        cnn_small, p_small = train_cnn(Xtr, ytr, Xte, yte, in_ch, 8, args.epochs, device)
        cnn_big, p_big = train_cnn(Xtr, ytr, Xte, yte, in_ch, 32, args.epochs, device)

        results[tag] = {
            "knn_raw_pixels": knn_acc,
            "logistic_raw_pixels": lin_acc,
            "tiny_cnn_w8": {"acc": cnn_small, "params": p_small},
            "small_cnn_w32": {"acc": cnn_big, "params": p_big},
            "probe_subset": s,
            "cnn_epochs": args.epochs,
            "cnn_train_n": len(Xtr),
        }
        print(f"\n=== CIFAR-10 {tag} reference ceilings ===")
        print(f"  kNN(k=5) raw pixels:   {knn_acc:.3f}  (subset {s})")
        print(f"  logistic raw pixels:   {lin_acc:.3f}")
        print(f"  tiny CNN (w8, {p_small//1000}k params): {cnn_small:.3f}")
        print(f"  small CNN (w32, {p_big//1000}k params): {cnn_big:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"reference_ceiling_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


def main():
    p = argparse.ArgumentParser(description="CIFAR-10 reference ceilings")
    p.add_argument("--both-color", action="store_true", default=True)
    p.add_argument("--gray", action="store_true", default=True)
    p.add_argument("--probe-subset", type=int, default=5000)
    p.add_argument("--test-subset", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
