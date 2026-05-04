#!/usr/bin/env python3
"""
CNN Transfer-Learning Analysis — main entry point.

Trains ResNet-18 and ResNet-50 under three modes (scratch / fine-tune / linear-probe)
across multiple dataset fractions and random seeds, then computes CKA
representation analysis and generates all paper-quality visualisations.

Usage
-----
  python run_cnn.py                          # full run (all archs, datasets, modes, fractions)
  python run_cnn.py --arch resnet18          # single architecture
  python run_cnn.py --dataset cifar10        # single dataset
  python run_cnn.py --mode scratch           # single mode
  python run_cnn.py --skip-train             # skip training, load saved metrics & plot
  python run_cnn.py --config path/to/cfg.yml # custom config
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import yaml

from cnn.cka import (
    compute_cka_matrix,
    extract_features,
    extract_features_with_labels,
)
from cnn.data import DATASET_STATS, get_dataloaders
from cnn.model import ResNetExtractor
from cnn.trainer import Trainer
from cnn.visualizations import (
    plot_accuracy_vs_fraction,
    plot_cka_across_fractions,
    plot_cka_heatmap,
    plot_feature_tsne,
    plot_summary_table,
    plot_training_curves,
    plot_transfer_comparison,
    plot_transfer_gain,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_device(cfg: dict) -> torch.device:
    if cfg["training"]["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    print("  [info] CUDA not available — falling back to CPU.")
    return torch.device("cpu")


def checkpoint_path(cfg: dict, arch: str, dataset: str, mode: str, frac: float, seed: int) -> str:
    return os.path.join(
        cfg["results"]["checkpoint_dir"],
        arch, dataset, mode,
        f"frac{int(frac * 100)}",
        f"seed{seed}.pt",
    )


def metrics_path(cfg: dict, arch: str, dataset: str, mode: str, frac: float, seed: int) -> str:
    return os.path.join(
        cfg["results"]["metrics_dir"],
        arch, dataset, mode,
        f"frac{int(frac * 100)}",
        f"seed{seed}.json",
    )


def loader_kwargs(cfg: dict, dataset: str, frac: float, seed: int):
    return dict(
        dataset_name=dataset,
        root=cfg["datasets"]["root"],
        fraction=frac,
        seed=seed,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["datasets"]["num_workers"],
        image_size=cfg["datasets"]["image_size"],
    )


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def run_training(cfg, arch, dataset, mode, frac, seed, device, out_dir) -> dict:
    set_seed(seed)
    num_classes = DATASET_STATS[dataset]["num_classes"]
    train_loader, val_loader = get_dataloaders(**loader_kwargs(cfg, dataset, frac, seed))
    model = ResNetExtractor(num_classes=num_classes, mode=mode, arch=arch)
    ckpt = checkpoint_path(cfg, arch, dataset, mode, frac, seed)
    trainer = Trainer(model, train_loader, val_loader, mode, cfg["training"], device)
    result = trainer.fit(checkpoint_path=ckpt)

    mpath = metrics_path(cfg, arch, dataset, mode, frac, seed)
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "w") as fh:
        json.dump(result, fh, indent=2)

    plot_training_curves(
        history=result["history"],
        mode=mode,
        dataset_name=dataset,
        fraction=frac,
        seed=seed,
        output_dir=out_dir,
        dpi=cfg["visualization"]["figure_dpi"],
    )
    return result


def load_metrics(cfg, arch, dataset, mode, frac, seed) -> dict:
    mpath = metrics_path(cfg, arch, dataset, mode, frac, seed)
    if os.path.isfile(mpath):
        with open(mpath) as fh:
            return json.load(fh)
    return {}


# ------------------------------------------------------------------
# CKA analysis
# ------------------------------------------------------------------

def run_cka(cfg, arch, dataset, mode, frac, seed, device, layer_names, out_dir):
    """Load checkpoint, extract features, compute & plot CKA matrix."""
    num_classes = DATASET_STATS[dataset]["num_classes"]
    ckpt = checkpoint_path(cfg, arch, dataset, mode, frac, seed)
    if not os.path.isfile(ckpt):
        return None, None

    model = ResNetExtractor(num_classes=num_classes, mode=mode, arch=arch)
    saved = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(saved["state_dict"])
    model.to(device)

    _, val_loader = get_dataloaders(**loader_kwargs(cfg, dataset, 1.0, seed))

    features = extract_features(
        model, val_loader, layer_names, cfg["cka"]["num_samples"], device
    )

    cka_mat = compute_cka_matrix(features)
    plot_cka_heatmap(
        cka_matrix=cka_mat,
        layer_names=layer_names,
        mode=mode,
        dataset_name=dataset,
        fraction=frac,
        output_dir=out_dir,
        dpi=cfg["visualization"]["figure_dpi"],
    )
    return cka_mat, features


# ------------------------------------------------------------------
# t-SNE
# ------------------------------------------------------------------

def run_tsne(cfg, arch, dataset, mode, seed, device, layer_names, out_dir):
    """Visualise penultimate-layer features (full dataset, seed 0 checkpoint)."""
    num_classes = DATASET_STATS[dataset]["num_classes"]
    ckpt = checkpoint_path(cfg, arch, dataset, mode, 1.0, seed)
    if not os.path.isfile(ckpt):
        return

    model = ResNetExtractor(num_classes=num_classes, mode=mode, arch=arch)
    saved = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(saved["state_dict"])
    model.to(device)

    _, val_loader = get_dataloaders(**loader_kwargs(cfg, dataset, 1.0, seed))

    tsne_layer = layer_names[-1]   # avgpool — richest representation
    feats, labels = extract_features_with_labels(
        model, val_loader, tsne_layer, 500, device
    )

    plot_feature_tsne(
        features=feats,
        labels=labels,
        mode=mode,
        dataset_name=dataset,
        fraction=1.0,
        layer_name=tsne_layer,
        num_classes=num_classes,
        output_dir=out_dir,
        dpi=cfg["visualization"]["figure_dpi"],
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CNN Transfer-Learning Analysis")
    parser.add_argument("--config",     default="config_CNN.yml")
    parser.add_argument("--arch",       default=None, help="Run a single architecture only")
    parser.add_argument("--dataset",    default=None, help="Run a single dataset only")
    parser.add_argument("--mode",       default=None, help="Run a single training mode only")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; load saved metrics and generate plots only")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs where a checkpoint already exists; train only missing ones")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    device  = get_device(cfg)
    dpi     = cfg["visualization"]["figure_dpi"]
    base_out = cfg["visualization"]["output_dir"]
    print(f"Device: {device}\n")

    archs     = [args.arch]    if args.arch    else cfg["model"]["architectures"]
    datasets  = [args.dataset] if args.dataset else cfg["datasets"]["names"]
    modes     = [args.mode]    if args.mode    else cfg["model"]["training_modes"]
    fractions = cfg["datasets"]["fractions"]
    seeds     = list(range(cfg["training"]["num_seeds"]))

    all_results: dict = {}

    for arch in archs:
        all_results[arch] = {}
        out_dir = os.path.join(base_out, arch)
        layer_names = cfg["cka"]["layers"][arch]

        print(f"\n{'='*60}\nArchitecture: {arch.upper()}\n{'='*60}")

        # ---- 1. Training loop ----
        for ds in datasets:
            all_results[arch][ds] = {}
            print(f"\n{'='*60}\nDataset: {ds.upper()}\n{'='*60}")

            for mode in modes:
                all_results[arch][ds][mode] = {}
                print(f"\n  mode: {mode}")

                for frac in fractions:
                    fkey = str(frac)
                    all_results[arch][ds][mode][fkey] = {}
                    print(f"  fraction: {int(frac * 100)}%")

                    for seed in seeds:
                        skey = str(seed)
                        print(f"    seed {seed}:")

                        ckpt = checkpoint_path(cfg, arch, ds, mode, frac, seed)
                        mpath = metrics_path(cfg, arch, ds, mode, frac, seed)
                        already_done = os.path.isfile(ckpt) and os.path.isfile(mpath)

                        if args.skip_train or (args.resume and already_done):
                            result = load_metrics(cfg, arch, ds, mode, frac, seed)
                            if result:
                                print(f"      (loaded from checkpoint)")
                        else:
                            result = run_training(cfg, arch, ds, mode, frac, seed, device, out_dir)

                        if result:
                            all_results[arch][ds][mode][fkey][skey] = result
                            print(f"      best val acc: {result.get('best_val_acc', 0):.2f}%")

        # Persist per-arch aggregate metrics
        agg_path = os.path.join(cfg["results"]["metrics_dir"], arch, "all_results.json")
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        with open(agg_path, "w") as fh:
            json.dump(all_results[arch], fh, indent=2)

        arch_results = all_results[arch]

        # ---- 2. Aggregate accuracy plots ----
        print(f"\n{'='*60}\nGenerating plots for {arch}…\n{'='*60}")

        for ds in datasets:
            plot_accuracy_vs_fraction(arch_results, ds, fractions, out_dir, dpi)

        plot_transfer_comparison(arch_results, datasets, fractions, out_dir, dpi)
        plot_transfer_gain(arch_results, datasets, fractions, out_dir, dpi)
        plot_summary_table(arch_results, datasets, fractions, modes, out_dir, dpi)

        # ---- 3. CKA analysis & evolution plots ----
        for ds in datasets:
            for mode in modes:
                cka_per_frac: dict = {}
                for frac in fractions:
                    mat, _ = run_cka(cfg, arch, ds, mode, frac, seeds[0], device, layer_names, out_dir)
                    if mat is not None:
                        cka_per_frac[str(frac)] = mat

                if cka_per_frac:
                    plot_cka_across_fractions(
                        cka_matrices=cka_per_frac,
                        layer_names=layer_names,
                        mode=mode,
                        dataset_name=ds,
                        fractions=fractions,
                        output_dir=out_dir,
                        dpi=dpi,
                    )

        # ---- 4. t-SNE visualisations ----
        for ds in datasets:
            for mode in modes:
                run_tsne(cfg, arch, ds, mode, seeds[0], device, layer_names, out_dir)

    print(f"\nAll outputs written to: {base_out}")
    print("Done.")


if __name__ == "__main__":
    main()
