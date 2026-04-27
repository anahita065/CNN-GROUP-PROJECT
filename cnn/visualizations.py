"""All visualisation functions for the CNN transfer-learning analysis."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from typing import Dict, List, Any

# ------------------------------------------------------------------
# Shared style
# ------------------------------------------------------------------

PALETTE = {
    "scratch":      "#E74C3C",
    "finetune":     "#3498DB",
    "linear_probe": "#2ECC71",
}
MODE_LABELS = {
    "scratch":      "From Scratch",
    "finetune":     "Fine-tune (ImageNet)",
    "linear_probe": "Linear Probe (ImageNet)",
}
DATASET_LABELS = {
    "cifar10":       "CIFAR-10",
    "cifar100":      "CIFAR-100",
    "tiny_imagenet": "Tiny ImageNet",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _save(fig, path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def _seed_accs(results: Dict, dataset: str, mode: str, frac_key: str) -> List[float]:
    """Safely collect per-seed best-val-acc values."""
    try:
        return [
            results[dataset][mode][frac_key][s]["best_val_acc"]
            for s in results[dataset][mode][frac_key]
        ]
    except KeyError:
        return []


# ------------------------------------------------------------------
# 1. Accuracy vs dataset fraction  (per dataset, all modes)
# ------------------------------------------------------------------

def plot_accuracy_vs_fraction(
    results: Dict,
    dataset_name: str,
    fractions: List[float],
    output_dir: str,
    dpi: int = 150,
):
    """Line plot with ± 1 std error band for each training mode."""
    fig, ax = plt.subplots(figsize=(7, 5))
    frac_pct = [f * 100 for f in fractions]

    for mode in ("scratch", "finetune", "linear_probe"):
        means, stds = [], []
        for frac in fractions:
            accs = _seed_accs(results, dataset_name, mode, str(frac))
            means.append(np.mean(accs) if accs else 0.0)
            stds.append(np.std(accs)  if accs else 0.0)

        ax.plot(frac_pct, means, marker="o", linewidth=2,
                color=PALETTE[mode], label=MODE_LABELS[mode])
        ax.fill_between(frac_pct,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=PALETTE[mode])

    ax.set_xlabel("Training Data Fraction (%)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title(f"Accuracy vs. Dataset Size — {DATASET_LABELS[dataset_name]}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(frac_pct)
    ax.set_xlim(frac_pct[0] - 3, 105)
    _save(fig, os.path.join(output_dir, f"accuracy_vs_fraction_{dataset_name}.png"), dpi)


# ------------------------------------------------------------------
# 2. Transfer-learning comparison bar chart  (all datasets × fractions)
# ------------------------------------------------------------------

def plot_transfer_comparison(
    results: Dict,
    dataset_names: List[str],
    fractions: List[float],
    output_dir: str,
    dpi: int = 150,
):
    """Grouped bar chart comparing modes at each fraction per dataset."""
    modes = ["scratch", "finetune", "linear_probe"]
    n_ds = len(dataset_names)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    x = np.arange(len(fractions))
    w = 0.25

    for ax, ds in zip(axes, dataset_names):
        for i, mode in enumerate(modes):
            means, errs = [], []
            for frac in fractions:
                accs = _seed_accs(results, ds, mode, str(frac))
                means.append(np.mean(accs) if accs else 0.0)
                errs.append(np.std(accs)   if accs else 0.0)
            offset = (i - 1) * w
            ax.bar(x + offset, means, w, yerr=errs, capsize=3,
                   label=MODE_LABELS[mode], color=PALETTE[mode], alpha=0.85)
        ax.set_title(DATASET_LABELS[ds], fontsize=12)
        ax.set_xlabel("Dataset Fraction", fontsize=11)
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Transfer Learning Mode Comparison — ResNet-18", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "transfer_comparison.png"), dpi)


# ------------------------------------------------------------------
# 3. Transfer-learning accuracy gain over training from scratch
# ------------------------------------------------------------------

def plot_transfer_gain(
    results: Dict,
    dataset_names: List[str],
    fractions: List[float],
    output_dir: str,
    dpi: int = 150,
):
    """Show (fine-tune / linear-probe) − scratch accuracy at each fraction."""
    n_ds = len(dataset_names)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=False)
    if n_ds == 1:
        axes = [axes]

    frac_pct = [f * 100 for f in fractions]

    for ax, ds in zip(axes, dataset_names):
        for mode in ("finetune", "linear_probe"):
            gains, stds = [], []
            for frac in fractions:
                scratch_accs = _seed_accs(results, ds, "scratch", str(frac))
                mode_accs    = _seed_accs(results, ds, mode,      str(frac))
                if scratch_accs and mode_accs:
                    gain = np.mean(mode_accs) - np.mean(scratch_accs)
                    std  = np.sqrt(np.var(mode_accs) + np.var(scratch_accs))
                else:
                    gain, std = 0.0, 0.0
                gains.append(gain)
                stds.append(std)

            ax.plot(frac_pct, gains, marker="o", linewidth=2,
                    color=PALETTE[mode], label=MODE_LABELS[mode])
            ax.fill_between(frac_pct,
                            [g - s for g, s in zip(gains, stds)],
                            [g + s for g, s in zip(gains, stds)],
                            alpha=0.15, color=PALETTE[mode])

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Training Data Fraction (%)", fontsize=11)
        ax.set_ylabel("Accuracy Gain over Scratch (pp)", fontsize=11)
        ax.set_title(DATASET_LABELS[ds], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(frac_pct)

    fig.suptitle(
        "Transfer Learning Gain over Training from Scratch — ResNet-18",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "transfer_gain.png"), dpi)


# ------------------------------------------------------------------
# 4. Training / validation curves for one run
# ------------------------------------------------------------------

def plot_training_curves(
    history: Dict,
    mode: str,
    dataset_name: str,
    fraction: float,
    seed: int,
    output_dir: str,
    dpi: int = 150,
):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#E74C3C")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#3498DB")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title(
        f"Loss — {MODE_LABELS[mode]} | {DATASET_LABELS[dataset_name]} | {int(fraction*100)}%"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", color="#E74C3C")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#3498DB")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 Accuracy (%)")
    axes[1].set_title(
        f"Accuracy — {MODE_LABELS[mode]} | {DATASET_LABELS[dataset_name]} | {int(fraction*100)}%"
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fname = f"curves_{dataset_name}_{mode}_frac{int(fraction*100)}_seed{seed}.png"
    _save(fig, os.path.join(output_dir, "training_curves", fname), dpi)


# ------------------------------------------------------------------
# 5. CKA heatmap for one (mode, dataset, fraction)
# ------------------------------------------------------------------

def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    layer_names: List[str],
    mode: str,
    dataset_name: str,
    fraction: float,
    output_dir: str,
    dpi: int = 150,
):
    short = [n.replace(".", "\n") for n in layer_names]
    n = len(short)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap="magma", aspect="auto")
    plt.colorbar(im, ax=ax, label="CKA Similarity")

    ax.set_xticks(range(n)); ax.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=7)
    ax.set_title(
        f"Layer CKA — {MODE_LABELS[mode]} | {DATASET_LABELS[dataset_name]} | "
        f"{int(fraction * 100)}% data",
        fontsize=11,
    )

    # Annotate each cell
    thresh = 0.55
    for i in range(n):
        for j in range(n):
            val = cka_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if val < thresh else "black")

    fig.tight_layout()
    fname = f"cka_{dataset_name}_{mode}_frac{int(fraction * 100)}.png"
    _save(fig, os.path.join(output_dir, "cka_heatmaps", fname), dpi)


# ------------------------------------------------------------------
# 6. CKA evolution across dataset fractions (fixed mode & dataset)
# ------------------------------------------------------------------

def plot_cka_across_fractions(
    cka_matrices: Dict[str, np.ndarray],   # str(frac) -> matrix
    layer_names: List[str],
    mode: str,
    dataset_name: str,
    fractions: List[float],
    output_dir: str,
    dpi: int = 150,
):
    available = [f for f in fractions if str(f) in cka_matrices]
    n = len(available)
    if n == 0:
        return

    short = [nm.split(".")[-1] if "." in nm else nm for nm in layer_names]
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, frac in zip(axes, available):
        mat = cka_matrices[str(frac)]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="magma", aspect="auto")
        ax.set_title(f"{int(frac * 100)}% data", fontsize=10)
        ax.set_xticks(range(len(short)))
        ax.set_yticks(range(len(short)))
        ax.set_xticklabels(short, fontsize=6, rotation=45, ha="right")
        ax.set_yticklabels(short, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"CKA Evolution with Dataset Size — {MODE_LABELS[mode]} | "
        f"{DATASET_LABELS[dataset_name]}",
        fontsize=12,
    )
    fig.tight_layout()
    fname = f"cka_evolution_{dataset_name}_{mode}.png"
    _save(fig, os.path.join(output_dir, "cka_heatmaps", fname), dpi)


# ------------------------------------------------------------------
# 7. t-SNE feature visualisation
# ------------------------------------------------------------------

def plot_feature_tsne(
    features: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
    dataset_name: str,
    fraction: float,
    layer_name: str,
    num_classes: int,
    output_dir: str,
    dpi: int = 150,
):
    X = features.numpy()
    y = labels.numpy()

    perplexity = min(30, max(5, X.shape[0] // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X2d = tsne.fit_transform(X)

    n_show = min(num_classes, 10)   # cap legend at 10 classes for readability
    palette = sns.color_palette("tab10", n_show)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in range(n_show):
        mask = y == cls
        if mask.sum() > 0:
            ax.scatter(X2d[mask, 0], X2d[mask, 1], s=10, alpha=0.6,
                       color=palette[cls], label=f"class {cls}")

    ax.set_title(
        f"t-SNE of [{layer_name}] — {MODE_LABELS[mode]} | "
        f"{DATASET_LABELS[dataset_name]} | {int(fraction * 100)}% data",
        fontsize=11,
    )
    ax.legend(markerscale=2, fontsize=8, loc="upper right",
              ncol=2 if n_show > 5 else 1)
    ax.set_axis_off()
    fig.tight_layout()

    layer_tag = layer_name.replace(".", "_")
    fname = f"tsne_{dataset_name}_{mode}_frac{int(fraction * 100)}_{layer_tag}.png"
    _save(fig, os.path.join(output_dir, "tsne", fname), dpi)


# ------------------------------------------------------------------
# 8. Summary results table
# ------------------------------------------------------------------

def plot_summary_table(
    results: Dict,
    dataset_names: List[str],
    fractions: List[float],
    modes: List[str],
    output_dir: str,
    dpi: int = 150,
):
    rows = []
    for ds in dataset_names:
        for mode in modes:
            row = [DATASET_LABELS.get(ds, ds), MODE_LABELS.get(mode, mode)]
            for frac in fractions:
                accs = _seed_accs(results, ds, mode, str(frac))
                if accs:
                    row.append(f"{np.mean(accs):.1f} ± {np.std(accs):.1f}")
                else:
                    row.append("—")
            rows.append(row)

    col_labels = ["Dataset", "Mode"] + [f"{int(f * 100)}%" for f in fractions]
    n_rows = len(rows)

    fig, ax = plt.subplots(figsize=(11, 0.55 * (n_rows + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ECF0F1")
        cell.set_edgecolor("#BDC3C7")

    ax.set_title(
        "ResNet-18 — Top-1 Accuracy (mean ± std %) by Dataset, Mode, and Training Fraction",
        fontsize=11, pad=14,
    )
    _save(fig, os.path.join(output_dir, "summary_table.png"), dpi)
