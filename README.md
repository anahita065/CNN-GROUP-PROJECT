# CNN Transfer Learning Analysis — ResNet-18

This is the CNN component of our group project for CS 7643 Deep Learning at Georgia Tech. The goal is to analyze how representations learned by ResNet-18 change depending on how much training data is available and how the model is initialized (from scratch vs pretrained).

---

## What This Does

We train ResNet-18 under three conditions:

- **Scratch** — random initialization, trained entirely on the target dataset
- **Fine-tune** — starts from ImageNet pretrained weights, all layers updated
- **Linear Probe** — ImageNet pretrained weights, backbone is frozen, only the classification head is trained

Each of these is run on CIFAR-10, CIFAR-100, and Tiny ImageNet, at four different dataset sizes (10%, 25%, 50%, 100% of the training set). The idea is to see how transfer learning helps (or doesn't) as the amount of labeled data changes.

After training, we run Centered Kernel Alignment (CKA) to look at the internal representations at six different layers of the network and compare how similar or different they are across modes and dataset sizes.

---

## Why These Choices

**Why ResNet-18?**
ResNet-18 is small enough to train on CPU in a reasonable amount of time, but deep enough to have interesting intermediate representations worth analyzing. It also has well-understood ImageNet pretrained weights available through torchvision, which made it a natural fit for the transfer learning comparison. Larger models like ResNet-50 would've been harder to run across multiple seeds and fractions without heavy compute.

**Why these three modes?**
The three modes represent a spectrum of how much you lean on pretrained knowledge. Scratch is the baseline — no prior knowledge at all. Fine-tuning uses pretrained weights but lets the whole network adapt to the new dataset. Linear probe is the most constrained — it tests whether ImageNet features are useful as-is, without any adaptation. Comparing these three directly shows whether the pretrained representations are transferable, and how much adaptation is actually needed.

**Why vary the dataset size?**
This is the core question of the project. Transfer learning is most valuable when labeled data is scarce. At 100% of the data, a scratch model might eventually catch up. But at 10%, the pretrained model has a big head start. By testing at 10%, 25%, 50%, and 100%, we can see exactly where the crossover happens for each dataset.

**Why CIFAR-10, CIFAR-100, and Tiny ImageNet?**
These three datasets represent increasing task difficulty — 10 classes vs 100 classes vs 200 classes with higher resolution. They're also all visually similar enough to ImageNet that transfer learning should help, but different enough that we'd expect some variation in how well it transfers. Using three datasets gives more generalizable conclusions than just one.

**Why CKA for representation analysis?**
Accuracy alone tells you whether the model works, but not what it's actually learned internally. CKA lets us compare the feature maps at different layers — if two models have high CKA scores at a given layer, their representations are geometrically similar even if the weights look completely different. It's a way to ask "are scratch and fine-tuned models learning the same things?" at each stage of the network.

**Why those six layers?**
We picked one layer from each major stage of ResNet-18 — the initial stem (conv1), one from each of the four residual block groups (layer1 through layer4), and the global average pool right before the classifier. This gives a picture of how representations evolve from low-level edges and textures at the bottom to high-level semantic features at the top.

**Why label smoothing and gradient clipping?**
Label smoothing (ε = 0.1) stops the model from becoming overconfident on training labels, which helps generalization especially when the dataset is small. Gradient clipping prevents the occasional exploding gradient that can derail training when learning rates are a bit high. Both are standard regularization techniques that are particularly useful when training on small data fractions.

**Why cosine annealing with linear warmup?**
Starting with a small learning rate and ramping up (warmup) avoids instability in the early epochs when the model weights are far from a good solution. Cosine decay then smoothly reduces the learning rate over training rather than dropping it sharply, which tends to give better final accuracy. This schedule works well across all three training modes without needing per-mode tuning.

**Why stratified subsampling?**
When we take 10% of the data, we want to make sure all classes are still represented. Random sampling at low fractions could accidentally leave out some classes entirely, especially for CIFAR-100 with 100 classes. Stratified sampling guarantees each class gets the same fraction of samples, keeping the training set balanced.

---

## Checkpoints

Pre-trained model weights for all runs are available on Google Drive:

**[Download checkpoints](https://drive.google.com/drive/folders/1w_SDW6sDYIt6J8GlDDBWwntAjXvAc1k0?usp=sharing)**

Download and place the contents under `results/checkpoints/`. Once in place, you can skip training entirely and go straight to plot generation:

```bash
python run_cnn.py --config config_CNN_test.yml --skip-train
```

---

## Setup

You need Python 3.10+ and a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

The main dependencies are PyTorch, torchvision, matplotlib, seaborn, and scikit-learn.

**For Tiny ImageNet** — this one needs a manual download since it's not available through torchvision:

```bash
Invoke-WebRequest -Uri "http://cs231n.stanford.edu/tiny-imagenet-200.zip" -OutFile ".\data\tiny-imagenet-200.zip" -UseBasicParsing
Expand-Archive -Path ".\data\tiny-imagenet-200.zip" -DestinationPath ".\data\"
```

Then uncomment `tiny_imagenet` in `config_CNN_test.yml`.

---

## Running the Code

There are two config files. Use `config_CNN_test.yml` for a quick CPU run (small epochs, 32×32 images, 1 seed). Use `config_CNN.yml` for the full experiment (needs a GPU, 3 seeds, 224×224 images, up to 100 epochs).

**Full pipeline:**
```bash
python run_cnn.py --config config_CNN_test.yml
```

**Resume if it gets interrupted** (it saves checkpoints as it goes):
```bash
python run_cnn.py --config config_CNN_test.yml --resume
```

**Just one dataset:**
```bash
python run_cnn.py --config config_CNN_test.yml --dataset cifar10
```

**Skip training and regenerate plots from saved checkpoints:**
```bash
python run_cnn.py --config config_CNN_test.yml --skip-train
```

---

## Output Structure

Everything gets written to `results/`:

```
results/
├── checkpoints/        model weights (.pt) for each run
├── metrics/            accuracy and loss history (.json) for each run
└── plots/
    ├── training_curves/    loss and accuracy curves per run
    ├── accuracy_vs_fraction_<dataset>.png
    ├── transfer_comparison.png
    ├── transfer_gain.png
    ├── cka_heatmap_*.png
    ├── cka_across_fractions_*.png
    ├── tsne_*.png
    └── summary_table.png
```

---

## Visualizations

**Training Curves** — one plot per run showing train/val loss and accuracy over epochs. Useful for spotting overfitting or underfitting.

**Accuracy vs Fraction** — shows how validation accuracy changes as you increase the amount of training data, separately for each mode. This is probably the clearest way to see the transfer learning benefit — fine-tuning tends to stay ahead of scratch, especially at low data fractions.

**Transfer Comparison** — puts all three modes side by side across datasets. Lets you compare how much pretrained weights help on CIFAR-10 vs CIFAR-100 vs Tiny ImageNet.

**Transfer Gain** — the accuracy difference between fine-tune/linear-probe and scratch. Positive values mean pretraining helped; how this changes with dataset size is one of the main things we're analyzing.

**CKA Heatmaps** — a 6×6 grid where each cell shows how similar two layers are to each other. The six layers go from early (conv1) to deep (avgpool). Scratch-trained models typically show a more block-diagonal structure; pretrained models tend to have higher similarity across layers.

**CKA Across Fractions** — shows how the CKA values shift as you add more training data. If representations converge toward the pretrained structure as data increases, you'd see that here.

**t-SNE** — projects the final layer activations (before the classifier) down to 2D, colored by class. A well-separated plot means the network has learned a good feature space. Fine-tuned models usually show cleaner clusters, especially with less data.

**Summary Table** — a table of final validation accuracies across all datasets, modes, and fractions (mean ± std across seeds in the full config).

---

## Config Options

The main things you might want to change in `config_CNN_test.yml`:

| Field | Default | What it does |
|-------|---------|-------------|
| `num_epochs` | scratch: 3, others: 2 | Training epochs per mode |
| `fractions` | [0.10, 1.00] | Dataset size fractions to test |
| `num_seeds` | 1 | Seeds for variance estimation |
| `device` | cpu | Change to `cuda` if you have a GPU |
| `image_size` | 32 | 224 for the full config (matches ImageNet pretraining) |
| `batch_size` | 128 | Reduce if running out of memory |

---

## Notes

- CIFAR-10 and CIFAR-100 are auto-downloaded on first run
- Tiny ImageNet requires the manual download above
- The `--resume` flag is helpful if you're running on CPU and the training gets interrupted — it checks for existing checkpoints and skips completed runs
- Training curves are saved during training, but the aggregate plots (accuracy vs fraction, CKA, t-SNE, etc.) only get generated at the end once all runs are done
