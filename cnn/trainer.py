"""Training loop, evaluation, checkpointing, and metric logging for ResNet."""

import math
import os
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any, Optional


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

class AverageMeter:
    """Tracks running mean of a scalar."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [
            correct[:k].reshape(-1).float().sum().mul_(100.0 / batch_size).item()
            for k in topk
        ]


# ------------------------------------------------------------------
# Optimizer / scheduler builders
# ------------------------------------------------------------------

def build_optimizer(model, mode: str, cfg: Dict[str, Any]):
    lr = cfg["learning_rates"][mode]
    wd = cfg["weight_decay"]
    params = model.get_trainable_params()
    if cfg["optimizer"] == "sgd":
        return SGD(params, lr=lr, momentum=cfg.get("momentum", 0.9), weight_decay=wd)
    if cfg["optimizer"] == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {cfg['optimizer']!r}")


def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int) -> LambdaLR:
    """Linear warmup followed by cosine annealing (per-epoch updates)."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class Trainer:
    """Runs training + evaluation for a single (model, mode, dataset) configuration."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        mode: str,
        cfg: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mode = mode
        self.device = device
        # Label smoothing (0.1) regularises against overconfident predictions
        label_smoothing = cfg.get("label_smoothing", 0.1)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.grad_clip = cfg.get("grad_clip", 1.0)

        self.num_epochs = cfg["num_epochs"][mode]
        self.optimizer = build_optimizer(model, mode, cfg)
        self.scheduler = build_scheduler(
            self.optimizer, self.num_epochs, cfg["scheduler"]["warmup_epochs"]
        )
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

    # ------------------------------------------------------------------

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        loss_m, acc_m = AverageMeter(), AverageMeter()
        for images, targets in self.train_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(images)
            loss = self.criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_trainable_params(), self.grad_clip)
            self.optimizer.step()
            acc = topk_accuracy(out, targets, topk=(1,))[0]
            loss_m.update(loss.item(), images.size(0))
            acc_m.update(acc, images.size(0))
        return {"loss": loss_m.avg, "acc": acc_m.avg}

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.model.eval()
        loss_m, acc_m = AverageMeter(), AverageMeter()
        for images, targets in self.val_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            out = self.model(images)
            loss = self.criterion(out, targets)
            acc = topk_accuracy(out, targets, topk=(1,))[0]
            loss_m.update(loss.item(), images.size(0))
            acc_m.update(acc, images.size(0))
        return {"loss": loss_m.avg, "acc": acc_m.avg}

    # ------------------------------------------------------------------

    def fit(self, checkpoint_path: Optional[str] = None) -> Dict:
        best_val_acc = 0.0
        best_state: Optional[Dict] = None

        for epoch in range(1, self.num_epochs + 1):
            train_m = self._train_epoch()
            val_m   = self._evaluate()
            self.scheduler.step()

            self.history["train_loss"].append(train_m["loss"])
            self.history["train_acc"].append(train_m["acc"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["val_acc"].append(val_m["acc"])

            if val_m["acc"] > best_val_acc:
                best_val_acc = val_m["acc"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if epoch % 10 == 0 or epoch == self.num_epochs:
                print(
                    f"    epoch {epoch:3d}/{self.num_epochs}  "
                    f"train {train_m['loss']:.4f}/{train_m['acc']:.1f}%  "
                    f"val {val_m['loss']:.4f}/{val_m['acc']:.1f}%  "
                    f"best {best_val_acc:.1f}%"
                )

        if checkpoint_path and best_state is not None:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({"state_dict": best_state, "best_val_acc": best_val_acc}, checkpoint_path)

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return {"best_val_acc": best_val_acc, "history": self.history}
