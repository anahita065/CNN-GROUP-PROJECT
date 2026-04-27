"""ResNet-18 with three transfer-learning modes and forward-hook feature extraction."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List, Optional


class ResNetExtractor(nn.Module):
    """ResNet-18 supporting scratch / fine-tune / linear-probe training modes.

    Intermediate activations are extracted through PyTorch forward hooks,
    spatially averaged to a flat vector so they are directly usable for CKA.
    """

    def __init__(self, num_classes: int, mode: str = "scratch"):
        super().__init__()
        if mode not in ("scratch", "finetune", "linear_probe"):
            raise ValueError(f"mode must be scratch | finetune | linear_probe, got {mode!r}")

        weights = (
            None
            if mode == "scratch"
            else models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.backbone = models.resnet18(weights=weights)
        # Replace classifier head for the target dataset
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        if mode == "linear_probe":
            # Freeze everything; only the new head trains
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.fc.parameters():
                p.requires_grad = True

        self.mode = mode
        self._handles: Dict[str, object] = {}
        self._features: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hooks(self, layer_names: List[str]) -> None:
        """Attach a forward hook to each named layer."""
        self.remove_hooks()
        for name in layer_names:
            module = self._resolve(name)
            if module is not None:
                self._handles[name] = module.register_forward_hook(
                    self._make_hook(name)
                )

    def remove_hooks(self) -> None:
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        self._features.clear()

    def get_features(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._features.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve(self, dotted_name: str) -> Optional[nn.Module]:
        """Navigate self.backbone via a dot-separated path."""
        module = self.backbone
        for part in dotted_name.split("."):
            try:
                module = getattr(module, part)
            except AttributeError:
                try:
                    module = module[int(part)]
                except Exception:
                    return None
        return module

    def _make_hook(self, name: str):
        def hook(_, __, output):
            feat = output.detach()
            if feat.dim() == 4:  # (B, C, H, W) -> (B, C) via spatial mean
                feat = feat.mean(dim=[2, 3])
            self._features[name] = feat
        return hook

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
