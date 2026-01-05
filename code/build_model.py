from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn as nn
import timm

import yaml  # pip install pyyaml


class ConfigError(ValueError):
    pass


def _load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ConfigError(f"Unsupported config extension '{suffix}'. Use .yaml/.yml or .json")


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    arch: str
    num_classes: int
    in_chans: int
    pretrained: bool
    img_size: Optional[int] = None
    timm_kwargs: Dict[str, Any] = None  # will be normalized to dict in builder


class BuildModel:
    """
    Factory for time-of-day classification models.

    Config format:
      global:
        num_classes: 24
        in_chans: 3
        pretrained: true

      models:
        resnet50:
          arch: resnet50
          timm_kwargs: {}
        efficientnet_b0:
          arch: efficientnet_b0
        vit_tiny:
          arch: vit_tiny_patch16_224
          img_size: 224

    Resolution rule (NO deep merge):
      - Start from global scalar defaults: num_classes, in_chans, pretrained
      - If model overrides exist for those keys, use them
      - timm_kwargs is taken from the model section only (no merging with global)
    """

    ALLOWED = {"resnet50", "efficientnet_b0", "vit_tiny", "vit_small"}
    GLOBAL_KEYS = {"num_classes", "in_chans", "pretrained"}

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.cfg = _load_config(self.config_path)

        if not isinstance(self.cfg, dict):
            raise ConfigError("Config root must be a mapping/dict.")

        if "global" not in self.cfg or "models" not in self.cfg:
            raise ConfigError("Config must contain top-level keys: 'global' and 'models'.")

        if not isinstance(self.cfg["global"], dict):
            raise ConfigError("'global' must be a dict.")
        if not isinstance(self.cfg["models"], dict):
            raise ConfigError("'models' must be a mapping from model_name -> config dict.")

    def _get_model_spec(self, model_name: str) -> ModelSpec:
        if model_name not in self.ALLOWED:
            raise ConfigError(f"model_name must be one of {sorted(self.ALLOWED)}. Got: {model_name}")

        global_cfg: Dict[str, Any] = self.cfg.get("global", {}) or {}
        models_cfg: Dict[str, Any] = self.cfg.get("models", {}) or {}

        if model_name not in models_cfg:
            raise ConfigError(f"Config missing entry for models.{model_name}")

        model_cfg = models_cfg[model_name]
        if not isinstance(model_cfg, dict):
            raise ConfigError(f"models.{model_name} must be a dict.")

        # Required: arch must exist in model section
        arch = model_cfg.get("arch")
        if not arch or not isinstance(arch, str):
            raise ConfigError(f"models.{model_name}.arch is required and must be a string (timm architecture).")

        # Scalar defaults from global, overridden by model if provided (no merging beyond these keys)
        def _get_scalar(key: str, default: Any) -> Any:
            if key in model_cfg:
                return model_cfg[key]
            if key in global_cfg:
                return global_cfg[key]
            return default

        num_classes = int(_get_scalar("num_classes", 24))
        in_chans = int(_get_scalar("in_chans", 3))
        pretrained = bool(_get_scalar("pretrained", True))

        # img_size is model-only (typical)
        img_size = model_cfg.get("img_size", None)
        if img_size is not None:
            img_size = int(img_size)

        # timm_kwargs is model-only and must be a dict if present
        timm_kwargs = model_cfg.get("timm_kwargs", {})
        if timm_kwargs is None:
            timm_kwargs = {}
        if not isinstance(timm_kwargs, dict):
            raise ConfigError(f"models.{model_name}.timm_kwargs must be a dict if provided.")

        return ModelSpec(
            model_name=model_name,
            arch=arch,
            num_classes=num_classes,
            in_chans=in_chans,
            pretrained=pretrained,
            img_size=img_size,
            timm_kwargs=timm_kwargs,
        )

    def build(self, model_name: str) -> nn.Module:
        spec = self._get_model_spec(model_name)

        # timm args: model-only timm_kwargs + mandatory overrides
        kwargs: Dict[str, Any] = dict(spec.timm_kwargs or {})
        kwargs["pretrained"] = spec.pretrained
        kwargs["in_chans"] = spec.in_chans
        kwargs["num_classes"] = spec.num_classes

        if spec.img_size is not None:
            kwargs["img_size"] = spec.img_size

        return timm.create_model(spec.arch, **kwargs)

    def describe(self, model_name: str) -> Dict[str, Any]:
        spec = self._get_model_spec(model_name)
        return {
            "model_name": spec.model_name,
            "arch": spec.arch,
            "num_classes": spec.num_classes,
            "in_chans": spec.in_chans,
            "pretrained": spec.pretrained,
            "img_size": spec.img_size,
            "timm_kwargs": spec.timm_kwargs or {},
        }


def main() -> None:
    config_path = "models_config.yaml"
    builder = BuildModel(config_path)

    for name in ["resnet50", "efficientnet_b0", "vit_tiny"]:
        model = builder.build(name)
        info = builder.describe(name)
        print(f"\nBuilt: {name}")
        print(info)
        print("Params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")


if __name__ == "__main__":
    main()
