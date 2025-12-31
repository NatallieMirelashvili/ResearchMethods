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
        if yaml is None:
            raise ImportError("YAML config requested but PyYAML is not installed. Run: pip install pyyaml")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ConfigError(f"Unsupported config extension '{suffix}'. Use .yaml/.yml or .json")




@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    arch: str
    num_classes: int
    in_chans: int
    pretrained: bool
    img_size: Optional[int] = None
    timm_kwargs: Optional[Dict[str, Any]] = None


class BuildModel:
    """
    Factory for time-of-day classification models.

    Usage:
        builder = BuildModel(config_path="models_config.yaml")
        model = builder.build("resnet50")  # or efficientnet_b0 / vit_tiny / vit_small
    """

    ALLOWED = {"resnet50", "efficientnet_b0", "vit_tiny", "vit_small"}

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.cfg = _load_config(self.config_path)

        if "global" not in self.cfg or "models" not in self.cfg:
            raise ConfigError("Config must contain top-level keys: 'global' and 'models'.")

        # Basic validation
        if not isinstance(self.cfg["models"], dict):
            raise ConfigError("'models' must be a mapping from model_name -> config dict.")

    def _get_model_spec(self, model_name: str) -> ModelSpec:
        if model_name not in self.ALLOWED:
            raise ConfigError(f"model_name must be one of {sorted(self.ALLOWED)}. Got: {model_name}")

        global_cfg = self.cfg.get("global", {}) or {}
        models_cfg = self.cfg.get("models", {}) or {}

        if model_name not in models_cfg:
            raise ConfigError(f"Config missing entry for models.{model_name}")

        merged = _deep_merge(global_cfg, models_cfg[model_name])

        # Required
        arch = merged.get("arch")
        if not arch:
            raise ConfigError(f"models.{model_name}.arch is required (timm architecture string).")

        # Defaults with validation
        num_classes = int(merged.get("num_classes", 24))
        in_chans = int(merged.get("in_chans", 3))
        pretrained = bool(merged.get("pretrained", True))

        img_size = merged.get("img_size", None)
        if img_size is not None:
            img_size = int(img_size)

        # Extra timm args (optional)
        timm_kwargs = merged.get("timm_kwargs", None)
        if timm_kwargs is not None and not isinstance(timm_kwargs, dict):
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
        """
        Build and return a torch.nn.Module according to config.
        """
        spec = self._get_model_spec(model_name)

        # Core timm args
        kwargs: Dict[str, Any] = dict(spec.timm_kwargs or {})
        kwargs["pretrained"] = spec.pretrained
        kwargs["in_chans"] = spec.in_chans
        kwargs["num_classes"] = spec.num_classes

        # ViT sometimes wants img_size explicitly if you deviate from 224.
        if spec.img_size is not None:
            kwargs["img_size"] = spec.img_size

        model = timm.create_model(spec.arch, **kwargs)
        return model

    def describe(self, model_name: str) -> Dict[str, Any]:
        """
        Returns the resolved merged config for this model (useful for logging).
        """
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
    # Example
    config_path = "models_config.yaml"
    builder = BuildModel(config_path)

    for name in ["resnet50", "efficientnet_b0", "vit_tiny"]:
        model = builder.build(name)
        info = builder.describe(name)
        print(f"\nBuilt: {name}")
        print(info)
        # sanity: print classifier head name/shape
        # (varies by architecture)
        print("Params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")


if __name__ == "__main__":
    main()
