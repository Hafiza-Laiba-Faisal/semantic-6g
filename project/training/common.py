"""Common training infrastructure (audit section 7, item 8).

Responsibilities (strictly separated - no model/channel/loss mathematics
lives here):

* deterministic setup via the existing config utilities
* optimizer construction (Adam, frozen project defaults)
* gradient-norm helpers and finite-value checks
* checkpoint save/load with full reproduction metadata (+ optional RNG
  state capture) under results/checkpoints (gitignored)
* configuration serialization (plain JSON-able dicts)
* history logging helpers

Checkpoint determinism, stated honestly: saving/loading model, optimizer,
step and history is exact. Restoring the process RNG states (torch global
+ purpose generators) is SUPPORTED when the caller passes their states;
the current Step-4/5 trainers create their generators internally at call
time, so bitwise mid-run continuation of those specific loops requires
exposing their generator states - a documented limitation, not silently
assumed away.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from project import config

CHECKPOINT_DIR = Path("results/checkpoints")


# ---------------------------------------------------------------------------
# deterministic setup / optimizer
# ---------------------------------------------------------------------------

def setup_deterministic(seed: int) -> Dict[str, torch.Generator]:
    """Set all global seeds and return fresh purpose-separated generators.

    Uses config.set_global_seed (Python/NumPy/Torch/CUDA). The returned
    generators are for explicit callers; nothing here relies on implicit
    global randomness afterwards.
    """
    config.set_global_seed(seed)
    return {purpose: config.derive_generator(purpose, seed)
            for purpose in config.RNG_KEYS}


def build_optimizer(model: torch.nn.Module,
                    lr: Optional[float] = None) -> torch.optim.Adam:
    """Frozen project optimizer: Adam with config.LEARNING_RATE default."""
    return torch.optim.Adam(model.parameters(),
                            lr=config.LEARNING_RATE if lr is None else lr)


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """Global L2 norm over the provided parameters' gradients."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.norm()) ** 2
    return math.sqrt(total)


def check_finite_loss(loss: torch.Tensor) -> bool:
    """True iff the loss tensor is a finite scalar."""
    return bool(torch.isfinite(loss))


def check_finite_grads(params: Iterable[torch.nn.Parameter]) -> bool:
    """True iff every present gradient is finite."""
    for p in params:
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            return False
    return True


def log_history(history: list, entry: Dict) -> list:
    """Append one JSON-able history entry and return the list (chainable)."""
    history.append(entry)
    return history


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------

def to_jsonable(obj):
    """Recursively convert dataclasses/tensors/primitives to JSON types."""
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: os.PathLike | str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    seed: Optional[int] = None,
    metadata: Optional[Dict] = None,
    history: Optional[list] = None,
    capture_rng: bool = True,
) -> Path:
    """Save a deterministic, self-describing checkpoint.

    Contents: model state, optimizer state, step, seed, metadata (method,
    k, channel, training/model configuration ...), history and (optionally)
    the torch global RNG state for continuation. No secrets are stored.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict = {
        "format_version": 1,
        "step": int(step),
        "seed": seed,
        "metadata": to_jsonable(metadata or {}),
        "history": to_jsonable(history or []),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if capture_rng:
        payload["torch_rng_state"] = torch.get_rng_state()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: os.PathLike | str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    restore_rng: bool = True,
) -> Dict:
    """Load a checkpoint into the given model (and optimizer).

    Returns the payload (step, seed, metadata, history, ...). With
    ``restore_rng=True`` the torch global RNG state is restored when
    present, so continued sampling follows the saved stream.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng and "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    return payload


def models_identical(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    """Bitwise parameter/state equality between two modules."""
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys():
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)
