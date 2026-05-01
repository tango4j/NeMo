"""Logging helpers for MoE Transformer encoders.

Provides :func:`log_moe_diagnostics` -- a single function that should be
called from a Lightning ``on_train_batch_end`` hook on a model whose
``encoder`` is a :class:`MoETransformerEncoder`. It:

* Triggers a tiny all-reduce and reads the encoder's cumulative routing stats.
* Logs scalar diagnostics (load CV, router entropy, dead-expert count, etc.)
  to whatever Lightning logger is attached, every
  ``trainer.log_every_n_steps`` steps.
* On rank 0 only, every ``trainer.val_check_interval`` steps, also logs to
  W&B:
    - a flat histogram of per-expert load fractions across all MoE layers,
    - per-layer histograms of expert load,
    - a ``[n_moe_layers, num_experts]`` heatmap of expert load.

All cadence checks rely on Lightning's rank-synchronized
``model.global_step``, so every rank reaches the all-reduce together and we
cannot deadlock on the NCCL group. Network payload per call is
``O(n_moe_layers * num_experts)`` floats / longs (kilobytes).
"""

from __future__ import annotations

import io
from typing import Optional


def _resolve_cadence(
    model, scalar_every: Optional[int], histogram_every: Optional[int]
):
    """Pick scalar / histogram logging cadences from explicit args or trainer."""
    if scalar_every is None:
        if model.trainer is not None and getattr(model.trainer, "log_every_n_steps", 0):
            scalar_every = int(model.trainer.log_every_n_steps)
        else:
            scalar_every = 100
    if histogram_every is None:
        if model.trainer is not None and getattr(model.trainer, "val_check_interval", 0):
            v = model.trainer.val_check_interval
            histogram_every = int(v) if isinstance(v, int) and v > 0 else max(scalar_every * 25, 2500)
        else:
            histogram_every = max(scalar_every * 25, 2500)
    return max(int(scalar_every), 1), max(int(histogram_every), 1)


def _is_wandb_logger(logger) -> bool:
    if logger is None:
        return False
    cls_name = type(logger).__name__
    if cls_name == "WandbLogger":
        return True
    return any(type(lg).__name__ == "WandbLogger" for lg in getattr(logger, "_logger_iterable", []) or [])


def _wandb_run(model):
    """Return the underlying wandb Run, or None if the W&B logger isn't attached."""
    logger = getattr(model, "logger", None)
    if logger is None:
        return None
    candidates = [logger]
    if hasattr(logger, "_logger_iterable") and logger._logger_iterable:
        candidates.extend(list(logger._logger_iterable))
    for lg in candidates:
        if type(lg).__name__ == "WandbLogger":
            try:
                return lg.experiment
            except Exception:
                return None
    return None


def _log_histograms_and_heatmap(model, per_layer):
    """Rank-0 only. Pushes per-layer histograms + a heatmap of expert load to W&B.

    Robust to missing matplotlib / wandb (no-op if unavailable).
    """
    run = _wandb_run(model)
    if run is None:
        return

    try:
        import wandb
    except Exception:
        return

    load = per_layer["load"]  # CPU tensor [L, E]
    L, E = load.shape
    step = model.global_step

    payload = {}

    # Aggregate histogram across all (layer, expert) pairs.
    flat = load.reshape(-1).numpy()
    payload["moe/expert_load_global"] = wandb.Histogram(flat)

    # One histogram per MoE layer (W&B handles 32 series fine).
    for l in range(L):
        payload[f"moe/expert_load_per_layer/layer_{l:02d}"] = wandb.Histogram(
            load[l].numpy()
        )

    # Heatmap of expert load (layer x expert). Optional: needs matplotlib.
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(4, 0.3 * E + 2), max(4, 0.18 * L + 1)))
        im = ax.imshow(load.numpy(), aspect="auto", cmap="viridis", vmin=0)
        ax.set_xlabel("expert")
        ax.set_ylabel("MoE layer slot")
        ax.set_title(f"MoE expert load (step {step})")
        fig.colorbar(im, ax=ax, label="dispatch fraction")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        payload["moe/expert_load_heatmap"] = wandb.Image(buf, caption=f"step {step}")
    except Exception:
        pass

    try:
        run.log(payload, step=step)
    except Exception:
        pass


def log_moe_diagnostics(
    model,
    scalar_every: Optional[int] = None,
    histogram_every: Optional[int] = None,
) -> None:
    """Lightning ``on_train_batch_end`` hook for MoE diagnostics.

    Args:
        model: A LightningModule with ``encoder = MoETransformerEncoder``.
        scalar_every: Log scalars every N steps. ``None`` => use
            ``trainer.log_every_n_steps`` (fallback 100).
        histogram_every: Log histograms / heatmap every N steps. ``None`` =>
            use ``trainer.val_check_interval`` (fallback 2500).
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "get_moe_metrics"):
        return

    scalar_every, histogram_every = _resolve_cadence(model, scalar_every, histogram_every)
    step = int(model.global_step)
    next_step = step + 1  # we hook AT the END of step `step`

    # Cadence triggers must be identical on every DDP rank. global_step is
    # Lightning-synchronized, so they are.
    fire_scalars = (next_step % scalar_every) == 0
    fire_histograms = (next_step % histogram_every) == 0

    if not (fire_scalars or fire_histograms):
        return

    metrics = encoder.get_moe_metrics(distributed=True, reset=True)
    if metrics is None:
        return

    # Scalars: Lightning's `log()` handles rank-0-only routing for us.
    for k, v in metrics["scalars"].items():
        try:
            model.log(
                k,
                v,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                rank_zero_only=False,
                sync_dist=False,  # values are already global (post all-reduce)
            )
        except TypeError:
            # Older Lightning may not accept rank_zero_only kwarg; fall back.
            model.log(k, v, on_step=True, on_epoch=False, prog_bar=False)

    if fire_histograms and getattr(model, "global_rank", 0) == 0:
        _log_histograms_and_heatmap(model, metrics["per_layer"])
