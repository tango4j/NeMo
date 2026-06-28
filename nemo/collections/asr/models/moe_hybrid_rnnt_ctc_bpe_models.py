"""
Hybrid RNNT-CTC BPE model with Mixture-of-Experts (MoE) encoder auxiliary loss support.

Extends EncDecHybridRNNTCTCBPEModel to collect and add the MoE load-balancing
auxiliary loss from MoETransformerEncoder during training, and to log MoE
routing diagnostics (load CV, router entropy, dead-expert count, per-layer
histograms, expert-load heatmap) to W&B in a multinode-safe way.
"""

import torch

from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.utils.moe_logging import log_moe_diagnostics

__all__ = ['EncDecMoEHybridRNNTCTCBPEModel']


class EncDecMoEHybridRNNTCTCBPEModel(EncDecHybridRNNTCTCBPEModel):
    """Hybrid RNNT-CTC BPE model with MoE encoder auxiliary loss support.

    Inherits all functionality from :class:`EncDecHybridRNNTCTCBPEModel` and:

    * Overrides :meth:`add_auxiliary_losses` to collect the MoE load-balancing
      loss from the encoder during training.
    * Overrides :meth:`on_train_batch_end` to log MoE routing diagnostics to
      whatever Lightning logger is attached (W&B in our setup).

    Use this model class with :class:`MoETransformerEncoder` to train Hybrid
    RNNT-CTC models with MoE feed-forward layers and the auxiliary load-balancing loss.
    """

    def add_auxiliary_losses(self, loss: torch.Tensor, reset_registry: bool = False) -> torch.Tensor:
        """Add auxiliary losses including MoE load-balancing loss.

        Args:
            loss: The primary loss value.
            reset_registry: Whether to reset the AccessMixin registry.

        Returns:
            Loss tensor with auxiliary losses added.
        """
        loss = super().add_auxiliary_losses(loss, reset_registry=reset_registry)

        if hasattr(self.encoder, 'get_moe_auxiliary_loss'):
            moe_loss = self.encoder.get_moe_auxiliary_loss()
            if moe_loss is not None and moe_loss.requires_grad:
                loss = loss + moe_loss
                self.log('moe_aux_loss', moe_loss.detach())

        return loss

    def on_train_batch_end(self, *args, **kwargs):  # noqa: D401 (Lightning hook)
        """Log MoE routing diagnostics at the configured cadence.

        Cadence is taken from ``trainer.log_every_n_steps`` (scalars) and
        ``trainer.val_check_interval`` (histograms / heatmap).
        """
        super().on_train_batch_end(*args, **kwargs)
        log_moe_diagnostics(self)
