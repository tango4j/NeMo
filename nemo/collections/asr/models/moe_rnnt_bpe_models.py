"""
RNNT BPE model with Mixture-of-Experts (MoE) encoder auxiliary loss support.

Extends EncDecRNNTBPEModel to collect and add the MoE load-balancing auxiliary
loss from MoETransformerEncoder during training.
"""

import torch

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

__all__ = ['EncDecMoERNNTBPEModel']


class EncDecMoERNNTBPEModel(EncDecRNNTBPEModel):
    """Encoder-Decoder RNNT BPE model with MoE encoder auxiliary loss support.

    Inherits all functionality from :class:`EncDecRNNTBPEModel` and overrides
    :meth:`add_auxiliary_losses` to collect the MoE load-balancing loss from the
    encoder during training.

    Use this model class with :class:`MoETransformerEncoder` to train RNNT models
    with MoE feed-forward layers and the auxiliary load-balancing loss.
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
