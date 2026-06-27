# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager

import torch
from torch.nn import Module

from nemo.core.classes.common import FileIO, Serialization, Typing
from nemo.utils import logging

__all__ = ['NeuralModule', 'freeze', 'unfreeze']


def freeze(module: Module) -> None:
    """Freeze all parameters of ``module`` and snapshot their prior ``requires_grad`` state.

    The snapshot is stored on ``module._frozen_grad_map`` so a later call to ``unfreeze(..., partial=True)``
    can restore the pre-freeze state instead of unconditionally enabling gradients.
    """
    grad_map = {pname: param.requires_grad for pname, param in module.named_parameters()}
    for param in module.parameters():
        param.requires_grad = False
    if not hasattr(module, '_frozen_grad_map'):
        module._frozen_grad_map = grad_map
    else:
        module._frozen_grad_map.update(grad_map)
    module.eval()


def unfreeze(module: Module, partial: bool = False) -> None:
    """Unfreeze parameters of ``module``.

    If ``partial=True``, restore each parameter's ``requires_grad`` from the snapshot recorded by
    ``freeze(module)``; otherwise enable gradients on every parameter. The snapshot is cleared in
    both cases and ``module.train()`` is called.
    """
    if partial and not hasattr(module, '_frozen_grad_map'):
        raise ValueError("Cannot unfreeze partially without first freezing the module with `freeze()`")

    for pname, param in module.named_parameters():
        if not partial:
            param.requires_grad = True
        elif pname in module._frozen_grad_map:
            param.requires_grad = module._frozen_grad_map[pname]
        else:
            logging.warning(
                f"Parameter {pname} not found in list of previously frozen parameters. Unfreezing this parameter."
            )
            param.requires_grad = True

    if hasattr(module, '_frozen_grad_map'):
        delattr(module, '_frozen_grad_map')

    module.train()


class NeuralModule(Module, Typing, Serialization, FileIO):
    """
    Abstract class offering interface shared between all PyTorch Neural Modules.
    """

    @property
    def num_weights(self):
        """
        Utility property that returns the total number of parameters of NeuralModule.
        """
        return self._num_weights()

    @torch.jit.ignore
    def _num_weights(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num

    def input_example(self, max_batch=None, max_dim=None):
        """
        Override this method if random inputs won't work
        Returns:
            A tuple sample of valid input data.
        """

        return None

    def freeze(self) -> None:
        r"""
        Freeze all params for inference.

        This method sets `requires_grad` to False for all parameters of the module.
        It also stores the original `requires_grad` state of each parameter in a dictionary,
        so that `unfreeze()` can restore the original state if `partial=True` is set in `unfreeze()`.
        """
        grad_map = {}

        for pname, param in self.named_parameters():
            # Store the original grad state
            grad_map[pname] = param.requires_grad
            # Freeze the parameter
            param.requires_grad = False

        # Store the frozen grad map
        if not hasattr(self, '_frozen_grad_map'):
            self._frozen_grad_map = grad_map
        else:
            self._frozen_grad_map.update(grad_map)

        self.eval()

    def unfreeze(self, partial: bool = False) -> None:
        """
        Unfreeze all parameters for training.

        Allows for either total unfreeze or partial unfreeze (if the module was explicitly frozen previously with `freeze()`).
        The `partial` argument is used to determine whether to unfreeze all parameters or only the parameters that were
        previously unfrozen prior `freeze()`.

        Example:
            Consider a model that has an encoder and a decoder module. Assume we want the encoder to be frozen always.

            ```python
            model.encoder.freeze()  # Freezes all parameters in the encoder explicitly
            ```

            During inference, all parameters of the model should be frozen - we do this by calling the model's freeze method.
            This step records that the encoder module parameters were already frozen, and so if partial unfreeze is called,
            we should keep the encoder parameters frozen.

            ```python
            model.freeze()  # Freezes all parameters in the model; encoder remains frozen
            ```

            Now, during fine-tuning, we want to unfreeze the decoder but keep the encoder frozen. We can do this by calling
            `unfreeze(partial=True)`.

            ```python
            model.unfreeze(partial=True)  # Unfreezes only the decoder; encoder remains frozen
            ```

        Args:
            partial: If True, only unfreeze parameters that were previously frozen. If the parameter was already frozen
                when calling `freeze()`, it will remain frozen after calling `unfreeze(partial=True)`.
        """
        if partial and not hasattr(self, '_frozen_grad_map'):
            raise ValueError("Cannot unfreeze partially without first freezing the module with `freeze()`")

        for pname, param in self.named_parameters():
            if not partial:
                # Unfreeze all parameters
                param.requires_grad = True
            else:
                # Unfreeze only parameters that were previously frozen

                # Check if the parameter was frozen
                if pname in self._frozen_grad_map:
                    param.requires_grad = self._frozen_grad_map[pname]
                else:
                    # Log a warning if the parameter was not found in the frozen grad map
                    logging.warning(
                        f"Parameter {pname} not found in list of previously frozen parameters. "
                        f"Unfreezing this parameter."
                    )
                    param.requires_grad = True

        # Clean up the frozen grad map
        if hasattr(self, '_frozen_grad_map'):
            delattr(self, '_frozen_grad_map')

        self.train()

    @contextmanager
    def as_frozen(self):
        """
        Context manager which temporarily freezes a module, yields control and finally unfreezes the module partially
        to return to original state.

        Allows for either total unfreeze or partial unfreeze (if the module was explicitly frozen previously with `freeze()`).
        The `partial` argument is used to determine whether to unfreeze all parameters or only the parameters that were
        previously unfrozen prior `freeze()`.

        Example:
            with model.as_frozen():  # by default, partial = True
                # Do something with the model
                pass

            # Model's parameters are now back to original state of requires_grad
        """
        training_mode = self.training
        self.freeze()
        try:
            yield
        finally:
            self.unfreeze(partial=True)

            if training_mode:
                self.train()
            else:
                self.eval()
