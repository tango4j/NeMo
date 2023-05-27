# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import torch

from nemo.utils import logging

try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False

try:
    from torchaudio.prototype.pipelines import SQUIM_OBJECTIVE

    HAVE_SQUIM = True
except ModuleNotFoundError:
    HAVE_SQUIM = False


class ParametricMultichannelWienerFilter(torch.nn.Module):
    """Parametric multichannel Wiener filter, with an adjustable
    tradeoff between noise reduction and speech distortion.
    It supports automatic reference channel selection based
    on the estimated output SNR.

    Args:
        beta: Parameter of the parameteric filter, tradeoff between noise reduction
              and speech distortion (0: MVDR, 1: MWF).
        rank: Rank assumption for the speech covariance matrix.
        postfilter: Optional postfilter. If None, no postfilter is applied.
        ref_channel: Optional, reference channel. If None, it will be estimated automatically.
        ref_hard: If true, estimate a hard (one-hot) reference. If false, a soft reference.
        ref_hard_use_grad: If true, use straight-through gradient when using the hard reference
        ref_subband_weighting: If true, use subband weighting when estimating reference channel
        num_subbands: Optional, used to determine the parameter size for reference estimation
        output_channel_reduction: Optional, may be used to apply channel averaging after the filter
        diag_reg: Optional, diagonal regularization for the multichannel filter
        eps: Small regularization constant to avoid division by zero
        dtype: Data type for computations

    References:
        [1] Souden et al., On Optimal Frequency-Domain Multichannel Linear Filtering for Noise Reduction, 2010
    """

    def __init__(
        self,
        beta: float = 1.0,
        rank: str = 'one',
        postfilter: Optional[str] = None,
        ref_channel: Optional[int] = None,
        ref_hard: bool = True,
        ref_hard_use_grad: bool = True,
        ref_subband_weighting: bool = False,
        num_subbands: Optional[int] = None,
        output_channel_reduction: Optional[str] = None,
        diag_reg: Optional[float] = 1e-6,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.cdouble,
    ):
        if not HAVE_TORCHAUDIO:
            logging.error('Could not import torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary to instantiate a {self.__class__.__name__}"
            )

        super().__init__()

        # Parametric filter
        # 0=MVDR, 1=MWF
        self.beta = beta

        # Rank
        # Assumed rank for the signal covariance matrix (psd_s)
        self.rank = rank

        if self.rank == 'full' and self.beta == 0:
            raise ValueError(f'Rank {self.rank} is not compatible with beta {self.beta}.')

        # Postfilter, applied on the output of the multichannel filter
        if postfilter not in [None, 'ban']:
            raise ValueError(f'Postfilter {postfilter} is not supported.')
        self.postfilter = postfilter

        # Regularization
        if diag_reg is not None and diag_reg < 0:
            raise ValueError(f'Diagonal regularization {diag_reg} must be positive.')
        self.diag_reg = diag_reg

        if eps <= 0:
            raise ValueError(f'Epsilon {eps} must be positive.')
        self.eps = eps

        # PSD estimator
        self.psd = torchaudio.transforms.PSD()

        # Reference channel
        self.ref_channel = ref_channel
        if self.ref_channel == 'max_snr':
            self.ref_estimator = ReferenceChannelEstimatorSNR(
                hard=ref_hard,
                hard_use_grad=ref_hard_use_grad,
                subband_weighting=ref_subband_weighting,
                num_subbands=num_subbands,
                eps=eps,
            )
        else:
            self.ref_estimator = None
        # Flag to determine if the filter is MISO or MIMO
        self.is_mimo = self.ref_channel is None

        # Output channel reduction
        if output_channel_reduction not in [None, 'avg', 'avg_pow']:
            raise ValueError(f'Output channel reduction {output_channel_reduction} is not supported.')
        self.output_channel_reduction = output_channel_reduction

        using_ref_channel = self.ref_channel is not None or self.ref_estimator is not None
        if using_ref_channel and self.output_channel_reduction is not None:
            logging.warning(
                f'Output channel reduction {self.output_channel_reduction} is not required when using a reference channel'
            )

        self.dtype = dtype

        logging.debug('Initialized %s', self.__class__.__name__)
        logging.debug('\tbeta:                     %f', self.beta)
        logging.debug('\trank:                     %s', self.rank)
        logging.debug('\tpostfilter:               %s', self.postfilter)
        logging.debug('\tdiag_reg:                 %g', self.diag_reg)
        logging.debug('\teps:                      %g', self.eps)
        logging.debug('\tref_channel:              %s', self.ref_channel)
        logging.debug('\tis_mimo:                  %s', self.is_mimo)
        logging.debug('\toutput channel reduction: %s', self.output_channel_reduction)
        logging.debug('\tdtype:                    %s', self.dtype)

    @staticmethod
    def trace(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Calculate trace of matrix slices over the last
        two dimensions in the input tensor.

        Args:
            x: tensor, shape (..., C, C)

        Returns:
            Trace for each (C, C) matrix. shape (...)
        """
        trace = torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)
        if keepdim:
            trace = trace.unsqueeze(-1).unsqueeze(-1)
        return trace

    def apply_diag_reg(self, psd: torch.Tensor) -> torch.Tensor:
        """Apply diagonal regularization on psd.

        Args:
            psd: tensor, shape (..., C, C)

        Returns:
            Tensor, same shape as input.
        """
        # Regularization: diag_reg * trace(psd) + eps
        diag_reg = self.diag_reg * self.trace(psd).real + self.eps

        # Apply regularization
        psd = psd + torch.diag_embed(diag_reg.unsqueeze(-1) * torch.ones(psd.shape[-1], device=psd.device))

        return psd

    def apply_filter(self, input: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
        """Apply the MIMO filter on the input.

        Args:
            input: batch with C input channels, shape (B, C, F, T)
            filter: batch of C-input, M-output filters, shape (B, F, C, M)
        
        Returns:
            M-channel filter output, shape (B, M, F, T)
        """
        if not filter.is_complex():
            raise TypeError(f'Expecting complex-valued filter, found {filter.dtype}')

        if not input.is_complex():
            raise TypeError(f'Expecting complex-valued input, found {input.dtype}')

        if filter.ndim != 4 or filter.size(-2) != input.size(-3) or filter.size(-3) != input.size(-2):
            raise ValueError(f'Filter shape {filter.shape}, not compatible with input shape {input.shape}')

        output = torch.einsum('bfcm,bcft->bmft', filter.conj(), input)

        return output

    def apply_ban(self, input: torch.Tensor, filter: torch.Tensor, psd_n: torch.Tensor) -> torch.Tensor:
        """Apply blind analytic normalization postfilter. Note that this normalization has been
        derived for the GEV beamformer in [1]. More specifically, the BAN postfilter aims to scale GEV
        to satisfy the distortionless constraint and the final analytical expression is derived using
        an assumption on the norm of the transfer function.
        However, this may still be useful in some instances.

        Args:
            input: batch with M output channels (B, M, F, T)
            filter: batch of C-input, M-output filters, shape (B, F, C, M)
            psd_n: batch of noise PSDs, shape (B, F, C, C)
        
        Returns:
            Filtered input, shape (B, M, F, T)

        References:
            [1] Warsitz and Haeb-Umbach, Blind Acoustic Beamforming Based on Generalized Eigenvalue Decomposition, 2007
        """
        # number of input channel, used to normalize the numerator
        num_inputs = filter.size(-2)
        numerator = torch.einsum('bfcm,bfci,bfij,bfjm->bmf', filter.conj(), psd_n, psd_n, filter)
        numerator = torch.sqrt(numerator.abs() / num_inputs)

        denominator = torch.einsum('bfcm,bfci,bfim->bmf', filter.conj(), psd_n, filter)
        denominator = denominator.abs()

        # Scalar filter per output channel, frequency and batch
        # shape (B, M, F)
        ban = numerator / (denominator + self.eps)

        input = ban[..., None] * input

        return input

    def forward(self, input: torch.Tensor, mask_s: torch.Tensor, mask_n: torch.Tensor) -> torch.Tensor:
        """Return processed signal.
        The output has either one channel (M=1) if a ref_channel is selected,
        or the same number of channels as the input (M=C) if ref_channel is None.

        Args:
            input: Input signal, complex tensor with shape (B, C, F, T)
            mask_s: Mask for the desired signal, shape (B, F, T)
            mask_n: Mask for the undesired noise, shape (B, F, T)

        Returns:
            Processed signal, shape (B, M, F, T)
        """
        iodtype = input.dtype

        with torch.cuda.amp.autocast(enabled=False):
            input = input.to(dtype=self.dtype)
            if self.dtype == torch.cdouble:
                # Convert to double
                mask_s = mask_s.double()
                mask_n = mask_n.double()
            elif self.dtype == torch.cfloat:
                # Convert to float
                mask_s = mask_s.float()
                mask_n = mask_n.float()
            else:
                raise ValueError(f'Unsupported dtype {self.dtype}')

            # Calculate signal statistics
            psd_s = self.psd(input, mask_s)
            psd_n = self.psd(input, mask_n)

            if self.rank == 'one':
                # Calculate filter W using (18) in [1]
                # Diagonal regularization
                if self.diag_reg:
                    psd_n = self.apply_diag_reg(psd_n)

                # MIMO filter
                # (B, F, C, C)
                W = torch.linalg.solve(psd_n, psd_s)
                lam = self.trace(W, keepdim=True).real
                W = W / (self.beta + lam + self.eps)
            elif self.rank == 'full':
                # Calculate filter W using (15) in [1]
                psd_sn = psd_s + self.beta * psd_n

                if self.diag_reg:
                    psd_sn = self.apply_diag_reg(psd_sn)

                # MIMO filter
                # (B, F, C, C)
                W = torch.linalg.solve(psd_sn, psd_s)
            else:
                raise RuntimeError(f'Unexpected rank {self.rank}')

            if torch.jit.isinstance(self.ref_channel, int):
                # Fixed ref channel
                # (B, F, C, 1)
                W = W[..., self.ref_channel].unsqueeze(-1)

            elif self.ref_estimator is not None:
                # Estimate the ref channel tensor (one-hot or soft across C)
                # (B, C)
                ref_channel_tensor = self.ref_estimator(W=W, psd_s=psd_s, psd_n=psd_n).to(W.dtype)
                # Weighting across channels
                # (B, F, C, 1)
                W = torch.sum(W * ref_channel_tensor[:, None, None, :], dim=-1, keepdim=True)

            # Calculate the output with the selected filter
            output = self.apply_filter(input=input, filter=W)

            # Optional: postfilter
            if self.postfilter == 'ban':
                output = self.apply_ban(input=output, filter=W, psd_n=psd_n)

            if self.output_channel_reduction == 'avg':
                if self.ref_channel is not None or self.ref_estimator is not None:
                    logging.warning(
                        f'Output channel reduction {self.output_channel_reduction} is not required when using a reference channel'
                    )
                output = torch.mean(output, axis=1, keepdim=True)
            elif self.output_channel_reduction == 'avg_pow':
                if self.ref_channel is not None or self.ref_estimator is not None:
                    logging.warning(
                        f'Output channel reduction {self.output_channel_reduction} is not required when using a reference channel'
                    )
                # Phase of the first channel
                output_angle = torch.angle(output[:, 0, ...])
                # Average power
                output_pow = torch.mean(torch.abs(output) ** 2, axis=1)
                # Convert to complex and unsqueeze to add channel dim
                output = torch.polar(torch.sqrt(output_pow), output_angle).unsqueeze(1)

        return output.to(iodtype)


class ReferenceChannelEstimatorSNR(torch.nn.Module):
    """Estimate a reference channel by selecting the reference
    that maximizes the output SNR. It returns one-hot encoded
    vector or a soft reference.

    A straight-through estimator is used for gradient when using
    hard reference.

    Args:
        hard: If true, use hard estimate of ref channel.
            If false, use a soft estimate across channels.
        hard_use_grad: Use straight-through estimator for
            the gradient.
        subband_weighting: If true, use subband weighting when
            adding across subband SNRs. If false, use average
            across subbands.

    References:
        Boeddeker et al., Front-End Processing for the CHiME-5 Dinner Party Scenario, 2018
    """

    def __init__(
        self,
        hard: bool = True,
        hard_use_grad: bool = True,
        subband_weighting: bool = False,
        num_subbands: Optional[int] = None,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.hard = hard
        self.hard_use_grad = hard_use_grad
        self.subband_weighting = subband_weighting
        self.eps = eps

        if subband_weighting and num_subbands is None:
            raise ValueError(f'Number of subbands must be provided when using subband_weighting={subband_weighting}.')
        # Subband weighting
        self.weight_s = torch.nn.Parameter(torch.ones(num_subbands)) if subband_weighting else None
        self.weight_n = torch.nn.Parameter(torch.ones(num_subbands)) if subband_weighting else None

        logging.debug('Initialized %s', self.__class__.__name__)
        logging.debug('\thard:              %d', self.hard)
        logging.debug('\thard_use_grad:     %d', self.hard_use_grad)
        logging.debug('\tsubband_weighting: %d', self.subband_weighting)
        logging.debug('\tnum_subbands:      %s', num_subbands)
        logging.debug('\teps:               %e', self.eps)

    def forward(self, W: torch.Tensor, psd_s: torch.Tensor, psd_n: torch.Tensor) -> torch.Tensor:
        """
        Args:
            W: Multichannel input multichannel output filter, shape (B, F, C, C)
            psd_s: Covariance for the signal, shape (B, F, C, C)
            psd_n: Covariance for the noise, shape (B, F, C, C)

        Returns:
            One-hot or soft reference channel, shape (B, C)
        """
        if self.subband_weighting:
            # (B, F, M)
            pow_s = torch.einsum('...jm,...jk,...km->...m', W.conj(), psd_s, W).abs()
            pow_n = torch.einsum('...jm,...jk,...km->...m', W.conj(), psd_n, W).abs()

            # Subband-weighting
            # (B, F, M) -> (B, M)
            pow_s = torch.sum(pow_s * self.weight_s.softmax(dim=0).unsqueeze(1), dim=-2)
            pow_n = torch.sum(pow_n * self.weight_n.softmax(dim=0).unsqueeze(1), dim=-2)
        else:
            # Sum across f as well
            # (B, F, C, M), (B, F, C, C), (B, F, C, M) -> (B, M)
            pow_s = torch.einsum('...fjm,...fjk,...fkm->...m', W.conj(), psd_s, W).abs()
            pow_n = torch.einsum('...fjm,...fjk,...fkm->...m', W.conj(), psd_n, W).abs()

        # Estimated SNR per channel (B, C)
        snr = pow_s / (pow_n + self.eps)
        snr = 10 * torch.log10(snr + self.eps)

        # Soft reference
        ref_soft = snr.softmax(dim=-1)

        if self.hard:
            _, idx = ref_soft.max(dim=-1, keepdim=True)
            ref_hard = torch.zeros_like(snr).scatter(-1, idx, 1.0)
            if self.hard_use_grad:
                # Straight-through for gradient
                # Propagate ref_soft gradient, as if thresholding is identity
                ref = ref_hard - ref_soft.detach() + ref_soft
            else:
                # No gradient
                ref = ref_hard
        else:
            ref = ref_soft

        return ref


class ReferenceChannelEstimatorSQUIM(torch.nn.Module):
    """Estimate a reference channel by selecting the reference
    that maximizes a SQUIM metric.

    A straight-through estimator is used for gradient when using
    hard reference.

    Args:
        metric: squim objective metric
        hard: If true, use hard estimate of ref channel.
            If false, use a soft estimate across channels.
        hard_use_grad: Use straight-through estimator for
            the gradient.

    References:
        TorchAudio-SQUIM
    """

    def __init__(
        self,
        metric: str,
        hard: bool = True,
        hard_use_grad: bool = True,
        channel_step: int = 8,
    ):
        if not HAVE_SQUIM:
            logging.error('Could not import SQUIM from torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                "torchaudio with SQUIM is not installed but is necessary to instantiate a {self.__class__.__name__}"
            )

        super().__init__()

        self.metric = metric
        self.hard = hard
        self.hard_use_grad = hard_use_grad

        if self.metric in ['stoi', 'pesq', 'si-sdr']:
            self.model = SQUIM_OBJECTIVE.get_model()
        else:
            raise ValueError(f'Unknown metric {self.metric}')

        self.channel_step = channel_step

        logging.debug('Initialized %s', self.__class__.__name__)
        logging.debug('\tmetric:            %d', self.metric)
        logging.debug('\thard:              %d', self.hard)
        logging.debug('\thard_use_grad:     %d', self.hard_use_grad)
        logging.debug('\tchannel_step:      %d', self.channel_step)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Multichannel signal, shape (B, C, T)

        Returns:
            One-hot or soft reference channel, shape (B, C)
        """
        assert input.ndim == 3, f'Expected 3D tensor, got {input.ndim}D'
        B, C, T = input.shape

        score = torch.zeros(B, C, device=input.device)

        # Process a few channels at a time
        # Processing all C channels at once caused OOM on GPU
        for b in range(B):
            for c in range(0, C, self.channel_step):
                # Calculate score for each channel
                model_output = self.model(input[b, c:c+self.channel_step, :])

                # Select the desired metric
                if self.metric == 'stoi':
                    metric_val = model_output[0]
                elif self.metric == 'pesq':
                    metric_val = model_output[1]
                elif self.metric == 'si-sdr':
                    metric_val = model_output[2]
                else:
                    raise ValueError(f'Unknown metric: {self.metric}')

                # Save the score
                score[b, c:c+self.channel_step] = metric_val

        # Soft reference across channels
        ref_soft = score.softmax(dim=-1)

        if self.hard:
            _, idx = ref_soft.max(dim=-1, keepdim=True)
            ref_hard = torch.zeros_like(score).scatter(-1, idx, 1.0)
            if self.hard_use_grad:
                # Straight-through for gradient
                # Propagate ref_soft gradient, as if thresholding is identity
                ref = ref_hard - ref_soft.detach() + ref_soft
            else:
                # No gradient
                ref = ref_hard
        else:
            ref = ref_soft

        return ref


def linsolve_cholesky(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return linsolve(A, B) using Cholesky decomposition.
    The assumption is that A is symmetric and positive definite.
    """
    L = torch.linalg.cholesky(A)
    X = torch.linalg.solve_triangular(L, B, upper=False)
    X = torch.linalg.solve_triangular(L.conj().transpose(-2, -1), X, upper=True)

    return X


def estimate_steering_vector(psd_s: torch.Tensor, psd_n: torch.Tensor) -> torch.Tensor:
    """Estimate the steering vector using noise covariance whitening [1].

    Args:
        psd_s:
        psd_n:

    Returns:
        steering_vector: shape (B, F, C)

    References:
        [1] S. Markovich-Golan, Performance analysis of the covariance subtraction method for relative transfer function estimation and comparison to the covariance whitening method, 2015
    """
    # whitening
    L = torch.linalg.cholesky(psd_n)
    inv_L = L.inverse()

    whitened = torch.matmul(inv_L, torch.matmul(psd_s, inv_L.conj().transpose(-2, -1)))
    eigvals, eigvecs = torch.linalg.eigh(whitened)

    # Transform back from the whitened domain
    steering_vector = torch.matmul(L, eigvecs[..., [-1]]).squeeze(-1)

    return steering_vector


class WeightedMinimumPowerDistortionlessResponseFilter(torch.nn.Module):
    """Weighted MPDR filter.
    """

    def __init__(
        self,
        num_iterations=5,
        postfilter: Optional[str] = None,
        ref_channel: Optional[int] = None,
        ref_hard: bool = True,
        ref_hard_use_grad: bool = True,
        ref_subband_weighting: bool = False,
        num_subbands: Optional[int] = None,
        output_channel_reduction: Optional[str] = None,
        diag_reg: Optional[float] = 1e-6,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.cdouble,
    ):

        if not HAVE_TORCHAUDIO:
            logging.error('Could not import torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary to instantiate a {self.__class__.__name__}"
            )

        super().__init__()

        # Number of iterations for the filter
        self.num_iterations = num_iterations

        # Postfilter, applied on the output of the multichannel filter
        if postfilter not in [None, 'ban']:
            raise ValueError(f'Postfilter {postfilter} is not supported.')
        self.postfilter = postfilter

        # Regularization
        if diag_reg is not None and diag_reg < 0:
            raise ValueError(f'Diagonal regularization {diag_reg} must be positive.')
        self.diag_reg = diag_reg

        if eps <= 0:
            raise ValueError(f'Epsilon {eps} must be positive.')
        self.eps = eps

        # Reference channel
        self.ref_channel = ref_channel
        if self.ref_channel == 'max_snr':
            self.ref_estimator = ReferenceChannelEstimatorSNR(
                hard=ref_hard,
                hard_use_grad=ref_hard_use_grad,
                subband_weighting=ref_subband_weighting,
                num_subbands=num_subbands,
                eps=eps,
            )
        else:
            self.ref_estimator = None
        # Flag to determine if the filter is MISO or MIMO
        self.is_mimo = self.ref_channel is None

        # Output channel reduction
        if output_channel_reduction not in [None, 'avg', 'avg_pow']:
            raise ValueError(f'Output channel reduction {output_channel_reduction} is not supported.')
        self.output_channel_reduction = output_channel_reduction

        using_ref_channel = self.ref_channel is not None or self.ref_estimator is not None
        if using_ref_channel and self.output_channel_reduction is not None:
            logging.warning(
                f'Output channel reduction {self.output_channel_reduction} is not required when using a reference channel'
            )

        # Internal calculations
        assert dtype in [torch.cfloat, torch.cdouble], f'Unsupported dtype {dtype}, expecting cfloat or cdouble'
        self.dtype = dtype

        logging.debug('Initialized %s', self.__class__.__name__)
        logging.debug('\tnum_iterations:           %s', self.num_iterations)
        logging.debug('\tpostfilter:               %s', self.postfilter)
        logging.debug('\tdiag_reg:                 %g', self.diag_reg)
        logging.debug('\teps:                      %g', self.eps)
        logging.debug('\tref_channel:              %s', self.ref_channel)
        logging.debug('\tis_mimo:                  %s', self.is_mimo)
        logging.debug('\toutput channel reduction: %s', self.output_channel_reduction)
        logging.debug('\tdtype:                    %s', self.dtype)

    @staticmethod
    def trace(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Calculate trace of matrix slices over the last
        two dimensions in the input tensor.

        Args:
            x: tensor, shape (..., C, C)

        Returns:
            Trace for each (C, C) matrix. shape (...)
        """
        trace = torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)
        if keepdim:
            trace = trace.unsqueeze(-1).unsqueeze(-1)
        return trace

    def apply_diag_reg(self, psd: torch.Tensor) -> torch.Tensor:
        """Apply diagonal regularization on psd.

        Args:
            psd: tensor, shape (..., C, C)

        Returns:
            Tensor, same shape as input.
        """
        # Regularization: diag_reg * trace(psd) + eps
        diag_reg = self.diag_reg * self.trace(psd).real + self.eps

        # Apply regularization
        psd = psd + torch.diag_embed(diag_reg.unsqueeze(-1) * torch.ones(psd.shape[-1], device=psd.device))

        return psd

    def apply_filter(self, input: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
        """Apply the MIMO filter on the input.

        Args:
            input: batch with C input channels, shape (B, C, F, T)
            filter: batch of C-input, M-output filters, shape (B, F, C, M)
        
        Returns:
            M-channel filter output, shape (B, M, F, T)
        """
        if not filter.is_complex():
            raise TypeError(f'Expecting complex-valued filter, found {filter.dtype}')

        if not input.is_complex():
            raise TypeError(f'Expecting complex-valued input, found {input.dtype}')

        if filter.ndim != 4 or filter.size(-2) != input.size(-3) or filter.size(-3) != input.size(-2):
            raise ValueError(f'Filter shape {filter.shape}, not compatible with input shape {input.shape}')

        output = torch.einsum('bfcm,bcft->bmft', filter.conj(), input)

        return output

    def apply_ban(self, input: torch.Tensor, filter: torch.Tensor, psd_n: torch.Tensor) -> torch.Tensor:
        """Apply blind analytic normalization postfilter. Note that this normalization has been
        derived for the GEV beamformer in [1]. More specifically, the BAN postfilter aims to scale GEV
        to satisfy the distortionless constraint and the final analytical expression is derived using
        an assumption on the norm of the transfer function.
        However, this may still be useful in some instances.

        Args:
            input: batch with M output channels (B, M, F, T)
            filter: batch of C-input, M-output filters, shape (B, F, C, M)
            psd_n: batch of noise PSDs, shape (B, F, C, C)
        
        Returns:
            Filtered input, shape (B, M, F, T)

        References:
            [1] Warsitz and Haeb-Umbach, Blind Acoustic Beamforming Based on Generalized Eigenvalue Decomposition, 2007
        """
        # number of input channel, used to normalize the numerator
        num_inputs = filter.size(-2)
        numerator = torch.einsum('bfcm,bfci,bfij,bfjm->bmf', filter.conj(), psd_n, psd_n, filter)
        numerator = torch.sqrt(numerator.abs() / num_inputs)

        denominator = torch.einsum('bfcm,bfci,bfim->bmf', filter.conj(), psd_n, filter)
        denominator = denominator.abs()

        # Scalar filter per output channel, frequency and batch
        # shape (B, M, F)
        ban = numerator / (denominator + self.eps)

        input = ban[..., None] * input

        return input

    def forward(self, input: torch.Tensor, mask_s: torch.Tensor, mask_n: torch.Tensor) -> torch.Tensor:
        """Return processed signal
        """
        iodtype = input.dtype

        with torch.cuda.amp.autocast(enabled=False):
            input = input.to(dtype=self.dtype)
            if self.dtype == torch.cdouble:
                # Convert to double
                mask_s = mask_s.double()
                mask_n = mask_n.double()
            elif self.dtype == torch.cfloat:
                # Convert to float
                mask_s = mask_s.float()
                mask_n = mask_n.float()
            else:
                raise ValueError(f'Unsupported dtype {self.dtype}')

            # Calculate signal statistics
            psd_s = torchaudio.functional.psd(input, mask_s)
            psd_n = torchaudio.functional.psd(input, mask_n)

            if self.diag_reg:
                psd_n = self.apply_diag_reg(psd_n)

            # Estimate the steering vector
            steering_vector = estimate_steering_vector(psd_s, psd_n)

            # Number of time steps for weighted PSD normalization
            T = input.size(-1)

            for n in range(self.num_iterations):
                # Calculate the power by averaging over channels
                if n == 0:
                    power = torch.mean(input.abs().pow(2), axis=-3)
                else:
                    power = torch.mean(output.abs().pow(2), axis=-3)

                # weighted PSD with 1/power
                power_weight = 1 / (power + self.eps)
                psd_w = torchaudio.functional.psd(input, power_weight, normalize=False) / T

                # Apply diagonal regularization
                if self.diag_reg:
                    psd_w = self.apply_diag_reg(psd_w)

                # Estimate the filter
                W = linsolve_cholesky(psd_w, steering_vector[..., None])
                W = W / (torch.matmul(steering_vector[..., None].conj().transpose(-2, -1), W).abs() + self.eps)

                # Apply reference channel scaling for all input channels
                ref_scale = steering_vector.abs().pow(2) / steering_vector
                W = W * ref_scale[..., None, :]

                if torch.jit.isinstance(self.ref_channel, int):
                    # Fixed ref channel
                    # (B, F, C, 1)
                    W = W[..., self.ref_channel].unsqueeze(-1)
                elif self.ref_estimator is not None:
                    # Estimate ref channel tensor (one-hot or soft across C)
                    # (B, C)
                    ref_channel_tensor = self.ref_estimator(W=W, psd_s=psd_s, psd_n=psd_n).to(W.dtype)
                    # Weighting across channels
                    # (B, F, C, 1)
                    W = torch.sum(W * ref_channel_tensor[:, None, None, :], dim=-1, keepdim=True)

                output = self.apply_filter(input=input, filter=W)

            # Optional: postfilter
            if self.postfilter == 'ban':
                output = self.apply_ban(input=output, filter=W, psd_n=psd_n)

            if self.output_channel_reduction == 'avg':
                if self.ref_channel is not None or self.ref_estimator is not None:
                    logging.warning(
                        f'Output channel reduction {self.output_channel_reduction} is not required when using a reference channel'
                    )
                output = torch.mean(output, axis=1, keepdim=True)
            elif self.output_channel_reduction == 'avg_pow':
                if self.ref_channel is not None or self.ref_estimator is not None:
                    logging.warning(
                        f'Output channel reduction {self.output_channel_reduction} is not required when using a reference channel'
                    )
                # Phase of the first channel
                output_angle = torch.angle(output[:, 0, ...])
                # Average power
                output_pow = torch.mean(torch.abs(output) ** 2, axis=1)
                # Convert to complex and unsqueeze to add channel dim
                output = torch.polar(torch.sqrt(output_pow), output_angle).unsqueeze(1)

        return output.to(iodtype)
