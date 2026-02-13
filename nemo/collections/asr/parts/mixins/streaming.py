# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod


class StreamingEncoder(ABC):
    @abstractmethod
    def setup_streaming_params(
        self,
        max_look_ahead: int = 10000,
    ):
        """
        This function sets the needed values and parameters to perform streaming. The configuration (CacheAwareStreamingConfig) need to be stored in self.streaming_cfg.
        The streaming configuration is needed to simulate streaming inference. It would set the following
        """
        pass

    @abstractmethod
    def get_initial_cache_state(self, batch_size, dtype, device, max_dim):
        pass

    @staticmethod
    def to_numpy(tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def cache_aware_stream_step(
        self,
        processed_signal,
        processed_signal_length=None,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        keep_all_outputs=True,
        drop_extra_pre_encoded=None,
        bypass_pre_encode=False,
    ):
        if self.streaming_cfg is None:
            self.setup_streaming_params()
        if drop_extra_pre_encoded is not None:
            prev_drop_extra_pre_encoded = self.streaming_cfg.drop_extra_pre_encoded
            self.streaming_cfg.drop_extra_pre_encoded = drop_extra_pre_encoded
        else:
            prev_drop_extra_pre_encoded = None

        if processed_signal_length is None:
            processed_signal_length = processed_signal.new_full(processed_signal.size(0), processed_signal.size(-1))

        encoder_output = self(
            audio_signal=processed_signal,
            length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
        )

        encoder_output = self.streaming_post_process(encoder_output, keep_all_outputs=keep_all_outputs)

        if prev_drop_extra_pre_encoded is not None:
            self.streaming_cfg.drop_extra_pre_encoded = prev_drop_extra_pre_encoded

        return encoder_output

    def cache_aware_stream_step_with_diarization(
        self,
        processed_signal,
        processed_signal_length=None,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        keep_all_outputs=True,
        drop_extra_pre_encoded=None,
        bypass_pre_encode=False,
        diar_preds=None,
        diar_fusion_fn=None,
    ):
        """
        Cache-aware streaming step with optional diarization fusion.

        Runs the standard encoder streaming step via cache_aware_stream_step(),
        then applies diarization fusion on the encoded output if diar_preds and
        diar_fusion_fn are both provided.

        The diar_fusion_fn is expected to be provided by the model class
        (e.g. MSEncDecRNNTBPEModel.fuse_diar_preds) that owns the learned
        fusion parameters (projection layers, normalization, etc.).

        Args:
            processed_signal: input mel-spectrogram features (B, D, T)
            processed_signal_length: lengths (B,)
            cache_last_channel: cache tensor for last channel layers (MHA)
            cache_last_time: cache tensor for last time layers (convolutions)
            cache_last_channel_len: lengths for cache_last_channel
            keep_all_outputs: if True, keep all outputs including right-context
            drop_extra_pre_encoded: steps to drop after downsampling
            bypass_pre_encode: if True, skip pre-encode (already pre-encoded)
            diar_preds: (B, T_diar, num_speakers) frame-level speaker predictions
            diar_fusion_fn: callable(encoded, diar_preds) -> fused_encoded
                A function that fuses diarization predictions with encoder states.
                Typically MSEncDecRNNTBPEModel.fuse_diar_preds.

        Returns:
            Same 5-tuple as cache_aware_stream_step:
            (encoded, encoded_len, cache_last_channel_next,
             cache_last_time_next, cache_last_channel_next_len)
            where encoded has diar fusion applied if diar_preds was provided.
        """
        (
            encoded,
            encoded_len,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_next_len,
        ) = self.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            bypass_pre_encode=bypass_pre_encode,
        )

        # Apply diarization fusion on the encoder output
        if diar_preds is not None and diar_fusion_fn is not None:
            encoded = diar_fusion_fn(encoded, diar_preds)

        return (
            encoded,
            encoded_len,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_next_len,
        )
