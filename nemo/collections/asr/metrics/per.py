# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List

import editdistance
import torch
from torchmetrics import Metric

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging


class PER(Metric):
    """
    This metric computes the numerator and denominator for the overall Phoneme Error Rate (PER) between
    prediction and reference texts.
    It is very similar to the computation of Word Error Rate (WER); you can think of each phoneme as a word.
    When doing distributed training/evaluation the result of res=PER(predictions, targets, target_lengths) calls
    will be all-reduced between all workers using SUM operations.
    The output is two numbers res=[per_numerator, per_denominator]. PER=per_numerator/per_denominator.

    If used with PytorchLightning LightningModule, include per_numerator and per_denominators inside validation_step results.
    Then aggregate (sum) then at the end of validation epoch to correctly compute validation PER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            per_num, per_denom = self.__per(predictions, transcript, transcript_len)
            return {'val_loss': loss_value, 'val_per_num': per_num, 'val_per_denom': per_denom}

        def validation_epoch_end(self, outputs):
            ...
            per_num = torch.stack([x['val_per_num'] for x in outputs]).sum()
            per_denom = torch.stack([x['val_per_denom'] for x in outputs]).sum()
            tensorboard_logs = {'validation_loss': val_loss_mean, 'validation_avg_per': per_num / per_denom}
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        tokenizer: NeMo tokenizer object.
        batch_dim_index: Index of the batch dimension.
        ctc_decode: Whether to use CTC decoding or not. Currently, must be set.
        log_prediction: Whether to log a single decoded sample per call.

    Returns:
        res: a torch.Tensor object with two elements: [per_numerator, per_denominator]. To correctly compute the
            average text phoneme error rate, compute per=per_numerator/per_denominator.
    """

    def __init__(
        self, tokenizer, batch_dim_index=0, ctc_decode=True, log_prediction=True, dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.tokenizer = tokenizer
        self.batch_dim_index = batch_dim_index
        self.blank_id = tokenizer.vocab_size
        self.ctc_decode = ctc_decode
        self.log_prediction = log_prediction

        self.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("phonemes", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def ctc_decoder_predictions_tensor(
        self, predictions: torch.Tensor, predictions_len: torch.Tensor = None, return_hypotheses: bool = False,
    ) -> List[str]:
        """
        Decodes a sequence of labels to phonemes. This is functionally the same as in the WER class.

        Args:
            predictions: A torch.Tensor of shape [Batch, Time] of integer indices that correspond
                to the index of some phoneme in the label set.
            predictions_len: Optional tensor of length `Batch` which contains the integer lengths
                of the sequence in the padded `predictions` tensor.
            return_hypotheses: Bool flag whether to return just the decoding predictions of the model
                or a Hypothesis object that holds information such as the decoded `text`,
                the `alignment` of emited by the CTC Model, and the `length` of the sequence (if available).
                May also contain the log-probabilities of the decoder (if this method is called via
                transcribe())

        Returns:
            Either a list of str which represent the CTC decoded strings per sample,
            or a list of Hypothesis objects containing additional information.
        """
        hypotheses = []
        # Drop predictions to CPU
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]
            # CTC decoding procedure
            decoded_prediction = []
            previous = self.blank_id
            for p in prediction:
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                previous = p

            text = self.decode_tokens_to_str(decoded_prediction)

            if not return_hypotheses:
                hypothesis = text
            else:
                hypothesis = Hypothesis(
                    y_sequence=None,
                    score=-1.0,
                    text=text,
                    alignments=prediction,
                    length=predictions_len[ind] if predictions_len is not None else 0,
                )

            hypotheses.append(hypothesis)
        return hypotheses

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented in order to decode a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = ' '.join(self.decode_ids_to_tokens(tokens))
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        token_list = [self.tokenizer.inv_vocab[t] for t in tokens if t != self.blank_id]
        return token_list

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictions_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        phonemes = 0.0
        scores = 0.0
        references = []
        with torch.no_grad():
            # prediction_cpu_tensor = tensors[0].long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()

            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[self.batch_dim_index]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decode_tokens_to_str(target)
                references.append(reference)
            if self.ctc_decode:
                hypotheses = self.ctc_decoder_predictions_tensor(predictions, predictions_lengths)
            else:
                raise NotImplementedError("Implement me if you need non-CTC decode on predictions")

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference:{references[0]}")
            logging.info(f"predicted:{hypotheses[0]}")

        for h, r in zip(hypotheses, references):
            h_list = h.split()
            r_list = r.split()
            phonemes += len(r_list)
            # Compute Levenshtein distance
            scores += editdistance.eval(h_list, r_list)

        self.scores = torch.tensor(scores, device=self.scores.device, dtype=self.scores.dtype)
        self.phonemes = torch.tensor(phonemes, device=self.phonemes.device, dtype=self.phonemes.dtype)

    def compute(self):
        scores = self.scores.detach().float()
        phonemes = self.phonemes.detach().float()
        return scores / phonemes, scores, phonemes
