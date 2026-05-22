#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Load a ParallelExpertEncoder bundle (.nemo) and run a small forward smoke test."""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nemo",
        type=str,
        default="/disk_a_nvd/models/rich_transcription/canary-enc-1b-v2-sortformer-v2.1.nemo",
        help="Path to ParallelExpertEncoderPT .nemo bundle.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device string, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--time-frames",
        type=int,
        default=256,
        help="Number of mel frames T (before subsampling inside the encoder).",
    )
    args = parser.parse_args()

    from nemo.collections.asr.modules.parallel_expert_encoder import (
        load_parallel_expert_encoder_from_nemo,
    )

    print(f"Loading bundle from {args.nemo!r} ...", flush=True)
    enc = load_parallel_expert_encoder_from_nemo(args.nemo, map_location="cpu")
    dev = torch.device(args.device)
    enc = enc.to(dev)
    enc.eval()

    b, t = args.batch, args.time_frames
    feat = enc._feat_in
    audio = torch.randn(b, feat, t, device=dev, dtype=torch.float32)
    length = torch.full((b,), t, device=dev, dtype=torch.int64)

    print(
        f"Smoke forward: batch={b} feat={feat} T={t} device={dev} "
        f"n_spk={enc.n_spk} d_model={enc.d_model} subsampling={enc.subsampling_factor}",
        flush=True,
    )
    with torch.no_grad():
        out, out_len = enc(audio_signal=audio, length=length)

    print(f"outputs shape: {tuple(out.shape)} encoded_lengths: {out_len.tolist()}", flush=True)
    assert out.shape[0] == b and out.shape[1] == enc.d_model
    assert (out_len <= out.shape[2]).all()
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
