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
"""Speaker-token alias utilities for SOT-style multi-speaker SALM training.

This module provides pure-Python (stdlib only) helpers to translate user-facing
speaker tags (e.g. ``[s0]``, ``[s1]``, ...) into pre-existing reserved tokens
in the LLM tokenizer's vocabulary (e.g. ``<SPECIAL_100>``). Aliasing onto
already-present token ids avoids ``resize_token_embeddings`` and stays
shape-compatible with the base HF safetensors checkpoint and FSDP2/EP sharding.

The same helper is used in two places:

* In the dataloader workers (``cut_to_conversation`` in ``cutset.py``), to
  rewrite assistant target text *before* tokenization.
* In ``SALMAutomodel`` (training process), to register the underlying special
  tokens with the tokenizer and to expose ``speaker_token_ids`` for the
  auxiliary speaker loss (LSS).

Keeping this in pure Python (no torch / transformers / lhotse imports) means
worker processes can import it cheaply and it never causes circular imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


@dataclass
class SpeakerAliasTable:
    """A bidirectional alias table between user-facing speaker tags and
    underlying tokenizer-vocabulary strings.

    Attributes:
        aliases: Mapping from user-facing tag (e.g. ``"[s0]"``) to the
            underlying tokenizer string (e.g. ``"<SPECIAL_100>"``).
        underlying_strings: Same as ``list(aliases.values())``, kept ordered
            so that downstream code can map alias index -> underlying string
            without ambiguity.
        max_speakers: Number of speakers supported (``len(aliases)``).
    """

    aliases: Mapping[str, str]
    underlying_strings: Sequence[str]
    max_speakers: int
    _alias_re: re.Pattern = field(init=False, repr=False)
    _dealias_re: re.Pattern = field(init=False, repr=False)
    _inverse: Mapping[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Sort longest-first so that "[s10]" is matched before "[s1]".
        # Same precaution for the inverse direction (defensive: tokens are unique
        # but compiling the same way keeps behaviour symmetric).
        if not self.aliases:
            self._alias_re = re.compile(r"(?!x)x")  # never matches
            self._dealias_re = re.compile(r"(?!x)x")
            self._inverse = {}
            return
        alias_keys = sorted(self.aliases.keys(), key=len, reverse=True)
        self._alias_re = re.compile("|".join(re.escape(k) for k in alias_keys))
        self._inverse = {v: k for k, v in self.aliases.items()}
        underlying_keys = sorted(self._inverse.keys(), key=len, reverse=True)
        self._dealias_re = re.compile("|".join(re.escape(k) for k in underlying_keys))

    @property
    def is_active(self) -> bool:
        """True iff the alias table has at least one alias."""
        return self.max_speakers > 0

    def apply(self, text: Optional[str]) -> Optional[str]:
        """Replace user-facing alias tags with the underlying tokenizer strings.

        Safe no-op for ``None``, empty strings, and inactive tables.
        """
        if not text or not self.is_active:
            return text
        return self._alias_re.sub(lambda m: self.aliases[m.group(0)], text)

    def undo(self, text: Optional[str]) -> Optional[str]:
        """Inverse of :meth:`apply`; for displaying decoded predictions."""
        if not text or not self.is_active:
            return text
        return self._dealias_re.sub(lambda m: self._inverse[m.group(0)], text)


def build_alias_table(cfg: Optional[Mapping]) -> SpeakerAliasTable:
    """Build a :class:`SpeakerAliasTable` from a YAML-style config dict.

    Expected schema (all keys optional with defaults shown)::

        speaker_aliases:
          enable: true                  # set false to return an empty table
          prefix: "[s"                  # alias becomes f"{prefix}{i}{suffix}"
          suffix: "]"
          max_speakers: 8
          # Either provide an explicit list of underlying tokens ...
          tokens: ["<SPECIAL_100>", "<SPECIAL_101>", ...]
          # ... or an arithmetic recipe (used iff `tokens` is absent):
          base_special_id: 100          # id 100 -> "[s0]", 101 -> "[s1]", ...
          template: "<SPECIAL_{i}>"

    Returns an empty (inactive) table when ``cfg`` is ``None`` or
    ``cfg.enable`` is ``False`` so call sites can unconditionally call
    ``table.apply(text)`` without checking.
    """
    if cfg is None:
        return SpeakerAliasTable(aliases={}, underlying_strings=(), max_speakers=0)
    if isinstance(cfg, Mapping):
        get = cfg.get
    else:
        # OmegaConf DictConfig also supports .get; this branch handles plain objects.
        get = lambda k, default=None: getattr(cfg, k, default)
    if not bool(get("enable", True)):
        return SpeakerAliasTable(aliases={}, underlying_strings=(), max_speakers=0)
    prefix = get("prefix", "[s")
    suffix = get("suffix", "]")
    max_speakers = int(get("max_speakers", 8))
    tokens = get("tokens", None)
    if tokens is not None:
        underlying = list(tokens)
        if len(underlying) < max_speakers:
            raise ValueError(
                f"speaker_aliases.tokens has {len(underlying)} entries but "
                f"max_speakers={max_speakers}; provide at least max_speakers tokens."
            )
        underlying = underlying[:max_speakers]
    else:
        base_id = int(get("base_special_id", 100))
        template = get("template", "<SPECIAL_{i}>")
        underlying = [template.format(i=base_id + i) for i in range(max_speakers)]
    aliases = {f"{prefix}{i}{suffix}": underlying[i] for i in range(max_speakers)}
    return SpeakerAliasTable(
        aliases=aliases, underlying_strings=tuple(underlying), max_speakers=max_speakers
    )
