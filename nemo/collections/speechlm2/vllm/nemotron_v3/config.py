# ============================================================================
# DECOMPILED FROM /home/taejinp/projects/canary-dev/canary-dev/speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/__pycache__/config.cpython-312.pyc
# ----------------------------------------------------------------------------
# This file was destroyed and only the compiled bytecode (.pyc) survived.
# pycdc (https://github.com/zrax/pycdc) was used to decompile it; the output
# below is INCOMPLETE because Python 3.12 introduced opcodes (MAKE_CELL,
# LOAD_FAST_AND_CLEAR, LOAD_FAST_CHECK) that pycdc does not fully support.
#
# Sources of truth for finishing the restoration:
#   * full pycdas disassembly:  /home/taejinp/projects/canary-dev/canary-dev/speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/config.py.pyc_disasm.txt
#   * pycdc warnings:           /home/taejinp/projects/canary-dev/canary-dev/speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/config.py.pyc_decompile_warn.txt
#   * recorded StrReplace ops:  /home/taejinp/projects/canary-dev/canary-dev/speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/config.py.RESTORE_TODO.md
#     (these were applied on top of the original file during the destroyed
#      session; apply them again after restoring the body)
#
# Until this file is fully restored, importing
#   nemo.collections.speechlm2.vllm.nemotron_v3
# may fail.  The original is intact on draco-oci-iad at
#   /lustre/fs12/portfolios/llmservice/users/taejinp/projects/canary-dev/canary-dev/
#       speechlm-2026h1/NeMo/nemo/collections/speechlm2/vllm/nemotron_v3/
# so an rsync from there is the cleanest recovery.
# ============================================================================

"""Configuration for NeMo Speech LM models in vLLM.

Provides ``NeMoSpeechLMConfig``, a HuggingFace-compatible config class
that wraps the LLM backbone's text config with NeMo-specific fields
(perception, audio_locator_tag, etc.).  The checkpoint's ``config.json``
determines which LLM backbone and encoder are used.
"""
from transformers import AutoConfig, PretrainedConfig

class NeMoSpeechLMConfig(PretrainedConfig):
    pass
# WARNING: Decompyle incomplete

