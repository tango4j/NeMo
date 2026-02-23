"""Rename <|spltoken*|> speaker tokens to [s*] in a SentencePiece model.

Operates in-place on the .model file and regenerates .vocab and vocab.txt.

Usage:
    python rename_spl_tokens_to_bracket.py \
        --tokenizer_dir /work/diarization/rnnt_bpe_cache_aware/tokenizer_spe_bpe_v1024_spk8brkt
"""

import argparse
import logging
import os
import re
import shutil

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OLD_PATTERN = re.compile(r"^<\|spltoken(\d+)\|>$")
NEW_TEMPLATE = "[s{}]"


def rename_tokens(tokenizer_dir: str, backup: bool = True):
    model_path = os.path.join(tokenizer_dir, "tokenizer.model")
    vocab_path = os.path.join(tokenizer_dir, "tokenizer.vocab")
    vocab_txt_path = os.path.join(tokenizer_dir, "vocab.txt")

    if backup:
        for fpath in [model_path, vocab_path, vocab_txt_path]:
            bak = fpath + ".bak"
            if os.path.exists(fpath) and not os.path.exists(bak):
                shutil.copy2(fpath, bak)
                log.info("Backed up %s -> %s", fpath, bak)

    model = sp_pb2.ModelProto()
    with open(model_path, "rb") as f:
        model.ParseFromString(f.read())

    renamed = {}
    for piece in model.pieces:
        m = OLD_PATTERN.match(piece.piece)
        if m:
            old_name = piece.piece
            new_name = NEW_TEMPLATE.format(m.group(1))
            piece.piece = new_name
            renamed[old_name] = new_name

    if not renamed:
        log.warning("No <|spltoken*|> tokens found — nothing to rename.")
        return

    log.info("Renamed %d tokens: %s", len(renamed), renamed)

    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())
    log.info("Wrote updated model to %s", model_path)

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    with open(vocab_path, "w", encoding="utf-8") as vf:
        for i in range(sp.GetPieceSize()):
            token = sp.IdToPiece(i)
            score = sp.GetScore(i)
            vf.write(f"{token}\t{score}\n")
    log.info("Wrote %s (%d entries)", vocab_path, sp.GetPieceSize())

    with open(vocab_txt_path, "w", encoding="utf-8") as tf:
        for i in range(sp.GetPieceSize()):
            token = sp.IdToPiece(i)
            if token.startswith("▁"):
                tf.write(token[1:] + "\n")
            else:
                tf.write("##" + token + "\n")
    log.info("Wrote %s (%d entries)", vocab_txt_path, sp.GetPieceSize())

    for old_name, new_name in renamed.items():
        token_id = sp.PieceToId(new_name)
        log.info("  %s -> %s  (id=%d)", old_name, new_name, token_id)

    log.info("Vocab size: %d", sp.GetPieceSize())


def main():
    parser = argparse.ArgumentParser(
        description="Rename <|spltoken*|> to [s*] in SentencePiece tokenizer"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=True,
        help="Directory containing tokenizer.model, tokenizer.vocab, vocab.txt",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Skip creating .bak backup files",
    )
    args = parser.parse_args()
    rename_tokens(args.tokenizer_dir, backup=not args.no_backup)


if __name__ == "__main__":
    main()
