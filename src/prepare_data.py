"""prepare_data.py — tokenize FineWeb-Edu parquet shards into uint16 token bins.

Outputs (in data/):
  tok_full_train.bin / tok_full_val.bin   : GPT-2 token ids (uint16), full 50257 vocab (Track B)
  tok16k_train.bin / tok16k_val.bin       : ids mapped to top-16383 GPT-2 tokens + UNK=16383 (Track C)
  counts16k.npy                           : train-window token counts for the 16k vocab (for hybrid tables)

Vocab-cap mapping: keep the top (V-1) most frequent GPT-2 tokens from the tokenized stream, map
everything else (incl. rare ids) to UNK = V-1.
"""
import argparse, os
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def build_tokenizer():
    tok = Tokenizer(BPE.from_file(os.path.join(DATA, "gpt2_vocab.json"), os.path.join(DATA, "gpt2_merges.txt"),
                                   unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    return tok

def tokenize_shard(path, tok, need):
    """stream-tokenize until `need` tokens collected (memory-safe)."""
    tbl = pq.read_table(path, columns=["text"])
    texts = tbl["text"].to_pylist()
    out = np.empty(need + 1_000_000, dtype=np.uint16)
    n = 0; B = 1000
    for i in range(0, len(texts), B):
        enc = tok.encode_batch(texts[i:i + B])
        for e in enc:
            m = len(e.ids)
            if n + m > need:
                out[n:need] = e.ids[:need - n]; n = need; break
            out[n:n + m] = e.ids; n += m
        if n >= need: break
        if (i // B) % 50 == 0: print(f"  tokenized {n/1e6:.0f}M tokens", flush=True)
    return out[:need]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=[os.path.join(DATA, "fineweb_edu_00.parquet")])
    ap.add_argument("--train_tokens", type=int, default=120_000_000, help="tokens for the train bin")
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--vcap", type=int, default=16384)
    args = ap.parse_args()
    tok = build_tokenizer()

    need = args.train_tokens + args.val_tokens
    chunks = []
    got = 0
    for sh in args.shards:
        p = os.path.join(DATA, sh) if not os.path.isabs(sh) else sh
        print("tokenizing", p, flush=True)
        ids = tokenize_shard(p, tok, need - got)
        chunks.append(ids); got += len(ids)
        if got >= need: break
    stream = np.concatenate(chunks)[:need]
    train, val = stream[:args.train_tokens], stream[args.train_tokens:need]

    train.tofile(os.path.join(DATA, "tok_full_train.bin"))
    val.tofile(os.path.join(DATA, "tok_full_val.bin"))
    print("full bins written:", len(train), len(val), flush=True)

    # ---- 16k capped vocab (Track C) ----
    V = args.vcap
    sample = train[: min(len(train), 40_000_000)]
    counts = np.bincount(sample.astype(np.int64), minlength=50257).astype(np.int64)
    keep = np.argsort(counts)[::-1][:V - 1]                      # top V-1 GPT-2 ids
    lut = np.full(50257, V - 1, dtype=np.uint16)                 # default UNK
    lut[keep] = np.arange(V - 1, dtype=np.uint16)
    tr16 = lut[train.astype(np.int64)]; va16 = lut[val.astype(np.int64)]
    tr16.tofile(os.path.join(DATA, "tok16k_train.bin"))
    va16.tofile(os.path.join(DATA, "tok16k_val.bin"))
    cnt16 = np.bincount(tr16.astype(np.int64), minlength=V).astype(np.int64)
    np.save(os.path.join(DATA, "counts16k.npy"), cnt16)
    print("16k bins written; UNK share in train:", float((tr16 == V - 1).mean()), flush=True)

EOT = 50256

if __name__ == "__main__":
    main()
