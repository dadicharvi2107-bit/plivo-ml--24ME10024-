"""Byte-level BPE tokenizer (GPT-2 style).

Lossless by construction: base vocab is the 256 raw bytes, and every merge
just glues two existing token-byte-sequences together. decode() is always
the exact inverse of encode() because we never do anything except
concatenate byte strings and utf-8 decode at the end.

Usage:
    # one-time, offline, on the corpus only:
    python tokenizer.py train --data ../data/train_corpus.txt \
        --vocab_size 2000 --out bpe_merges.json

    # used automatically by train.py / evaluate.py:
    tok = load()          # reads bpe_merges.json next to this file
    ids = tok.encode(s)
    s2  = tok.decode(ids)
    assert s2 == s
"""
import argparse
import collections
import json
import os
import re
import time

# GPT-2-style pretokenization pattern, unicode-aware so Devanagari words
# stay intact as "words" (this only affects how merges are learned, not
# correctness -- encode() falls back cleanly on anything).
_PAT = re.compile(r""" ?\w+| ?[^\w\s]+|\s+""", re.UNICODE)

_DEFAULT_MERGES_PATH = os.path.join(os.path.dirname(__file__), "bpe_merges.json")


class BPETokenizer:
    def __init__(self, merges=None):
        # merges: ordered list of [id_a, id_b] -> new_id (new_id implied by order, starting at 256)
        self.merges = [tuple(m) for m in merges] if merges else []
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)
        # build id -> bytes table for fast decode
        self._id_to_bytes = [bytes([i]) for i in range(256)]
        for pair in self.merges:
            a, b = pair
            self._id_to_bytes.append(self._id_to_bytes[a] + self._id_to_bytes[b])

    # ---------- encode / decode ----------
    def _encode_word(self, word_bytes):
        ids = list(word_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge rank (earliest-learned merge) present
            best_rank, best_i = None, None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                r = self.merge_rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            a, b = ids[best_i], ids[best_i + 1]
            new_id = 256 + best_rank
            ids = ids[:best_i] + [new_id] + ids[best_i + 2:]
        return ids

    def encode(self, text):
        if not self.merges:
            return list(text.encode("utf-8"))
        out = []
        for w in _PAT.findall(text):
            out.extend(self._encode_word(w.encode("utf-8")))
        return out

    def decode(self, ids):
        return b"".join(self._id_to_bytes[i] for i in ids).decode("utf-8", errors="replace")

    # ---------- persistence ----------
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": self.merges}, f)

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(merges=d["merges"])


def load(path=None):
    """Return the tokenizer used by evaluate.py / train.py. No-arg by contract."""
    path = path or _DEFAULT_MERGES_PATH
    if os.path.exists(path):
        return BPETokenizer.from_file(path)
    # fallback: pure byte tokenizer if merges haven't been trained yet
    return BPETokenizer(merges=[])


# ============================================================
# Training (run once, offline, python tokenizer.py train ...)
# ============================================================
def train_bpe(text, num_merges, verbose=True):
    words = _PAT.findall(text)
    word_freq = collections.Counter(words)
    # unique-word table: cost is O(unique words), not O(corpus size), per merge
    seqs = [list(w.encode("utf-8")) for w in word_freq]
    freqs = list(word_freq.values())

    # pair -> count, and pair -> set of seq indices containing it (for incremental updates)
    pair_count = collections.Counter()
    pair_where = collections.defaultdict(set)
    for idx, seq in enumerate(seqs):
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            pair_count[p] += freqs[idx]
            pair_where[p].add(idx)

    merges = []
    next_id = 256
    t0 = time.time()
    for m in range(num_merges):
        if not pair_count:
            break
        best = max(pair_count, key=pair_count.get)
        if pair_count[best] < 2:
            break
        merges.append(best)
        affected = list(pair_where[best])
        for idx in affected:
            seq = seqs[idx]
            f = freqs[idx]
            # remove old pair counts for this seq
            for i in range(len(seq) - 1):
                p = (seq[i], seq[i + 1])
                pair_count[p] -= f
                if pair_count[p] <= 0:
                    del pair_count[p]
                pair_where[p].discard(idx)
            # merge
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == best[0] and seq[i + 1] == best[1]:
                    new_seq.append(next_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seqs[idx] = new_seq
            # add new pair counts for this seq
            for i in range(len(new_seq) - 1):
                p = (new_seq[i], new_seq[i + 1])
                pair_count[p] += f
                pair_where[p].add(idx)
        del pair_where[best]
        next_id += 1
        if verbose and (m % 200 == 0 or m == num_merges - 1):
            print(f"merge {m+1}/{num_merges}  vocab={256+len(merges)}  "
                  f"elapsed={time.time()-t0:.1f}s")
    return merges


def _roundtrip_check(tok, text, n=200000):
    sample = text[:n]
    assert tok.decode(tok.encode(sample)) == sample, "tokenizer is NOT lossless!"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tp = sub.add_parser("train")
    tp.add_argument("--data", required=True)
    tp.add_argument("--vocab_size", type=int, default=2000)
    tp.add_argument("--out", default=_DEFAULT_MERGES_PATH)
    args = ap.parse_args()

    if args.cmd == "train":
        text = open(args.data, encoding="utf-8").read()
        num_merges = args.vocab_size - 256
        assert num_merges > 0
        merges = train_bpe(text, num_merges)
        tok = BPETokenizer(merges=merges)
        _roundtrip_check(tok, text)
        tok.save(args.out)
        print(f"saved {args.out}  vocab_size={tok.vocab_size}")


if __name__ == "__main__":
    main()
