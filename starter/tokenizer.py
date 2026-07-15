import os
import json

class BPETokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        
        # Load from file if it exists
        dir_path = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.join(dir_path, "bpe_vocab.json")
        if os.path.exists(vocab_path):
            self.load(vocab_path)
        else:
            print("WARNING: bpe_vocab.json not found! Running as fallback ByteTokenizer.")

    def encode(self, text):
        import re
        # Partition the text losslessly into words, spaces, and punctuation
        words = re.findall(r'\w+|\s+|[^\w\s]', text)
        
        cache = {}
        encoded_ids = []
        for word in words:
            if word not in cache:
                word_bytes = list(word.encode("utf-8"))
                ids = list(word_bytes)
                while len(ids) >= 2:
                    stats = {}
                    for pair in zip(ids, ids[1:]):
                        if pair in self.merges:
                            stats[pair] = self.merges[pair]
                    if not stats:
                        break
                    best_pair = min(stats, key=stats.get)
                    new_id = self.merges[best_pair]
                    new_ids = []
                    idx = 0
                    while idx < len(ids):
                        if idx < len(ids) - 1 and (ids[idx], ids[idx+1]) == best_pair:
                            new_ids.append(new_id)
                            idx += 2
                        else:
                            new_ids.append(ids[idx])
                            idx += 1
                    ids = new_ids
                cache[word] = ids
            encoded_ids.extend(cache[word])
        return encoded_ids

    def decode(self, ids):
        # Decode list of token IDs back to string (lossless)
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        merges_serial = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "merges": merges_serial
            }, f)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.merges = {}
        for k, v in data["merges"].items():
            pair = tuple(map(int, k.split(",")))
            self.merges[pair] = v
        # reconstruct vocab
        self.vocab = {i: bytes([i]) for i in range(256)}
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for pair, new_id in sorted_merges:
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]


def load(path=None):
    """Return the BPE tokenizer."""
    return BPETokenizer()
