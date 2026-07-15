import json
import os
import time

class BPETrainer:
    def __init__(self):
        self.vocab_size = 256
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    def train(self, text, target_vocab_size):
        assert target_vocab_size >= 256
        num_merges = target_vocab_size - 256
        text_bytes = list(text.encode("utf-8"))
        ids = list(text_bytes)
        
        t0 = time.time()
        for i in range(num_merges):
            stats = {}
            for pair in zip(ids, ids[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # replace all occurrences of best_pair in ids
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
            if (i + 1) % 50 == 0 or i == num_merges - 1:
                print(f"merge {i+1}/{num_merges}: {best_pair} -> {new_id} (frequency {stats[best_pair]})")
        
        print(f"BPE training took {time.time() - t0:.1f}s")
        self.vocab_size = target_vocab_size

    def save(self, path):
        merges_serial = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "merges": merges_serial
            }, f)

def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(dir_path, "../data/train_corpus.txt")
    print(f"Loading corpus from {corpus_path}...")
    text = open(corpus_path, "r", encoding="utf-8").read()
    
    # Train BPE on a subset (1 MB) to be fast
    subset_size = 200 * 1024
    print(f"Training BPE on first {subset_size:,} bytes of corpus...")
    trainer = BPETrainer()
    trainer.train(text[:subset_size], target_vocab_size=2000)
    
    out_path = os.path.join(dir_path, "bpe_vocab.json")
    print(f"Saving vocabulary to {out_path}...")
    trainer.save(out_path)
    print("Done!")

if __name__ == "__main__":
    main()
