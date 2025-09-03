import torch
import sys
import os

sys.path.append("src")

from data import GutenbergDataset
from config import small_config


# Minimal dummy text and tokenizer
class DummyTokenizer:
    def __init__(self, vocab_size=10):
        self.vocab_size = vocab_size
        self._tok_map = {}

        # First set up regular tokens
        for i in range(vocab_size):
            self._tok_map[f"tok{i}"] = i

        # Then override with special tokens
        self._tok_map[small_config["cls_token"]] = 0
        self._tok_map[small_config["pad_token"]] = 1
        print("\n\nToken mapping:", self._tok_map)

    def token_to_id(self, tok):
        return self._tok_map.get(tok, 999)

    def encode(self, text):
        token_ids = [self.token_to_id(t) for t in text.split()]
        return type("Dummy", (), {"ids": token_ids})()


# Set trunc_seq_prob high for demonstration
config = dict(small_config)
config["seq_len"] = 8

# Create dummy text (tokens separated by space)
dummy_text = (
    "tok2 tok3 tok4 tok5 tok6 tok7 tok8 tok9 tok2 tok3 tok4 tok5 tok6 tok7 tok8 tok9"
)
tokenizer = DummyTokenizer(vocab_size=10)

print("Testing random truncation in GutenbergDataset:")
print("Expected sequence without truncation: [0, 2, 3, 4, 5, 6, 7, 8]")
print(f"PAD token ID: {tokenizer.token_to_id(config['pad_token'])}")
print()

# Test with truncation OFF
config["trunc_seq_prob"] = 0.0
# Dataset will use cache functionality...
dataset = GutenbergDataset(dummy_text, tokenizer, config)
print("=== With truncation OFF (trunc_seq_prob=0.0) ===")
for i in range(3):
    seq = dataset[i].tolist()
    print(f"Sample {i}: {seq}")
print()

# Test with truncation ON
config["trunc_seq_prob"] = 1.0  # 100% for clear demonstration
dataset = GutenbergDataset(dummy_text, tokenizer, config)
print("=== With truncation ON (trunc_seq_prob=1.0) ===")
for i in range(5):
    seq = dataset[i].tolist()
    pad_id = tokenizer.token_to_id(config["pad_token"])
    print(f"Sample {i}: {seq}")
    if pad_id in seq[2:]:  # Check for padding after CLS and first real token
        first_pad_idx = seq.index(pad_id)
        print(f"  -> Truncated at index {first_pad_idx}, padded with {pad_id}")
