import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tokenizers
import tqdm
from random import random


####################
# DATA
####################

# Download novels from Project Gutenberg
DATASOURCE = {
    "moby_dick": "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "dracula": "https://www.gutenberg.org/ebooks/345.txt.utf-8",
    "little_women": "https://www.gutenberg.org/ebooks/37106.txt.utf-8",
    "pride_and_prejudice": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
    "alice_in_wonderland": "https://www.gutenberg.org/ebooks/11.txt.utf-8",
    "crime_and_punishment": "https://www.gutenberg.org/ebooks/2554.txt.utf-8",
    "tom_sawyer": "https://www.gutenberg.org/ebooks/74.txt.utf-8",
    "tale_of_two_cities": "https://www.gutenberg.org/ebooks/98.txt.utf-8",
    "sherlock_holmes": "https://www.gutenberg.org/ebooks/1661.txt.utf-8",
    "war_and_peace": "https://www.gutenberg.org/ebooks/2600.txt.utf-8",
}
for filename, url in DATASOURCE.items():
    if not os.path.exists(f"{filename}.txt"):
        response = requests.get(url)
        with open(f"{filename}.txt", "wb") as f:
            f.write(response.content)

# Read and preprocess the text
def preprocess_gutenberg(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Find the start and end of the actual content
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.find("\n", start) + 1
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")

    # Extract the main content
    text = text[start:end].strip()

    # Basic preprocessing
    # Remove multiple newlines and spaces
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text

def get_dataset_text():
    all_text = []
    for filename in DATASOURCE:
        text = preprocess_gutenberg(f"{filename}.txt")
        all_text.append(text)
    return all_text

# Tokenization with BPE
if os.path.exists("gutenberg_tokenizer.json"):
    tokenizer = tokenizers.Tokenizer.from_file("gutenberg_tokenizer.json")
else:
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    # Configure pre-tokenizer add space at beginning of the sentence
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    # Configure decoder so that would boundary symbol will be removed
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    # Train BPE
    VOCAB_SIZE = 10000
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[pad]", "[eos]"],
        show_progress=True
    )
    text = get_dataset_text()
    tokenizer.train_from_iterator(text, trainer=trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[pad]"), pad_token="[pad]")
    # Save the trained tokenizer
    tokenizer.save("gutenberg_tokenizer.json", pretty=True)

# Create PyTorch dataset
class GutenbergDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, seq_len=512):
        self.seq_len = seq_len
        # Encode the entire text
        self.encoded = tokenizer.encode(text).ids

    def __len__(self):
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]  # +1 for target
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y