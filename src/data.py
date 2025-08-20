import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tokenizers
import tqdm
from random import random
import logging

# --- Logger setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hrm_data")


####################
# DATA
####################

# Download novels from Project Gutenberg
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the data directory exists

# Mapping of book names to their download URLs
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
# Download each book if not already cached in the data directory
for filename, url in DATASOURCE.items():
    file_path = os.path.join(DATA_DIR, f"{filename}.txt")
    if not os.path.exists(file_path):
        logger.info(f"Downloading {filename} from {url}")
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Saved {filename} to {file_path}")
    else:
        logger.info(f"Using cached file for {filename}: {file_path}")


# Read and preprocess the text from a Gutenberg file
def preprocess_gutenberg(filename):
    logger.info(f"Preprocessing {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Find the start and end of the actual content using Gutenberg markers
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.find("\n", start) + 1
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")

    # Extract the main content between the markers
    text = text[start:end].strip()

    # Basic preprocessing: remove empty lines and extra spaces
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text


# Get all texts from the dataset, one per book
def get_dataset_text():
    logger.info("Loading and preprocessing all dataset texts")
    all_text = []
    for filename in DATASOURCE:
        file_path = os.path.join(DATA_DIR, f"{filename}.txt")
        text = preprocess_gutenberg(file_path)
        all_text.append(text)
    return all_text


# Tokenization with Byte-Pair Encoding (BPE)
SPECIAL_TOKENS = ["[CLS]", "[pad]", "[eos]"]

# Load tokenizer from file if available, otherwise train a new one
if os.path.exists("gutenberg_tokenizer.json"):
    logger.info("Using previously trained tokenizer from gutenberg_tokenizer.json")
    tokenizer = tokenizers.Tokenizer.from_file("gutenberg_tokenizer.json")
else:
    logger.info("No trained tokenizer found. Training new BPE tokenizer from scratch.")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    VOCAB_SIZE = 10000
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS, show_progress=True
    )
    text = get_dataset_text()
    tokenizer.train_from_iterator(text, trainer=trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[pad]"), pad_token="[pad]")
    tokenizer.save("gutenberg_tokenizer.json", pretty=True)
    logger.info("Saved trained tokenizer to gutenberg_tokenizer.json")

# Get the token ID for the [CLS] token
CLS_TOKEN_ID = tokenizer.token_to_id("[CLS]")


# PyTorch dataset for Gutenberg text
class GutenbergDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, seq_len=512, cls_token_id=None):
        logger.info("Initializing GutenbergDataset")
        self.seq_len = seq_len
        # Use provided CLS token ID or get it from tokenizer
        self.cls_token_id = (
            cls_token_id if cls_token_id is not None else tokenizer.token_to_id("[CLS]")
        )
        # Encode the entire text into token IDs
        self.encoded = tokenizer.encode(text).ids

    def __len__(self):
        # Number of possible sequences in the encoded text
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        # Prepend CLS token to each sequence for model input
        seq = [self.cls_token_id] + self.encoded[idx : idx + self.seq_len - 1]
        return torch.tensor(seq)
