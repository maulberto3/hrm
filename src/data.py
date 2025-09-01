import os
import requests
import torch
import tokenizers
import logging

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logger setup (module-level, best practice)
logger = logging.getLogger(__name__)

# Dynamically set VOCAB_SIZE from config
try:
    from config import small_config as config
except ImportError:
    from config import big_config as config

VOCAB_SIZE = config["vocab_size"]
tokenizer_path = f"data/gutenberg_tokenizer_vocab_size_{VOCAB_SIZE}.json"

# Tokenization with Byte-Pair Encoding (BPE)
SPECIAL_TOKENS = [
    config["cls_token"],
    config["pad_token"],
    config["eos_token"],
]

# Data directory for storing downloaded texts
DATA_DIR = "data"

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


# Download text files from Project Gutenberg
def download_text():
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename, url in DATASOURCE.items():
        file_path = os.path.join(DATA_DIR, f"{filename}.txt")
        if not os.path.exists(file_path):
            logger.info(f"Downloading {filename} from {url}")
            response = requests.get(url)
            with open(file_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Saved {filename} to {file_path}")
        else:
            logger.info(f"Using cached file for {filename[:5]}")


# Read and preprocess the text from a Gutenberg file
def preprocess_gutenberg(filename):
    logger.info(f"Preprocessing {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Find the start and end of the actual content using Gutenberg markers
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.find(start_marker)
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0  # If marker not found, start at beginning
    end = text.find(end_marker)
    if end == -1:
        end = len(text)  # If marker not found, go to end

    # Extract the main content between the markers
    main_text = text[start:end].strip()

    # Basic preprocessing: remove empty lines and extra spaces
    processed_text = "\n".join(
        line.strip() for line in main_text.split("\n") if line.strip()
    )
    return processed_text


# Get all texts from the dataset, one per book
def get_dataset_text():
    # Ensure all files are downloaded before preprocessing
    download_text()

    logger.info("Loading and preprocessing all dataset texts")
    concat_path = os.path.join(DATA_DIR, "concatenated_gutenberg.txt")
    if os.path.exists(concat_path):
        logger.info(f"Loading cached concatenated text from {concat_path}")
        with open(concat_path, "r", encoding="utf-8") as f:
            return [f.read()]
    else:
        all_text = []
        for filename in DATASOURCE:
            file_path = os.path.join(DATA_DIR, f"{filename}.txt")
            text = preprocess_gutenberg(file_path)
            all_text.append(text)

        # Cache concatenated text for future runs
        with open(concat_path, "w", encoding="utf-8") as f:
            f.write(" ".join(all_text))
        return all_text


# Load tokenizer from file if available, otherwise train a new one
def get_tokenizer_and_text():
    text = get_dataset_text()
    if os.path.exists(tokenizer_path):
        logger.info(f"Using previously trained tokenizer from {tokenizer_path}")
        tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    else:
        logger.info(
            "No trained tokenizer found. Training new BPE tokenizer from scratch."
        )
        tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
            add_prefix_space=True
        )
        tokenizer.decoder = tokenizers.decoders.ByteLevel()
        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS, show_progress=True
        )
        tokenizer.train_from_iterator(text, trainer=trainer)
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id("[PAD]"),
            pad_token="[PAD]",
            length=config["seq_len"],
        )
        tokenizer.save(tokenizer_path, pretty=True)
        logger.info(f"Saved trained tokenizer to {tokenizer_path}")
    return tokenizer, text


# PyTorch dataset for Gutenberg text
class GutenbergDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, seq_len):
        logger.info("Initializing GutenbergDataset")

        # Cache concatenated text for future runs
        concat_path = os.path.join(DATA_DIR, "concatenated_gutenberg.txt")
        if os.path.exists(concat_path):
            logger.info(f"Loading cached concatenated text from {concat_path}")
            with open(concat_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            logger.info(f"Caching concatenated text to {concat_path}")
            with open(concat_path, "w", encoding="utf-8") as f:
                f.write(text)

        self.seq_len = seq_len
        self.cls_token_id = tokenizer.token_to_id("[CLS]")
        self.encoded = tokenizer.encode(text).ids

    def __len__(self):
        # Number of possible sequences in the encoded text
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        # Prepend CLS token to each sequence for model input
        seq = [self.cls_token_id] + self.encoded[idx : idx + self.seq_len - 1]
        return torch.tensor(seq)
