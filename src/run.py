import os
import torch
from torch.utils.data import DataLoader
import logging
from data import GutenbergDataset, get_tokenizer_and_text
from hrm import HierarchicalReasonerModel
from config import ModelConfig
from train import train_model


# === Set up Logger ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s --",
)
logger = logging.getLogger("__name__")
logger.info("=== HRM RUN SCRIPT START ===")


# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# === Config selection ===
SMALL = True
if SMALL:
    logger.info("Using small_config.")
    from config import small_config as config
else:
    logger.info("Using big_config.")
    from config import big_config as config


# === Config ===
SEQ_LEN = config["seq_len"]
BATCH_SIZE = config["batch_size"]
EPOCHS = config["n_epochs"]
GENERATE_EVERY = config["generate_every"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
logger.info(f"Model config: {config}")


# === Prepare dataset ===
logger.info("--- Preparing dataset ---")
tokenizer, text = get_tokenizer_and_text()
dataset = GutenbergDataset(text, tokenizer, seq_len=SEQ_LEN + 1)


# === Create DataLoader ===
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    # num_workers=2,
)
logger.info(f"Dataloader ready. Number of batches per epoch: {len(dataloader):,}")


# === Model config and initialization ===
model_config = ModelConfig(**config)
model = HierarchicalReasonerModel(model_config).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
logger.info(f"Model initialized: {model.__class__.__name__}")
logger.info(f"Optimizer: Adam, lr={config['lr']}")


# === Training loop ===
TESTING = False
logger.info("--- Starting training loop ---")
avg_loss, trained_model = train_model(
    model,
    dataloader,
    optimizer,
    device=DEVICE,
    config=config,
    testing=TESTING,
)
logger.info(f"Training finished. Average loss: {avg_loss}")
