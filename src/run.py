import os
import torch
from torch.utils.data import DataLoader
import logging
from data import GutenbergDataset, get_tokenizer_and_text
from hrm import HierarchicalReasonerModel
from config import ModelConfig
from train import train_model

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Set up Logger ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s --",
)
logger = logging.getLogger("__name__")
logger.info("=== HRM RUN SCRIPT START ===")


# === Config selection ===
TESTING = True  # If True, use small config for fast testing; else use big config for full training
if TESTING:
    logger.info("TESTING is enabled => Using small_config.")
    from config import small_model_config as model_cfg_dict
else:
    logger.info("TESTING is disabled. Using big_config.")
    from config import big_model_config as model_cfg_dict


# === Config ===
SEQ_LEN = model_cfg_dict["seq_len"]
BATCH_SIZE = model_cfg_dict.get("batch_size", 2)
EPOCHS = model_cfg_dict.get("n_epochs", 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
logger.info(f"Model config: {model_cfg_dict}")


# === Prepare dataset ===
logger.info("--- Preparing dataset ---")
tokenizer, text = get_tokenizer_and_text()
dataset = GutenbergDataset(
    text, tokenizer, seq_len=SEQ_LEN + 1, cls_token_id=tokenizer.token_to_id("[cls]")
)
logger.info(f"Dataset ready. Number of sentences: {len(dataset):,}")


# === Create DataLoader ===
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)
logger.info(f"Dataloader ready. Number of batches per epoch: {len(dataloader):,}")


# === Model config and initialization ===
config = ModelConfig(**model_cfg_dict)
model = HierarchicalReasonerModel(config).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
logger.info(f"Model initialized: {model.__class__.__name__}")
logger.info(f"Optimizer: AdamW, lr=1e-4")


# === Training loop ===
logger.info("--- Starting training loop ---")
max_halt_steps = model_cfg_dict.get("max_halt_steps", 1)

# avg_loss, trained_model = train_one_batch(
#     model,
#     next(iter(dataloader)),
#     optimizer,
#     DEVICE,
#     max_halt_steps=model_cfg_dict.get("max_halt_steps", 8),
# )

avg_loss, trained_model = train_model(
    model,
    dataloader,
    optimizer,
    DEVICE,
    max_halt_steps=max_halt_steps,
    testing=TESTING,
)
logger.info(f"Training finished. Average loss: {avg_loss}")
