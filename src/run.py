import torch
from torch.utils.data import DataLoader
import logging
from data import GutenbergDataset, tokenizer, CLS_TOKEN_ID, get_dataset_text
from hrm import HierarchicalReasonerModel
from hrm_reasoner import ModelConfig
from utils import (
    train_model,
    train_one_epoch,
    train_one_batch,
    run_act_forward,
    run_standard_forward,
    compute_act_loss,
    init_hidden_states,
)


# --- Logger setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hrm_run")


# --- Config selection ---
QUICK_RUN = True  # If True, use small config for fast testing; else use big config for full training

if QUICK_RUN:
    logger.info("QUICK_RUN is enabled => Using small_config.")
    from small_config import small_model_config as model_cfg_dict
else:
    logger.info("QUICK_RUN is disabled. Using big_config.")
    from big_config import big_model_config as model_cfg_dict


# --- Config ---
SEQ_LEN = model_cfg_dict["seq_len"]  # Sequence length for model input
BATCH_SIZE = model_cfg_dict.get("batch_size", 2)  # Batch size for training
EPOCHS = model_cfg_dict.get("n_epochs", 1)  # Number of training epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(
    f"SEQ_LEN={SEQ_LEN}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, DEVICE={DEVICE}"
)


# --- Prepare dataset ---
logger.info("PREPARING DATASET...")
# Load and concatenate all texts, then create dataset and dataloader
texts = get_dataset_text()
dataset = GutenbergDataset(
    " ".join(texts), tokenizer, seq_len=SEQ_LEN, cls_token_id=CLS_TOKEN_ID
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
logger.info(f"Dataset ready. Number of batches per epoch: {len(dataloader)}")


# --- Model config ---
logger.info("INITIALIZING MODEL...")
# Build model config and instantiate model and optimizer
config = ModelConfig(**model_cfg_dict)
model = HierarchicalReasonerModel(config).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
logger.info("Model and optimizer initialized.")


# --- Training loop ---
logger.info("STARTING TRAINING LOOP...")
use_act = "max_segments" in model_cfg_dict and model_cfg_dict["max_segments"] > 1
max_segments = model_cfg_dict.get("max_segments", 1)
# Modular training call (can use train_model, train_one_epoch, etc. as needed)
train_model(
    model,
    dataloader,
    optimizer,
    tokenizer,
    DEVICE,
    epochs=EPOCHS,
    use_act=use_act,
    max_segments=max_segments,
    logger=logger,
    quick_run=QUICK_RUN,
)
