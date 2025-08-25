import torch
from torch.utils.data import DataLoader
import logging
from data import GutenbergDataset, tokenizer, CLS_TOKEN_ID, get_dataset_text
from hrm import HierarchicalReasonerModel
from config import ModelConfig
from utils import train_model


# === SECTION: Logger setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hrm_run")
logger.info("=== HRM RUN SCRIPT START ===")


# === SECTION: Config selection ===
QUICK_RUN = True  # If True, use small config for fast testing; else use big config for full training

if QUICK_RUN:
    logger.info("QUICK_RUN is enabled => Using small_config.")
    from config import small_model_config as model_cfg_dict
else:
    logger.info("QUICK_RUN is disabled. Using big_config.")
    from config import big_model_config as model_cfg_dict


# === SECTION: Config ===
SEQ_LEN = model_cfg_dict["seq_len"]
BATCH_SIZE = model_cfg_dict.get("batch_size", 2)
EPOCHS = model_cfg_dict.get("n_epochs", 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_ACT = True
logger.info(f"Model config: {model_cfg_dict}")
logger.info(
    f"Input params: SEQ_LEN={SEQ_LEN}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, DEVICE={DEVICE}, USE_ACT={USE_ACT}"
)


# === SECTION: Prepare dataset ===
logger.info("--- Preparing dataset ---")
texts = get_dataset_text()
texts = " ".join(texts)
dataset = GutenbergDataset(
    texts, tokenizer, seq_len=SEQ_LEN + 1, cls_token_id=CLS_TOKEN_ID
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
logger.info(f"Dataset ready. Number of batches per epoch: {len(dataloader)}")


# === SECTION: Model config and initialization ===
logger.info("--- Initializing model ---")
config = ModelConfig(**model_cfg_dict)
model = HierarchicalReasonerModel(config).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
logger.info(f"Model initialized: {model.__class__.__name__}")
logger.info(f"Optimizer: AdamW, lr=1e-4")


# === SECTION: Training loop ===
logger.info("--- Starting training loop ---")
max_halt_steps = model_cfg_dict.get("max_halt_steps", 1)

avg_loss, trained_model = train_model(
    model,
    dataloader,
    optimizer,
    tokenizer,
    DEVICE,
    epochs=EPOCHS,
    use_act=USE_ACT,
    max_halt_steps=max_halt_steps,
    logger=logger,
    quick_run=QUICK_RUN,
)
logger.info(f"Training finished. Average loss: {avg_loss}")
logger.info("=== HRM RUN SCRIPT END ===")
