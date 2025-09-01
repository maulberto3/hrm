import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from hrm import HierarchicalReasonerModel
from config import ModelConfig
from train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def test_train_model_simple():
    """
    Test train_model with testing=True (only first 5 batches).
    """
    print("\n=== TEST: train_model_simple ===")
    model_config = ModelConfig(
        seq_len=64,
        vocab_size=100,
        high_level_cycles=2,
        low_level_cycles=2,
        num_layers=2,
        hidden_size=32,
        num_heads=4,
        expansion=4,
        halt_max_steps=4,
        halt_exploration_prob=0.1,
        norm_epsilon=0.25,
        rope_theta=500.0,
        halt_min_steps=4,
        max_new_tokens=96,
        temperature=0.5,
        do_sample=True,
        top_p=0.995,
        max_length=256,
    )
    print("Model config:", model_config.__dict__)
    batch_size = 2
    seq_len = model_config.seq_len
    model = HierarchicalReasonerModel(model_config)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create a simple dataloader with dummy batches
    # Add + 1 to seq_len for input/target slicing in train_model
    dummy_batches = [
        torch.randint(
            0, model.config.vocab_size, (batch_size, seq_len + 1), device=device
        )
        for _ in range(10)  # 10 dummy batches
    ]

    config = {
        "seq_len": 256,
        "vocab_size": 32000,
        "high_level_cycles": 8,
        "low_level_cycles": 8,
        "num_layers": 6,
        "hidden_size": 504,
        "num_heads": 12,
        "expansion": 4.0,
        "norm_epsilon": 1e-5,
        "rope_theta": 10000.0,
        "halt_max_steps": 32,
        "halt_exploration_prob": 0.1,
        "halt_min_steps": 1,
        "max_new_tokens": 256,
        "temperature": 0.8,
        "do_sample": True,
        "top_p": 0.9,
        "max_length": 1024,
        "cls_token": "[CLS]",
        "pad_token": "[PAD]",
        "eos_token": "[EOS]",
        "batch_size": 16,
        "n_epochs": 50,
        "generate_every": 15,
        "lr": 1e-4,
        "checkpoint_dir": "/home/maulb/hrm/data/",
    }

    print("Training started...")
    avg_loss, _ = train_model(
        model,
        dummy_batches,
        optimizer,
        device,
        config,
        testing=True,  # Only use first 5 batches
    )
    print(f"train_model (ACT) average loss: {avg_loss}")
