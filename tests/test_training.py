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
    config = ModelConfig(
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
    )
    print("Model config:", config.__dict__)
    batch_size = 2
    seq_len = config.seq_len
    model = HierarchicalReasonerModel(config)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create a simple dataloader with dummy batches
    # Add + 1 to seq_len for input/target slicing in train_model
    dummy_batches = [
        torch.randint(0, config.vocab_size, (batch_size, seq_len + 1), device=device)
        for _ in range(10)  # 10 dummy batches
    ]

    print("Training started...")
    avg_loss, _ = train_model(
        model,
        dummy_batches,
        optimizer,
        device,
        epochs=1,
        max_halt_steps=config.halt_max_steps,
        testing=True,  # Only use first 5 batches
    )
    print(f"train_model (ACT) average loss: {avg_loss}")
