import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from hrm import HierarchicalReasonerModel
from config import ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- New tests using train_one_batch from utils.py ---
from utils import train_one_batch


def test_train_one_batch_simple():
    """
    Test train_one_batch with use_act=True (ACT logic).
    """
    print("\n=== TEST: train_one_batch_simple ===")
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
    print("\nModel config:", config.__dict__)
    batch_size = 2
    seq_len = config.seq_len
    model = HierarchicalReasonerModel(config)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy batch: [batch_size, seq_len] (CLS handled by dataloader in real data)
    # Trick here is to add + 1 to seq_len to anticipate nlp modelling
    # where the x is all tokens but last one and y is all tokens
    # starting from second one
    # However, as the model has already been configured with seq_len,
    # we can't make it work with that x i.e. one less token
    # so the trick is to add + 1 to seq_len in the batch
    # so that when the model sees the x it has the correct length
    batch = torch.randint(
        0, config.vocab_size, (batch_size, seq_len + 1), device=device
    )
    print("\nInput shape:", batch.shape)
    print("Batch contents:\n", batch)

    # Dummy tokenizer with token_to_id method
    class DummyTokenizer:
        def token_to_id(self, token):
            return 0

    tokenizer = DummyTokenizer()
    print("Training started...")
    print("ACT Mode ON:")
    loss, _ = train_one_batch(
        model,
        batch,
        optimizer,
        tokenizer,
        device,
        use_act=True,
        max_halt_steps=config.halt_max_steps,
    )
    print(f"train_one_batch (ACT) loss: {loss}")

    print("ACT Mode OFF:")
    loss, _ = train_one_batch(
        model,
        batch,
        optimizer,
        tokenizer,
        device,
        use_act=False,
        max_halt_steps=config.halt_max_steps,
    )
    print(f"train_one_batch (CrossEntropy) loss: {loss}")
    print("Training finished.")
