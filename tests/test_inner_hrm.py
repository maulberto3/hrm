import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from hrm_inner import HRMInner
from config import ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_hrm_inner_basic():
    """
    Essential test for HRMInner: checks output shapes and keys.
    """
    print("\n=== TEST: test_hrm_inner_basic ===")
    config = ModelConfig(
        seq_len=256,
        vocab_size=100,
        high_level_cycles=2,
        low_level_cycles=2,
        num_layers=2,
        hidden_size=16,
        num_heads=4,
        expansion=4,
        norm_epsilon=0.1,
        rope_theta=10000.0,
        halt_max_steps=4,
        halt_exploration_prob=0.1,
    )
    print("\nModel config:", config.__dict__)
    batch_size = 2
    seq_len = config.seq_len

    inner_model = HRMInner(config).to(device)

    # Assuming first token is CLS as per paper and here implemented in dataloader (not inside model)
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    print("Input shape:", inputs.shape)

    hidden_states = inner_model.initial_hidden_states(batch_size, seq_len, device)
    print("Hidden states (high_level) shape:", hidden_states["high_level"].shape)
    print("Hidden states (low_level) shape:", hidden_states["low_level"].shape)

    outputs = inner_model(hidden_states, inputs)

    assert "output" in outputs, "Output should contain 'output' key"
    print(f"Output shape: {outputs['output'].shape}")
    print(f"Output sample: {outputs['output'][0,0,:4].detach().cpu().numpy()}")
    print(f"High-level hidden shape: {outputs['hidden_states']['high_level'].shape}")
    print(f"Low-level hidden shape: {outputs['hidden_states']['low_level'].shape}")

    assert outputs["hidden_states"]["low_level"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), f"Unexpected low_level shape: {outputs['hidden_states']['low_level'].shape}"

    print("HRMInner essential test passed.")
