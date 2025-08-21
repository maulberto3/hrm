import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from hrm_inner import HRMInner
from hrm_reasoner import ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_hrm_inner_basic():
    """
    Essential test for HRMInner: checks output shapes and keys.
    """
    config = ModelConfig(
        seq_len=8,
        vocab_size=100,
        high_level_cycles=2,
        low_level_cycles=2,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        expansion=4,
        norm_epsilon=0.1,
        rope_theta=10000.0,
        halt_max_steps=4,
        halt_exploration_prob=0.1,
    )
    batch_size = 2
    seq_len = config.seq_len

    model = HRMInner(config).to(device)
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    hidden_states = model.initial_hidden_states(batch_size, seq_len, device)

    outputs = model(hidden_states, inputs)

    assert "output" in outputs, "Output should contain 'output' key"
    assert "hidden_states" in outputs, "Output should contain 'hidden_states' key"
    assert outputs["output"].shape == (
        batch_size,
        seq_len,
        config.vocab_size,
    ), f"Unexpected output shape: {outputs['output'].shape}"
    assert outputs["hidden_states"]["high_level"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), f"Unexpected high_level shape: {outputs['hidden_states']['high_level'].shape}"
    assert outputs["hidden_states"]["low_level"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), f"Unexpected low_level shape: {outputs['hidden_states']['low_level'].shape}"

    print("HRMInner essential test passed. Output shape:", outputs["output"].shape)


if __name__ == "__main__":
    test_hrm_inner_basic()
