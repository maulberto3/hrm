import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from hrm_reasoner import (
    ModelConfig,
    ReasoningBlock,
    ReasonerModule,
)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_reasoning_block_basic():
    """
    Test ReasoningBlock - combines attention, MLP, and normalization.
    Data flow: input -> Attention -> + input -> RMSNorm -> MLP -> + input -> RMSNorm -> output
    HRM Context:
    - Core building block of the ReasonerModule.
    - Performs attention-based information aggregation and non-linear transformation.
    """
    # ModelConfig needs to be created to instantiate ReasoningBlock
    config = ModelConfig(
        seq_len=16,
        vocab_size=100,
        high_level_cycles=2,
        low_level_cycles=2,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        expansion=4,
        halt_max_steps=16,
        halt_exploration_prob=0.1,
    )
    batch_size = 2
    seq_len = 8
    hidden_size = config.hidden_size

    reasoning_block = ReasoningBlock(config).to(device)
    # Input: [batch_size, seq_len, hidden_size]
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    output = reasoning_block(input_tensor)

    assert output.shape == (
        batch_size,
        seq_len,
        hidden_size,
    ), f"Unexpected shape: {output.shape}"
    # Verify ReasoningBlock actually processes the input
    assert not torch.allclose(
        input_tensor, output, atol=1e-3
    ), "ReasoningBlock should transform input"
    print("ReasoningBlock test passed. Output shape:", output.shape)


def test_reasoner_module_basic():
    """
    Test ReasonerModule - stacks multiple ReasoningBlocks.
    Data flow: input_injection + hidden_state -> [ReasoningBlock x num_layers] -> output
    HRM Context:
    - Applies multiple reasoning steps within either the high-level or low-level reasoner.
    - Allows for deeper information processing and transformation.
    """
    # ModelConfig needs to be created to instantiate ReasoningBlock and ReasonerModule
    config = ModelConfig(
        seq_len=16,
        vocab_size=100,
        high_level_cycles=2,
        low_level_cycles=2,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        expansion=4,
        halt_max_steps=16,
        halt_exploration_prob=0.1,
    )
    batch_size = 2
    seq_len = 8
    hidden_size = config.hidden_size

    reasoner_module = ReasonerModule(config).to(device)
    # Input: [batch_size, seq_len, hidden_size]
    hidden_state = torch.randn(batch_size, seq_len, hidden_size, device=device)
    input_injection = torch.randn(batch_size, seq_len, hidden_size, device=device)
    output = reasoner_module(hidden_state, input_injection)

    assert output.shape == (
        batch_size,
        seq_len,
        hidden_size,
    ), f"Unexpected shape: {output.shape}"
    # Verify ReasonerModule actually processes the input
    assert not torch.allclose(
        hidden_state, output, atol=1e-3
    ), "ReasonerModule should transform input"
    print("ReasonerModule test passed. Output shape:", output.shape)


if __name__ == "__main__":
    test_reasoning_block_basic()
    test_reasoner_module_basic()
