import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from hrm_reasoner import (
    ReasoningBlock,
    ReasonerModule,
)
from hrm_building_blocks import RotaryEmbedding
from config import ModelConfig

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_reasoning_block_basic():
    """
    Test ReasoningBlock - combines attention, MLP, and normalization.
    """
    print("\n=== TEST: test_reasoning_block_basic ===")

    # ModelConfig needs to be created to instantiate ReasoningBlock
    config = ModelConfig(
        seq_len=128,
        vocab_size=100,
        high_level_cycles=2,
        low_level_cycles=2,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        expansion=4,
        norm_epsilon=0.1,
        rope_theta=10000.0,
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        halt_min_steps=3,
        max_new_tokens=144,
        temperature=0.75,
        do_sample=True,
        top_p=0.65,
        max_length=256,
    )
    print("\nModel config:", config.__dict__)

    batch_size = 2
    hidden_size = config.hidden_size
    head_dim = hidden_size // config.num_heads
    num_heads = config.num_heads

    reasoning_block = ReasoningBlock(config).to(device)
    rotary_emb = RotaryEmbedding(head_dim, max_length=config.seq_len).to(device)
    # Input: [batch_size, seq_len, hidden_size]
    input_tensor = torch.randn(batch_size, config.seq_len, hidden_size, device=device)
    input_tensor_flat = input_tensor.view(
        batch_size * num_heads, config.seq_len, head_dim
    )

    # Get cos and sin embeddings
    cos_sin = rotary_emb(input_tensor_flat)

    output = reasoning_block(input_tensor, cos_sin)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0,0,:4].detach().cpu().numpy()}")
    assert output.shape == (
        batch_size,
        config.seq_len,
        hidden_size,
    ), f"Unexpected shape: {output.shape}"
    # Verify ReasoningBlock actually processes the input
    assert not torch.allclose(
        input_tensor, output, atol=1e-3
    ), "ReasoningBlock should transform input"
    print("ReasoningBlock test passed.")


def test_reasoner_module_basic():
    """
    Test ReasonerModule - stacks multiple ReasoningBlocks.
    """
    print("\n=== TEST: test_reasoner_module_basic ===")

    config = ModelConfig(
        seq_len=128,
        vocab_size=100,
        high_level_cycles=2,
        low_level_cycles=2,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        expansion=4,
        norm_epsilon=0.1,
        rope_theta=10000.0,
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        halt_min_steps=3,
        max_new_tokens=144,
        temperature=0.75,
        do_sample=True,
        top_p=0.65,
        max_length=256,
    )
    print("\nModel config:", config.__dict__)

    batch_size = 2
    hidden_size = config.hidden_size
    head_dim = hidden_size // config.num_heads
    num_heads = config.num_heads

    reasoner_module = ReasonerModule(config).to(device)
    rotary_emb = RotaryEmbedding(head_dim, max_length=config.seq_len).to(device)
    # Input: [batch_size, seq_len, hidden_size]
    hidden_state = torch.randn(batch_size, config.seq_len, hidden_size, device=device)
    input_injection = torch.randn(
        batch_size, config.seq_len, hidden_size, device=device
    )

    # Reshape input for rotary embedding
    input_injection_flat = input_injection.view(
        batch_size * num_heads, config.seq_len, head_dim
    )
    cos_sin = rotary_emb(input_injection_flat)

    output = reasoner_module(hidden_state, input_injection, cos_sin)
    print(f"Input shape: {hidden_state.shape}")
    print(f"Input injection shape: {input_injection.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0,0,:4].detach().cpu().numpy()}")
    assert output.shape == (
        batch_size,
        config.seq_len,
        hidden_size,
    ), f"Unexpected shape: {output.shape}"
    # Verify ReasonerModule actually processes the input
    assert not torch.allclose(
        hidden_state, output, atol=1e-3
    ), "ReasonerModule should transform input"
    print("ReasonerModule test passed.")
