import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from hrm_building_blocks import Embedding, Linear, RotaryEmbedding, Attention, SwiGLU

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_embedding_basic():
    """
    Test Embedding layer - maps token IDs to dense vectors.
    Data flow: token_ids (int) -> Embedding -> embedded_vectors (float)
    HRM Context: Converts discrete token IDs into continuous vector representations,
                 serving as the initial input to the model.
    """
    vocab_size = 10
    dim = 16
    batch_size = 4
    seq_len = 7

    embedding = Embedding(vocab_size, dim).to(device)
    # Input: token indices [batch_size, seq_len]
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Output: embeddings [batch_size, seq_len, dim]
    output = embedding(input_ids)

    assert output.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Unexpected shape: {output.shape}"
    print("Embedding test passed. Output shape:", output.shape)


def test_linear_basic():
    """
    Test Linear layer - applies a linear transformation.
    Data flow: input_features -> Linear -> output_features
    HRM Context: Used for various projections:
                 - QKV projections in Attention (query, key, value)
                 - Output head (hidden_size -> vocab_size)
                 - Q-ACT head (hidden_size -> 2, halt/continue)
                 - Gate/Up/Down projections in SwiGLU
    """
    in_dim = 16
    out_dim = 32
    batch_size = 4
    seq_len = 7

    linear = Linear(in_dim, out_dim).to(device)
    # Input: [batch_size, seq_len, in_dim]
    input_tensor = torch.randn(batch_size, seq_len, in_dim, device=device)
    # Output: [batch_size, seq_len, out_dim]
    output = linear(input_tensor)

    assert output.shape == (
        batch_size,
        seq_len,
        out_dim,
    ), f"Unexpected shape: {output.shape}"
    print("Linear test passed. Output shape:", output.shape)


def test_rotary_embedding_basic():
    """
    Test RotaryEmbedding layer - provides positional information.
    Data flow: query/key tensors -> RotaryEmbedding -> position-encoded tensors
    HRM Context:
    - Applied to query and key tensors within the Attention layer.
    - Encodes positional information into the attention mechanism.
    """
    dim = 64  # head_dim in attention
    max_length = 128
    batch_size = 4
    seq_len = 16
    num_heads = 8

    rotary_emb = RotaryEmbedding(dim, max_length).to(device)
    # Input: [batch_size, seq_len, num_heads, head_dim]
    input_tensor = torch.randn(batch_size, seq_len, num_heads, dim, device=device)
    # Output: [batch_size, seq_len, num_heads, head_dim]
    output = rotary_emb(input_tensor)

    assert output.shape == input_tensor.shape, f"Unexpected shape: {output.shape}"
    # Verify that rotary embedding actually changes the tensor
    assert not torch.allclose(
        input_tensor, output
    ), "RotaryEmbedding should modify input"
    print("RotaryEmbedding test passed. Output shape:", output.shape)


def test_attention_basic():
    """
    Test Attention layer - enables information flow between tokens.
    Data flow: input -> QKV projection -> attention -> output projection
    HRM Context:
    - Core of the ReasoningBlock, allowing tokens to attend to each other.
    - Uses RotaryEmbedding for position encoding.
    - Used in both high-level and low-level reasoners.
    """
    dim = 64
    head_dim = 16
    num_heads = 4
    batch_size = 2
    seq_len = 8

    attention = Attention(dim, head_dim, num_heads).to(device)
    rotary_emb = RotaryEmbedding(head_dim, max_length=128).to(device)

    # Input: [batch_size, seq_len, dim]
    input_tensor = torch.randn(batch_size, seq_len, dim, device=device)
    # Output: [batch_size, seq_len, dim]
    output = attention(input_tensor, rotary_emb=rotary_emb)

    assert output.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Unexpected shape: {output.shape}"
    # Verify attention actually processes the input
    assert not torch.allclose(
        input_tensor, output, atol=1e-3
    ), "Attention should transform input"
    print("Attention test passed. Output shape:", output.shape)


def test_swiglu_basic():
    """
    Test SwiGLU layer - applies a Swish-Gated Linear Unit activation.
    Data flow: input -> gate_proj + up_proj -> SwiGLU -> down_proj -> output
    HRM Context:
    - Activation function within the ReasoningBlock's MLP.
    - Introduces non-linearity for more complex feature transformations.
    """
    dim = 64
    expansion = 4.0
    batch_size = 2
    seq_len = 8

    swiglu = SwiGLU(dim, expansion).to(device)
    # Input: [batch_size, seq_len, dim]
    input_tensor = torch.randn(batch_size, seq_len, dim, device=device)
    # Output: [batch_size, seq_len, dim]
    output = swiglu(input_tensor)

    assert output.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Unexpected shape: {output.shape}"
    # Verify SwiGLU actually processes the input
    assert not torch.allclose(
        input_tensor, output, atol=1e-3
    ), "SwiGLU should transform input"
    print("SwiGLU test passed. Output shape:", output.shape)


if __name__ == "__main__":
    test_embedding_basic()
    test_linear_basic()
    test_rotary_embedding_basic()
    test_attention_basic()
    test_swiglu_basic()

"""
Overall HRM data flow:
1. token_ids -> Embedding -> embedded_tokens
2. embedded_tokens + cls_token -> input_embeddings
3. For each reasoning cycle:
   - low_level_state = ReasonerModule(low_level_state, high_level_state + input_embeddings)
   - high_level_state = ReasonerModule(high_level_state, low_level_state)
4. high_level_state -> output_head -> logits
5. high_level_state[:, 0] -> q_act_head -> halt/continue decisions

Where ReasonerModule contains:
- Multiple ReasoningBlocks, each with:
  - Attention layer (using Linear for QKV projections + RotaryEmbedding for position info)
  - SwiGLU MLP (using Linear for gate/up/down projections)
  - RMS normalization
"""
