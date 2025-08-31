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
    print("\n=== TEST: test_embedding_basic ===")

    vocab_size = 10
    dim = 16
    batch_size = 4
    seq_len = 7

    embedding = Embedding(vocab_size, dim).to(device)
    # Input: token indices [batch_size, seq_len]
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Output: embeddings [batch_size, seq_len, dim]
    output = embedding(input_ids)
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Unexpected shape: {output.shape}"
    print("Embedding test passed.")


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
    print("\n=== TEST: test_linear_basic ===")

    in_dim = 16
    out_dim = 32
    batch_size = 4
    seq_len = 7

    linear = Linear(in_dim, out_dim).to(device)
    # Input: [batch_size, seq_len, in_dim]
    input_tensor = torch.randn(batch_size, seq_len, in_dim, device=device)
    # Output: [batch_size, seq_len, out_dim]
    output = linear(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (
        batch_size,
        seq_len,
        out_dim,
    ), f"Unexpected shape: {output.shape}"
    print("Linear test passed.")


def test_rotary_embedding_basic():
    """
    Test RotaryEmbedding layer - provides positional information.
    Data flow: query/key tensors -> RotaryEmbedding -> position-encoded tensors
    HRM Context:
    - Applied to query and key tensors within the Attention layer.
    - Encodes positional information into the attention mechanism.
    """
    print("\n=== TEST: test_rotary_embedding_basic ===")
    # Illustrate multi-head attention reshaping for rotary embedding
    model_dim = 64  # Full model dimension
    num_heads = 4
    rotary_dim = model_dim // num_heads  # Dimension per head for rotary embedding
    batch_size = 4
    seq_len = 7

    # Input tensor for the full model (before splitting into heads)
    input_tensor_model = torch.randn(batch_size, seq_len, model_dim, device=device)

    # Reshape to [batch_size * num_heads, seq_len, rotary_dim] for rotary embedding
    input_tensor_rotary = input_tensor_model.view(
        batch_size * num_heads, seq_len, rotary_dim
    )

    rotary_emb = RotaryEmbedding(rotary_dim).to(device)
    output = rotary_emb(input_tensor_rotary)
    print(f"Original input shape (model_dim): {input_tensor_model.shape}")
    print(f"Input shape (rotary): {input_tensor_rotary.shape}")
    print(
        "NOTE: This reshaping is required because rotary embedding operates per head (rotary_dim), not over the full model dimension. If you use multi-head attention, you must reshape your input so rotary is applied per head. Forgetting this is a common source of bugs!"
    )
    print(f"Cos shape: {output[0].shape}, Sin shape: {output[1].shape}")
    print(
        f"Reshaped for rotary (batch_size * num_heads, seq_len, rotary_dim): {input_tensor_rotary.shape}"
    )
    assert isinstance(output, tuple), "RotaryEmbedding should return a tuple"
    assert len(output) == 2, "RotaryEmbedding should return a tuple of length 2"
    assert output[0].shape == (
        seq_len,
        rotary_dim,
    ), f"Unexpected shape for cos: {output[0].shape}"
    assert output[1].shape == (
        seq_len,
        rotary_dim,
    ), f"Unexpected shape for sin: {output[1].shape}"
    print("RotaryEmbedding test passed.")


def test_attention_basic():
    """
    Test Attention layer - enables information flow between tokens.
    """
    print("\n=== TEST: test_attention_basic ===")

    dim = 64
    head_dim = 16
    num_heads = 4
    batch_size = 2
    seq_len = 8

    rotary_emb = RotaryEmbedding(head_dim, max_length=128).to(device)
    input_tensor = torch.randn(batch_size, seq_len, dim, device=device)

    # Reshape input for rotary embedding
    input_tensor_flat = input_tensor.view(batch_size * num_heads, seq_len, head_dim)

    # Get cos and sin embeddings
    cos_sin = rotary_emb(input_tensor_flat)

    # Non-causal attention
    attention_noncausal = Attention(dim, head_dim, num_heads, causal=False).to(device)
    output_noncausal = attention_noncausal(input_tensor, cos_sin)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape (noncausal): {output_noncausal.shape}")
    assert output_noncausal.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Unexpected shape: {output_noncausal.shape}"
    assert not torch.allclose(
        input_tensor, output_noncausal, atol=1e-3
    ), "Attention should transform input"

    # Causal attention
    attention_causal = Attention(dim, head_dim, num_heads, causal=True).to(device)
    output_causal = attention_causal(input_tensor, cos_sin)
    print(f"Output shape (causal): {output_causal.shape}")
    assert output_causal.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Unexpected shape: {output_causal.shape}"
    assert not torch.allclose(
        input_tensor, output_causal, atol=1e-3
    ), "Attention should transform input"

    # Causal and non-causal outputs should differ
    assert not torch.allclose(
        output_noncausal, output_causal, atol=1e-3
    ), "Causal and non-causal attention outputs should differ"

    print("Attention test passed.")


def test_swiglu_basic():
    """
    Test SwiGLU layer - applies a Swish-Gated Linear Unit activation.
    Data flow: input -> gate_proj + up_proj -> SwiGLU -> down_proj -> output
    HRM Context:
    - Activation function within the ReasoningBlock's MLP.
    - Introduces non-linearity for more complex feature transformations.
    """
    print("\n=== TEST: test_swiglu_basic ===")

    dim = 64
    expansion = 4.0
    batch_size = 2
    seq_len = 8

    swiglu = SwiGLU(dim, expansion).to(device)
    # Input: [batch_size, seq_len, dim]
    input_tensor = torch.randn(batch_size, seq_len, dim, device=device)
    # Output: [batch_size, seq_len, dim]
    output = swiglu(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Unexpected shape: {output.shape}"
    # Verify SwiGLU actually processes the input
    assert not torch.allclose(
        input_tensor, output, atol=1e-3
    ), "SwiGLU should transform input"
    print("SwiGLU test passed.")


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
