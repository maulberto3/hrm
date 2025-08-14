import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from hrm import HierarchicalReasonerModel, ModelConfig

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_hierarchical_reasoner_model_basic():
    """
    Test HierarchicalReasonerModel - the full HRM model.
    Data flow: inputs -> Embedding -> input_embeddings -> reasoning cycles -> output_logits
    HRM Context:
    - Combines all components to perform hierarchical reasoning.
    - Tests the overall forward pass and shape consistency.
    - Output shapes should match expectations from both Python and Swift reference implementations.
    """
    config = ModelConfig(
        seq_len=32,
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
    seq_len = 32

    model = HierarchicalReasonerModel(config).to(device)
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    hidden_states = model.initial_hidden_states(batch_size)
    outputs = model(hidden_states, inputs)

    # Output should contain logits for each token (excluding CLS token)
    assert "output" in outputs, "Output should contain 'output' key"
    assert "hidden_states" in outputs, "Output should contain 'hidden_states' key"
    # output shape: [batch_size, seq_len, vocab_size] (since CLS token is removed)
    assert outputs["output"].shape == (
        batch_size,
        seq_len,
        config.vocab_size,
    ), f"Unexpected output shape: {outputs['output'].shape}"
    # hidden_states shapes: [batch_size, seq_len+1, hidden_size] (including CLS token)
    assert (
        "high_level" in outputs["hidden_states"]
    ), "Hidden states should contain 'high_level'"
    assert (
        "low_level" in outputs["hidden_states"]
    ), "Hidden states should contain 'low_level'"
    assert outputs["hidden_states"]["high_level"].shape == (
        batch_size,
        seq_len + 1,
        config.hidden_size,
    ), f"Unexpected high_level shape: {outputs['hidden_states']['high_level'].shape}"
    assert outputs["hidden_states"]["low_level"].shape == (
        batch_size,
        seq_len + 1,
        config.hidden_size,
    ), f"Unexpected low_level shape: {outputs['hidden_states']['low_level'].shape}"

    print(
        "HierarchicalReasonerModel test passed. Output shape:", outputs["output"].shape
    )


if __name__ == "__main__":
    test_hierarchical_reasoner_model_basic()
