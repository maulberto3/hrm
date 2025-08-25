import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from hrm import HierarchicalReasonerModel
from config import ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_hrm_model_act_wrapper_train():
    """
    Essential test for HierarchicalReasonerModel ACT wrapper in training mode.
    Checks output list length, output shapes, keys, and Q-learning targets.
    """
    print("\n=== TEST: test_hrm_model_act_wrapper_train ===")
    config = ModelConfig(
        seq_len=16,
        vocab_size=100,
        high_level_cycles=2,
        low_level_cycles=2,
        num_layers=2,
        hidden_size=192,
        num_heads=4,
        expansion=4,
        halt_max_steps=4,
        halt_exploration_prob=0.1,
    )
    print("\nModel config:", config.__dict__)
    batch_size = 2
    seq_len = config.seq_len
    model = HierarchicalReasonerModel(config).to(device)
    model.train()

    # Dummy batch: [batch_size, seq_len] (CLS handled by dataloader in real data)
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    print("Model config:", config.__dict__)
    print("Input shape:", inputs.shape)

    outputs_list = model(inputs)

    assert isinstance(
        outputs_list, list
    ), "Model should return a list of outputs (one per ACT step)"
    assert len(outputs_list) > 0, "Outputs list should not be empty"
    for i, outputs in enumerate(outputs_list):
        print(f"\n--- ACT Step {i} ---")
        print(f"Step index: {outputs.get('step', i)}")
        print(f"Halt Q: {outputs['q_halt'].detach().cpu().numpy()}")
        print(f"Continue Q: {outputs['q_continue'].detach().cpu().numpy()}")
        print(f"Output shape: {outputs['output'].shape}")
        print(
            f"High-level hidden shape: {outputs['hidden_states']['high_level'].shape}"
        )
        print(f"Low-level hidden shape: {outputs['hidden_states']['low_level'].shape}")
        print(f"Target Q present: {'target_q_continue' in outputs}")
        if "target_q_continue" in outputs:
            print(f"Target Q shape: {outputs['target_q_continue'].shape}")
        assert "output" in outputs, "Each output should contain 'output' key"
        assert (
            "hidden_states" in outputs
        ), "Each output should contain 'hidden_states' key"
        # Output excludes CLS token: shape [batch_size, seq_len-1, vocab_size]
        assert outputs["output"].shape == (
            batch_size,
            seq_len - 1,
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
        assert "q_halt" in outputs, "Each output should contain 'q_halt' key"
        assert "q_continue" in outputs, "Each output should contain 'q_continue' key"
        # Check Q-learning target if present (should be present except last step)
        if i < len(outputs_list) - 1:
            assert (
                "target_q_continue" in outputs
            ), "Output should contain 'target_q_continue' for Q-learning"

        print(f"HRM ACT wrapper TRAIN test passed. ACT steps: {len(outputs_list)}")
