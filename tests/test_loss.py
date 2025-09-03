import torch
from loss import compute_act_loss


def test_compute_act_loss_basic():
    """
    Test compute_act_loss with dummy ACT outputs and targets.
    """
    print("\n=== TEST: test_compute_act_loss_basic ===")
    batch_size, seq_len, vocab_size, act_steps = 2, 3, 5, 4
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Targets shape: {targets.shape}")
    outputs_list = []
    for i in range(act_steps):
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        q_halt = torch.randn(batch_size, requires_grad=True)
        q_continue = torch.randn(batch_size, requires_grad=True)
        outputs = {
            "output": logits,
            "q_halt": q_halt,
            "q_continue": q_continue,
        }
        outputs_list.append(outputs)
        print(
            f"Step {i}: logits shape {logits.shape}, q_halt shape {q_halt.shape}, q_continue shape {q_continue.shape}"
        )

    # Minimal config for ACT loss
    config = {
        "seq_len": 3,
        "vocab_size": 5,
        "halt_max_steps": 4,
        "cls_token": "[cls]",
    }

    loss = compute_act_loss(outputs_list, targets, config)
    print(f"Computed ACT loss: {loss.item()}")
    assert loss.item() > 0
    loss.backward()
    print("ACT loss test passed.")
