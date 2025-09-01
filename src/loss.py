import torch
import torch.nn as nn

# Logging setup is kept but not used in most functions; prefer print for compact logs in tests.
# If you want to use logging, uncomment logger lines below.
import logging


logger = logging.getLogger("hrm_utils")


def compute_act_loss(outputs_list, targets, config):
    """
    Compute ACT loss combining sequence-to-sequence loss and Q-learning loss.

    Args:
        outputs_list: List of model outputs from each ACT step
        targets: Target token sequences
        config: Model configuration containing halt_max_steps and other parameters.
                At the final step (>= config.halt_max_steps), forces Q-continue = Q-halt
                to ensure the model learns to halt when it can't continue further.
    """
    max_halt_steps = config["halt_max_steps"]
    total_loss = 0
    for m, outputs in enumerate(outputs_list):
        # Sequence-to-sequence loss
        seq_loss = nn.CrossEntropyLoss()(
            outputs["output"].reshape(-1, outputs["output"].shape[-1]),
            targets.contiguous().reshape(-1),
        )

        # G_halt logic: reward for correct prediction
        predictions = torch.argmax(outputs["output"], dim=-1)
        correct_predictions = (predictions == targets).float().mean(dim=1)
        g_halt = correct_predictions

        # Use model-provided Q-continue target if present
        if "target_q_continue" in outputs:
            g_continue = outputs["target_q_continue"]
        else:
            # Fallback: recompute as before
            if m == (len(outputs_list) - 1) or m >= (max_halt_steps - 1):
                g_continue = g_halt
            else:
                next_outputs = (
                    outputs_list[m + 1] if m + 1 < len(outputs_list) else outputs
                )
                g_continue = torch.max(
                    torch.sigmoid(next_outputs["q_halt"]),
                    torch.sigmoid(next_outputs["q_continue"]),
                )

        # Q-learning loss
        q_targets = torch.stack([g_halt, g_continue], dim=1)
        q_predictions = torch.stack([outputs["q_halt"], outputs["q_continue"]], dim=1)
        q_predictions = torch.sigmoid(q_predictions)
        q_loss = nn.BCELoss()(q_predictions, q_targets)

        # Combine losses
        total_loss += seq_loss + q_loss
    return total_loss / len(outputs_list)
