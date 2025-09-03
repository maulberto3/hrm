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

    Q-learning theory:
        - The Q-value update rule is:
            Q(s, a) ← Q(s, a) + α * [R + γ * max_a' Q(s', a') - Q(s, a)]
        - In this implementation:
            - Q(s, a) is predicted by the model (q_halt, q_continue)
            - R (reward) is g_halt (correct prediction reward)
            - max_a' Q(s', a') is g_continue (future Q-value, either from next step or forced to halt at last step)
            - α (learning rate) and γ (discount) are implicitly handled by optimizer and loss weighting
        - The loss encourages the model to predict Q-values that match the expected reward plus future value.
        - At the last step, g_continue is set to g_halt to force the model to learn to halt.
        - The BCELoss between predicted Q-values and targets implements the Q-learning update.
    """
    max_halt_steps = config["halt_max_steps"]
    total_loss = 0
    for m, outputs in enumerate(outputs_list):
        # --- Sequence-to-sequence loss ---
        # Standard next-token prediction loss (cross-entropy)
        seq_loss = nn.CrossEntropyLoss()(
            outputs["output"].reshape(-1, outputs["output"].shape[-1]),
            targets.contiguous().reshape(-1),
        )

        # --- Q-learning logic ---
        # Q(s, a) ← Q(s, a) + α * [R + γ * max_a' Q(s', a') - Q(s, a)]
        # In code:
        # - outputs["q_halt"], outputs["q_continue"]: model's Q-value predictions for halt/continue
        # - g_halt: reward for correct prediction (R)
        # - g_continue: future Q-value (max_a' Q(s', a'))

        # Compute reward for halting (g_halt): mean accuracy for this step
        predictions = torch.argmax(outputs["output"], dim=-1)
        correct_predictions = (predictions == targets).float().mean(dim=1)
        g_halt = correct_predictions

        # Compute target for continuing (g_continue):
        # - If model provides "target_q_continue", use it (for custom logic)
        # - Otherwise, at last step or max_halt_steps, force g_continue = g_halt (must halt)
        # - Else, use max of next step's predicted Q-values (future value)
        if "target_q_continue" in outputs:
            g_continue = outputs["target_q_continue"]
        else:
            if m == (len(outputs_list) - 1) or m >= (max_halt_steps - 1):
                # Last step: must halt, so future value = current reward
                g_continue = g_halt
            else:
                next_outputs = (
                    outputs_list[m + 1] if m + 1 < len(outputs_list) else outputs
                )
                # Take max Q-value for next step (future value)
                g_continue = torch.max(
                    torch.sigmoid(next_outputs["q_halt"]),
                    torch.sigmoid(next_outputs["q_continue"]),
                )

        # Stack targets and predictions for BCELoss
        # - q_targets: [g_halt, g_continue] (theory: [R, γ * max_a' Q(s', a')])
        # - q_predictions: [q_halt, q_continue] (model's predicted Q-values)
        q_targets = torch.stack([g_halt, g_continue], dim=1)
        q_predictions = torch.stack([outputs["q_halt"], outputs["q_continue"]], dim=1)
        q_predictions = torch.sigmoid(q_predictions)
        q_loss = nn.BCELoss()(q_predictions, q_targets)

        # --- Combine losses ---
        # Total loss = sequence loss + Q-learning loss
        total_loss += seq_loss + q_loss
    # Average over ACT steps
    return total_loss / len(outputs_list)
