import torch
import torch.nn as nn
import logging

# --- Logger setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hrm_utils")


def init_hidden_states(model, batch, device):
    """Initialize hidden states for a batch."""
    # logger.info(f"Initializing hidden states for batch of shape {batch.shape}")
    batch_size = batch.size(0)
    seq_len = batch.size(1)
    return model.initial_hidden_states(
        batch_size=batch_size, seq_len=seq_len, device=device
    )


def compute_act_loss(outputs_list, targets, max_halt_steps):
    """
    Compute ACT loss combining sequence-to-sequence loss and Q-learning loss.
    Uses max_halt_steps as the upper bound for ACT steps, matching raw_hrm.py behavior.
    """
    total_loss = 0

    for m, outputs in enumerate(outputs_list):
        # Sequence-to-sequence loss
        seq_loss = nn.CrossEntropyLoss()(
            outputs["output"].reshape(-1, outputs["output"].shape[-1]),
            targets.reshape(-1),
        )

        # Q-learning targets
        predictions = torch.argmax(outputs["output"], dim=-1)
        correct_predictions = (predictions == targets).float().mean(dim=1)

        # G_halt = 1{y_hat_m = y} (binary reward for correct prediction)
        g_halt = correct_predictions

        # Use max_halt_steps for G_continue computation
        if m == len(outputs_list) - 1 or m >= max_halt_steps - 1:
            # Last deliberation or max reached: G_continue = Q_halt of next (same as halt)
            g_continue = g_halt
        else:
            # G_continue = max(Q_halt_{m+1}, Q_continue_{m+1})
            next_outputs = outputs_list[m + 1] if m + 1 < len(outputs_list) else outputs
            g_continue = torch.max(next_outputs["q_halt"], next_outputs["q_continue"])

        # Q-learning loss using binary cross entropy
        q_targets = torch.stack([g_halt, g_continue], dim=1)
        q_predictions = torch.stack([outputs["q_halt"], outputs["q_continue"]], dim=1)
        q_loss = nn.BCELoss()(q_predictions, q_targets)

        # Combined ACT loss
        total_loss += seq_loss + q_loss

    return total_loss / len(outputs_list)


def run_act_forward(model, input_x, device, max_halt_steps):
    """Run ACT segments and collect outputs for a batch, using max_halt_steps as the upper bound."""
    # logger.info(
    #     f"Running ACT forward for input shape {input_x.shape} and max_halt_steps={max_halt_steps}"
    # )
    outputs_list = []
    active_batch_indices = torch.arange(input_x.size(0), device=device)
    current_x = input_x
    for step in range(max_halt_steps):
        if len(active_batch_indices) == 0:
            break
        # logger.info(f"Segment {deliberation}: current_x shape {current_x.shape}")
        hidden_states = init_hidden_states(model, current_x, device)
        outputs = model(
            hidden_states,
            current_x,
            step,
            max_halt_steps=max_halt_steps,
        )
        # logger.info(
        #     f"Segment {deliberation}: outputs['output'] shape {outputs['output'].shape}"
        # )
        outputs_list.append(outputs)
        if "should_halt" in dir(model):
            halt_decisions = []
            for j in range(current_x.size(0)):
                should_halt = model.should_halt(
                    outputs["q_halt"][j],
                    outputs["q_continue"][j],
                    step,
                    1,
                    max_halt_steps,
                )
                halt_decisions.append(should_halt)
            halt_mask = torch.tensor(halt_decisions, device=device)
            continue_mask = ~halt_mask
            if continue_mask.any() and (step < max_halt_steps - 1):
                active_batch_indices = active_batch_indices[continue_mask]
                current_x = current_x[continue_mask]
            else:
                # logger.info(f"Halting ACT at deliberation {deliberation}")
                break
    return outputs_list


def run_standard_forward(model, input_x, device):
    """Run a standard forward pass for a batch."""
    # logger.info(f"Running standard forward for input shape {input_x.shape}")
    hidden_states = init_hidden_states(model, input_x, device)
    out = model(hidden_states, input_x)
    return out


def train_one_batch(
    model,
    batch,
    optimizer,
    tokenizer,
    device,
    use_act=False,
    max_halt_steps=1,
):
    # logger.info(f"Training one batch: batch shape {batch.shape}, use_act={use_act}")
    batch = batch.to(device)
    # As it includes CLS via dataloader and hrm does not return the cls output
    input_x = batch[:, :-1]

    if use_act:
        outputs_list = run_act_forward(model, input_x, device, max_halt_steps)
        targets = batch[:, 2:]
        loss = compute_act_loss(outputs_list, targets, max_halt_steps)
    else:
        out = run_standard_forward(model, input_x, device)
        logits = out["output"]
        # Next-token prediction targets, skip cls and first token
        targets = batch[:, 2:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tokenizer.token_to_id("[pad]"),
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # logger.info(f"Batch training complete. Loss: {loss:.4f}")
    return loss.item(), model


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    tokenizer,
    device,
    use_act=False,
    max_halt_steps=1,
    logger=None,
    quick_run=False,
):
    # logger.info(
    #     f"Starting training epoch. use_act={use_act}, max_halt_steps={max_halt_steps}"
    # )
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        loss, model = train_one_batch(
            model,
            batch,
            optimizer,
            tokenizer,
            device,
            use_act=use_act,
            max_halt_steps=max_halt_steps,
        )
        epoch_loss += loss
        # if logger:
        #     logger.info(f"Epoch {i} Loss: {loss:.4f}")
        if quick_run and (i > 5):
            if logger:
                logger.info("Quick run finished")
            break
    avg_loss = epoch_loss / (i + 1)
    logger.info(f"Epoch complete. Avg loss: {avg_loss:.4f}")
    return avg_loss, model


def train_model(
    model,
    dataloader,
    optimizer,
    tokenizer,
    device,
    epochs,
    use_act=False,
    max_halt_steps=1,
    logger=None,
    quick_run=False,
):
    logger.info(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        avg_loss, model = train_one_epoch(
            model,
            dataloader,
            optimizer,
            tokenizer,
            device,
            use_act=use_act,
            max_halt_steps=max_halt_steps,
            logger=logger,
            quick_run=quick_run,
        )
        if logger:
            logger.info(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
    logger.info("Training complete.")
    return avg_loss, model
