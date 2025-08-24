import torch
import torch.nn as nn

# Logging setup is kept but not used in most functions; prefer print for compact logs in tests.
# If you want to use logging, uncomment logger lines below.
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hrm_utils")


def compute_act_loss(outputs_list, targets, max_halt_steps):
    """
    Compute ACT loss combining sequence-to-sequence loss and Q-learning loss.
    Uses max_halt_steps as the upper bound for ACT steps, matching raw_hrm.py behavior.
    """
    # Comment: Improved clarity and added comments for each step.
    total_loss = 0
    for m, outputs in enumerate(outputs_list):
        # Sequence-to-sequence loss
        seq_loss = nn.CrossEntropyLoss()(
            outputs["output"].reshape(-1, outputs["output"].shape[-1]),
            targets.reshape(-1),
        )
        # Q-learning targets: reward for correct prediction
        predictions = torch.argmax(outputs["output"], dim=-1)
        correct_predictions = (predictions == targets).float().mean(dim=1)
        g_halt = correct_predictions
        # G_continue logic
        if m == len(outputs_list) - 1 or m >= max_halt_steps - 1:
            g_continue = g_halt
        else:
            next_outputs = outputs_list[m + 1] if m + 1 < len(outputs_list) else outputs
            g_continue = torch.max(next_outputs["q_halt"], next_outputs["q_continue"])
        # Q-learning loss
        q_targets = torch.stack([g_halt, g_continue], dim=1)
        q_predictions = torch.stack([outputs["q_halt"], outputs["q_continue"]], dim=1)
        q_loss = nn.BCELoss()(q_predictions, q_targets)
        # Combine losses
        total_loss += seq_loss + q_loss
    return total_loss / len(outputs_list)


def train_one_batch(
    model,
    batch,
    optimizer,
    tokenizer,
    device,
    use_act=False,
    max_halt_steps=1,
):
    batch = batch.to(device)
    input_x = batch[:, :-1]
    targets = batch[:, 2:]

    if use_act:
        outputs_list = model(input_x)
        loss = compute_act_loss(outputs_list, targets, max_halt_steps)
    else:
        out = model(input_x)
        logits = out[0]["output"] if isinstance(out, list) else out["output"]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tokenizer.token_to_id("[pad]"),
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Batch loss: {loss.item()}", end=" ")
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
        print(f"Epoch batch {i} loss: {loss}", end=" ")
        if quick_run and (i > 5):
            if logger:
                logger.info("Quick run finished")
            break
    avg_loss = epoch_loss / (i + 1)
    print(f"\nEpoch complete. Avg loss: {avg_loss:.4f}")
    if logger:
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
    print(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch}")
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
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        if logger:
            logger.info(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
    print("Training complete.")
    if logger:
        logger.info("Training complete.")
    return avg_loss, model
