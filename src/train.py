import torch
import torch.nn as nn
from loss import compute_act_loss
import logging

logger = logging.getLogger("__name__")


def train_model(
    model,
    dataloader,
    optimizer,
    device,
    epochs=1,
    max_halt_steps=1,
    logger=None,
    testing=False,
):
    """
    Complete training function for n epochs.
    If testing=True, only trains on first 5 batches per epoch.
    """
    print(f"Starting training for {epochs} epochs")
    if testing:
        print("Testing mode: Only using first 5 batches per epoch")

    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        batch_count = 0

        for i, batch in enumerate(dataloader):
            # Testing condition: only first 5 batches
            if testing and i >= 5:
                print(f"\nTesting mode: stopping at batch {i}")
                break

            # Merged batch training logic directly into epoch
            batch = batch.to(device)
            input_x = batch[:, :-1]
            targets = batch[:, 2:]

            outputs_list = model(input_x)
            loss = compute_act_loss(outputs_list, targets, max_halt_steps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            print(f"Batch {i} loss: {loss.item():.4f}", end=" ")

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"\nEpoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")
        if logger:
            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

    print("Training complete.")
    if logger:
        logger.info("Training complete.")
    return avg_loss, model
