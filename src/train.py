import torch
import os
from loss import compute_act_loss
import logging
from utils import create_model_name
from generate import generate_text_basic, get_random_text_beginning


logger = logging.getLogger("__name__")


def train_model(model, dataloader, optimizer, device, config, testing):
    """
    Complete training function for n epochs using config parameters.
    Gets epochs, max_halt_steps, and testing mode from config.
    """
    epochs = config["n_epochs"]
    avg_loss = 0.0

    logger.info(f"Starting training for {epochs} epochs")
    if testing:
        logger.info("Testing mode: Only using first 5 batches per epoch")

    model.train()

    generate_every = config["generate_every"]
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_name = create_model_name(config)
    logger.info(f"Model name: {model_name}")

    # Check for existing checkpoint (single file per config)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")

    if os.path.exists(checkpoint_path):
        logger.info(f"Found existing checkpoint: {checkpoint_path}")
        logger.info(f"Resuming training from saved model state")

        # Load the checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.info("No existing checkpoint found. Starting fresh training.")
    try:
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            batch_count = 0

            for i, batch in enumerate(dataloader):
                # Testing condition: only first 5 batches
                if testing and i >= 5:
                    logger.info(f"Testing mode: stopping at batch {i}")
                    break

                batch = batch.to(device)
                input_x = batch[:, :-1]
                targets = batch[:, 2:]

                outputs_list = model(input_x)
                loss = compute_act_loss(outputs_list, targets, config)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                # Print to console immediately, separate from logger
                print(f"{loss.item():.4f}", end=" ", flush=True)

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            logger.info(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

            # Generate sample output every generate_every epochs
            if (epoch + 1) % generate_every == 0:
                logger.info("--- Sample Generation ---")
                try:
                    # Use Alice prompt for generation
                    prompt, _ = get_random_text_beginning()
                    # Assume tokenizer is available in config
                    tokenizer = config["tokenizer"]
                    gen_text = generate_text_basic(model, tokenizer, prompt, config)
                    logger.info(f"Prompt: {prompt}")
                    logger.info(f"Generated: {gen_text}")
                    logger.info(f"Sample generation at epoch {epoch + 1}: {gen_text}")
                except Exception as e:
                    logger.info(f"Generation error: {e}")
                    logger.error(f"Generation error at epoch {epoch + 1}: {e}")

            # Save checkpoint every few epochs (optional)
            save_every = config.get("save_every", 10)
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
                logger.info(f"Checkpoint saved to {checkpoint_path}")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected! Saving model checkpoint...")
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
        logger.info(f"Model checkpoint saved to {checkpoint_path}")

    finally:
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
        logger.info(f"Model checkpoint saved to {checkpoint_path}")

    logger.info("Training complete.")
    return avg_loss, model
