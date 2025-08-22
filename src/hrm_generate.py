import torch
import torch.nn.functional as F


def generate_reasoning_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """
    Autoregressive generation for HRM with ACT.
    Uses model's ACT wrapper for adaptive halting.
    """
    model.eval()
    device = next(model.parameters()).device

    # Encode the prompt and move to device
    input_ids = torch.tensor(tokenizer.encode(prompt).ids, device=device).unsqueeze(0)
    batch_size = input_ids.size(0)
    hidden_states = model.initial_hidden_states(batch_size, input_ids.size(1), device)

    generated = input_ids
    for _ in range(max_length):
        # Forward pass through ACT wrapper (no exploration, min_halt_steps=1)
        outputs_list = model(
            hidden_states,
            generated,
            halt_max_steps=model.config.halt_max_steps,
            halt_exploration_prob=0.0,
            min_halt_steps=1,
        )
        final_outputs = outputs_list[-1]  # Use last ACT step

        next_token_logits = final_outputs["output"][:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        hidden_states = final_outputs["hidden_states"]

        if next_token[0].item() == tokenizer.token_to_id("[eos]"):
            break

    return tokenizer.decode(generated[0].tolist())
