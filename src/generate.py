import torch
import torch.nn.functional as F


def get_random_text_small():
    """
    Get a small hardcoded text beginning from Alice in Wonderland.
    Uses a consistent, clean prompt for testing text generation.
    Returns the text snippet and the book name.
    """
    # Simple hardcoded prompt from Alice in Wonderland
    prompt = "CHAPTER I.\nDown the Rabbit-Hole\nAlice was beginning to get very tired of sitting by her sister on the bank, "
    book_name = "alice_in_wonderland"
    return prompt.strip(), book_name


def get_random_text_big():
    """
    Get a small hardcoded text beginning from Alice in Wonderland.
    Uses a consistent, clean prompt for testing text generation.
    Returns the text snippet and the book name.
    """
    # Simple hardcoded prompt from Alice in Wonderland
    prompt = "CHAPTER I.\nDown the Rabbit-Hole\nAlice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”\nSo she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her."
    book_name = "alice_in_wonderland"
    return prompt.strip(), book_name


def generate_text_basic(model, tokenizer, prompt, config, verbose=False):
    """
    Original autoregressive generation for HRM with ACT.
    Uses model's ACT wrapper for adaptive halting.
    """
    model.eval()
    device = next(model.parameters()).device
    seq_len = model.config.seq_len
    pad_token = config["pad_token"]
    pad_token_id = tokenizer.token_to_id(pad_token)

    # Always ensure prompt starts with cls_token (for ACT logic)
    cls_token = config["cls_token"]
    cls_token_id = tokenizer.token_to_id(cls_token)

    if not prompt.strip().startswith(cls_token):
        prompt_with_cls = f"{cls_token} {prompt}"
    else:
        prompt_with_cls = prompt

    # Tokenize
    ids = tokenizer.encode(prompt_with_cls).ids

    # Pad if needed
    if len(ids) < seq_len:
        ids = ids + [pad_token_id] * (seq_len - len(ids))

    # Truncate if too long (consistent with training)
    if len(ids) > seq_len:
        # Keep cls token at front, truncate from the end
        ids = [cls_token_id] + ids[1:seq_len]

    # Convert tokens to tensor
    input_ids = torch.tensor(ids, device=device).unsqueeze(0)

    # Track actual tokens separately from padded input
    input_ids_list = input_ids[0].tolist()
    last_non_pad_idx = max(i for i, t in enumerate(input_ids_list) if t != pad_token_id)
    actual_tokens = input_ids[:, : last_non_pad_idx + 1]

    if verbose:
        print(f"\n🔄 GENERATION LOOP (max_length={model.config.max_length}):")

    for step in range(model.config.max_length):
        # Prepare model input: always seq_len size
        if actual_tokens.size(1) < seq_len:
            # Pad to seq_len for model input
            padding_needed = seq_len - actual_tokens.size(1)
            pad_tensor = torch.full((1, padding_needed), pad_token_id, device=device)
            model_input = torch.cat([actual_tokens, pad_tensor], dim=1)
            if verbose:
                print(
                    f"\n Step {step}: actual_tokens({actual_tokens.size(1)}) < seq_len({seq_len}) -> PADDING {padding_needed}"
                )
                print(f"    actual_tokens: {actual_tokens[0].tolist()}")
                print(f"    model_input: {model_input[0].tolist()}")
        elif actual_tokens.size(1) == seq_len:
            model_input = actual_tokens
            if verbose:
                print(
                    f"\n  Step {step}: actual_tokens({actual_tokens.size(1)}) == seq_len({seq_len}) -> NO CHANGE"
                )
                print(f"\n    model_input: {model_input[0].tolist()}")
        else:
            # Once actual_tokens > seq_len, truncate from right to left
            # But also keep cls_token at front, take last (seq_len-1) tokens
            remaining_tokens = actual_tokens[:, -(seq_len - 1) :]
            cls_tensor = torch.tensor([[cls_token_id]], device=device)
            model_input = torch.cat([cls_tensor, remaining_tokens], dim=1)
            if verbose:
                print(
                    f"  Step {step}: actual_tokens({actual_tokens.size(1)}) > seq_len({seq_len}) -> TRUNCATE"
                )
                print(f"    actual_tokens: {actual_tokens[0].tolist()}")
                print(f"    remaining_tokens: {remaining_tokens[0].tolist()}")
                print(f"    model_input: {model_input[0].tolist()}")

        # Forward pass through ACT wrapper
        outputs_list = model(model_input)
        final_outputs = outputs_list[-1]  # Use last ACT step

        # Get the new token prediction
        next_token_logits = final_outputs["output"][:, -1, :] / model.config.temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append new token only to actual_tokens (no padding)
        actual_tokens = torch.cat([actual_tokens, next_token], dim=1)

        if verbose:
            next_token_id = next_token[0].item()
            try:
                next_token_str = tokenizer.decode([next_token_id])
            except:
                next_token_str = f"<UNK:{next_token_id}>"
            print(f"    next_token: {next_token_id} -> '{next_token_str}'")
            print(f"    updated actual_tokens: {actual_tokens[0].tolist()}")

        # End if max_length allowed or if "[eos]" token predicted
        if next_token[0].item() == tokenizer.token_to_id("[eos]"):
            if verbose:
                print(f"    🛑 EOS token detected, stopping generation")
            break

    # Decode only actual generated tokens (no pads)
    return tokenizer.decode(actual_tokens[0].tolist())
