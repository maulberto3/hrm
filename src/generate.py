import torch
import torch.nn.functional as F


def get_random_text_beginning():
    """
    Get a simple hardcoded text beginning from Alice in Wonderland.
    Uses a consistent, clean prompt for testing text generation.
    Returns the text snippet and the book name.
    """
    # Simple hardcoded prompt from Alice in Wonderland
    prompt = "CHAPTER I.\nDown the Rabbit-Hole\n\nAlice was beginning to get very tired of sitting by her sister on the bank, "
    book_name = "alice_in_wonderland"

    return prompt.strip(), book_name


def generate_text_basic(model, tokenizer, prompt, config):
    """
    Original autoregressive generation for HRM with ACT.
    Uses model's ACT wrapper for adaptive halting.
    """
    model.eval()
    device = next(model.parameters()).device

    # Always prefix prompt with CLS token from config if available
    cls_token = config["cls_token"]
    if not prompt.strip().startswith(cls_token):
        prompt = f"{cls_token} {prompt}"

    ids = tokenizer.encode(prompt).ids
    seq_len = model.config.seq_len
    pad_token_id = tokenizer.token_to_id("[pad]")

    # Ensure to pad/truncate to seq_len
    if len(ids) < seq_len:
        ids = ids + [pad_token_id] * (seq_len - len(ids))
    else:
        # It's ok to, at this stage, to truncate from left to right...
        ids = ids[:seq_len]
    input_ids = torch.tensor(ids, device=device).unsqueeze(0)

    generated = input_ids
    for _ in range(model.config.max_length):
        # Ensure input to model is always exactly seq_len
        model_input = generated
        if model_input.size(1) > seq_len:
            # ...but then at chat time we need to truncate from right to left
            model_input = model_input[:, -seq_len:]

        # Forward pass through ACT wrapper
        outputs_list = model(model_input)
        final_outputs = outputs_list[-1]  # Use last ACT step

        next_token_logits = final_outputs["output"][:, -1, :] / model.config.temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token[0].item() == tokenizer.token_to_id("[eos]"):
            break

    return tokenizer.decode(generated[0].tolist())


# def generate_text_advanced(
#     model,
#     tokenizer,
#     prompt=None,
# ):
#     """
#     Generate text completion using the HRM model with ACT.
#     All generation parameters are taken from config.
#     If no prompt is provided, uses a hardcoded Alice in Wonderland beginning.
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     # Extract generation parameters from config
#     max_new_tokens = model.config.max_new_tokens
#     temperature = model.config.temperature
#     do_sample = model.config.do_sample
#     top_p = model.config.top_p

#     # Get prompt
#     if prompt is None:
#         # Use model config if not provided
#         if config is None:
#             config = model.config
#         prompt, book_name = get_random_text_beginning(config)
#         print(f"Random prompt from '{book_name}':")
#         print(f"Prompt: {prompt}")
#         print("=" * 50)
#     else:
#         book_name = "user_provided"

#     # Tokenize the prompt
#     try:
#         encoded = tokenizer.encode(prompt)
#         input_ids = torch.tensor(encoded.ids, device=device).unsqueeze(0)
#     except Exception as e:
#         print(f"Tokenization error: {e}")
#         # Fallback to simple prompt
#         encoded = tokenizer.encode("The")
#         input_ids = torch.tensor(encoded.ids, device=device).unsqueeze(0)
#         prompt = "The"

#     print(f"Input shape: {input_ids.shape}")
#     generated_ids = input_ids.clone()

#     with torch.no_grad():
#         for step in range(max_new_tokens):
#             try:
#                 # Use the model's forward method (ACT wrapper)
#                 outputs_list = model(generated_ids)

#                 # Get the last output from ACT iterations
#                 final_outputs = outputs_list[-1]
#                 next_token_logits = final_outputs["output"][
#                     :, -1, :
#                 ]  # Last token predictions

#                 # Apply temperature
#                 if temperature > 0:
#                     next_token_logits = next_token_logits / temperature

#                 # Apply top-p sampling if enabled
#                 if do_sample and top_p < 1.0:
#                     sorted_logits, sorted_indices = torch.sort(
#                         next_token_logits, descending=True
#                     )
#                     cumulative_probs = torch.cumsum(
#                         F.softmax(sorted_logits, dim=-1), dim=-1
#                     )

#                     # Remove tokens with cumulative probability above the threshold
#                     sorted_indices_to_remove = cumulative_probs > top_p
#                     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
#                         ..., :-1
#                     ].clone()
#                     sorted_indices_to_remove[..., 0] = 0

#                     indices_to_remove = sorted_indices_to_remove.scatter(
#                         1, sorted_indices, sorted_indices_to_remove
#                     )
#                     next_token_logits[indices_to_remove] = float("-inf")

#                 # Sample or take argmax
#                 if do_sample and temperature > 0:
#                     probs = F.softmax(next_token_logits, dim=-1)
#                     next_token = torch.multinomial(probs, num_samples=1)
#                 else:
#                     next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

#                 # Append new token
#                 generated_ids = torch.cat([generated_ids, next_token], dim=1)

#                 # Check for EOS token
#                 if tokenizer.token_to_id("[eos]") is not None and next_token[
#                     0
#                 ].item() == tokenizer.token_to_id("[eos]"):
#                     break

#                 # Check for end of sequence length limit
#                 if generated_ids.size(1) >= model.config.seq_len:
#                     break

#             except Exception as e:
#                 print(f"Generation error at step {step}: {e}")
#                 break

#     # Decode the generated text
#     try:
#         generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
#         completion = tokenizer.decode(
#             generated_ids[0][input_ids.size(1) :].cpu().tolist()
#         )
#     except Exception as e:
#         print(f"Decoding error: {e}")
#         return prompt, "", book_name

#     return prompt, completion, book_name, generated_text
