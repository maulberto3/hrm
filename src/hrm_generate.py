# import torch
# import torch.nn.functional as F


# def generate_reasoning_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
#     """
#     Autoregressive generation for HRM.
#     Uses model's output head for token prediction.
#     Paper: "Autoregressive generation for hierarchical reasoning model."
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     # Encode the prompt and move to device
#     input_ids = torch.tensor(tokenizer.encode(prompt).ids, device=device).unsqueeze(0)
#     batch_size = input_ids.size(0)
#     hidden_states = model.initial_hidden_states(batch_size)

#     with torch.no_grad():
#         for _ in range(max_length):
#             outputs = model(hidden_states, input_ids)
#             next_token_logits = outputs["output"][:, -1, :] / temperature
#             probs = F.softmax(next_token_logits, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
#             input_ids = torch.cat([input_ids, next_token], dim=1)
#             hidden_states = outputs["hidden_states"]
#             if next_token[0].item() == tokenizer.token_to_id("[eos]"):
#                 break

#     return tokenizer.decode(input_ids[0].tolist())
