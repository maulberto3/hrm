import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from hrm import HierarchicalReasonerModel
from config import ModelConfig
from data import get_tokenizer_and_text
from generate import (
    get_random_text_small,
    # get_random_text_big,
    generate_text_basic,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def test_generate_text_basic():
    """
    Test generate_text_basic for HRM with ACT using Alice in Wonderland prompt.
    Uses the real tokenizer from data.py for consistency.
    Includes detailed logging of all generation steps.
    """
    config = {
        "seq_len": 32,
        "vocab_size": 1000,  # Match config from data.py
        "high_level_cycles": 2,
        "low_level_cycles": 2,
        "num_layers": 2,
        "hidden_size": 64,
        "num_heads": 4,
        "expansion": 4,
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.1,
        "norm_epsilon": 0.25,
        "rope_theta": 500.0,
        "halt_min_steps": 4,
        "max_new_tokens": 96,
        "temperature": 0.5,
        "do_sample": True,
        "top_p": 0.995,
        "max_length": 5,  # Reduced for detailed logging
        "cls_token": "[CLS]",
        "pad_token": "[PAD]",
    }
    model_config = ModelConfig(**config)
    model = HierarchicalReasonerModel(model_config).to(device)

    # Use real tokenizer from data.py
    tokenizer, _ = get_tokenizer_and_text()

    # Test both small and big prompts to see truncation logic
    print("\n" + "=" * 80)
    print("TESTING SMALL PROMPT")
    print("=" * 80)
    prompt_small, book_name = get_random_text_small()
    _test_prompt_with_logging(model, tokenizer, prompt_small, config, "SMALL")

    # print("\n" + "=" * 80)
    # print("TESTING BIG PROMPT")
    # print("=" * 80)
    # prompt_big, book_name = get_random_text_big()
    # _test_prompt_with_logging(model, tokenizer, prompt_big, config, "BIG")


def _test_prompt_with_logging(model, tokenizer, prompt, config, test_name):
    """Helper function to test a prompt with detailed logging"""

    print(f"\n🔵 {test_name} PROMPT TEST")
    print(f"Original prompt: '{prompt}'")
    print(f"Original prompt length (chars): {len(prompt)}")

    # Show tokenization steps
    cls_token = config["cls_token"]
    pad_token = config["pad_token"]
    seq_len = model.config.seq_len

    print(f"\n📝 TOKENIZATION STEPS:")
    print(f"cls_token: '{cls_token}'")
    print(f"pad_token: '{pad_token}'")
    print(f"seq_len: {seq_len}")

    # Check if prompt starts with cls_token
    if not prompt.strip().startswith(cls_token):
        prompt_with_cls = f"{cls_token} {prompt}"
        print(f"❌ Prompt doesn't start with cls_token")
        print(f"Prompt with cls: '{prompt_with_cls}'")
    else:
        prompt_with_cls = prompt
        print(f"✅ Prompt already starts with cls_token")

    # Tokenize
    ids = tokenizer.encode(prompt_with_cls).ids
    print(f"Tokenized ids: {ids}")
    print(f"Tokenized length: {len(ids)}")

    # Show padding/truncation logic
    print(f"\n🔄 PADDING/TRUNCATION LOGIC:")
    if len(ids) < seq_len:
        pad_token_id = tokenizer.token_to_id(pad_token)
        padding_needed = seq_len - len(ids)
        ids_padded = ids + [pad_token_id] * padding_needed
        print(f"✅ Padding needed: {padding_needed} tokens")
        print(f"Padded ids: {ids_padded}")
    elif len(ids) == seq_len:
        ids_padded = ids
        print(f"✅ Perfect fit, no padding/truncation needed")
    else:
        cls_token_id = tokenizer.token_to_id(cls_token)
        ids_truncated = [cls_token_id] + ids[1:seq_len]
        print(f"❌ Truncation needed: {len(ids)} -> {seq_len}")
        print(f"Original ids: {ids}")
        print(f"Truncated ids: {ids_truncated}")
        ids_padded = ids_truncated

    # Show initial actual_tokens
    input_ids_list = ids_padded
    pad_token_id = tokenizer.token_to_id(pad_token)
    last_non_pad_idx = max(i for i, t in enumerate(input_ids_list) if t != pad_token_id)
    actual_tokens_list = input_ids_list[: last_non_pad_idx + 1]

    print(f"\n🎯 ACTUAL TOKENS TRACKING:")
    print(f"Initial input_ids: {input_ids_list}")
    print(f"Last non-pad index: {last_non_pad_idx}")
    print(f"Initial actual_tokens: {actual_tokens_list}")

    # Now call the actual generation function
    print(f"\n🚀 STARTING GENERATION:")
    output = generate_text_basic(model, tokenizer, prompt, config, verbose=True)

    print(f"\n📤 FINAL RESULTS:")
    print(f"Generated output: '{output}'")
    print(f"Output length (chars): {len(output)}")

    assert isinstance(output, str), "Output should be a string"
