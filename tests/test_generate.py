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
    get_random_text_beginning,
    generate_text_basic,
    # generate_text_advanced,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def test_generate_text_basic():
    """
    Test generate_text_basic for HRM with ACT using Alice in Wonderland prompt.
    Uses the real tokenizer from data.py for consistency.
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
        "max_length": 256,
        "cls_token": "[CLS]",
    }
    model_config = ModelConfig(**config)
    model = HierarchicalReasonerModel(model_config).to(device)

    # Use real tokenizer from data.py
    tokenizer, _ = get_tokenizer_and_text()

    # Use Alice prompt from generate.py
    prompt, book_name = get_random_text_beginning()

    # No need to pad/truncate or prefix CLS, generate_text_basic handles it
    output = generate_text_basic(model, tokenizer, prompt, config)
    assert isinstance(output, str), "Output should be a string"
    print("Input prompt:", prompt)
    print("Book name:", book_name)
    print("Generated output:", output)


# def test_generate_text_completion_basic():
#     """
#     Test generate_text_advanced for HRM with ACT.
#     Checks that generation runs and returns expected tuple.
#     """

#     config = ModelConfig(
#         seq_len=8,
#         vocab_size=20,
#         high_level_cycles=2,
#         low_level_cycles=2,
#         num_layers=2,
#         hidden_size=16,
#         num_heads=4,
#         expansion=2,
#         halt_max_steps=4,
#         halt_exploration_prob=0.1,
#         max_new_tokens=5,
#         temperature=1.0,
#         do_sample=True,
#         top_p=1.0,
#     )
#     model = HierarchicalReasonerModel(config).to(device)
#     tokenizer = DummyTokenizer(vocab_size=20)
#     prompt, completion, book_name, generated_text = generate_text_advanced(
#         model, tokenizer, prompt="1 2 3"
#     )
#     assert isinstance(prompt, str)
#     assert isinstance(completion, str)
#     assert isinstance(book_name, str)
#     assert isinstance(generated_text, str)
#     print("Prompt:", prompt)
#     print("Completion:", completion)
#     print("Book name:", book_name)
#     print("Generated text:", generated_text)
