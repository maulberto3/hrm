# Hierarchical Reasoning Model (HRM) implementation
# This code implements the HRM as described in the paper:
# - Hierarchical processing: high-level (abstract, slow) and low-level (detailed, fast) modules
# - Temporal separation: high-level cycles (N), low-level cycles (T)
# - Recurrent connectivity: iterative refinement, feedback, and deep supervision
# - Adaptive computation time (ACT): Q-head for halt/continue decisions

import math
import torch
from torch import nn
from hrm_building_blocks import Embedding, trunc_normal_init_, Linear, RotaryEmbedding
from hrm_reasoner import ReasonerModule
from config import ModelConfig


# --- Hierarchical Reasoning Model (HRM) ---
class HRMInner(nn.Module):
    """
    Inner HRM model: contains the core reasoning logic with dynamic halting (ACT).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.input_embedding = Embedding(
            config.vocab_size,
            config.hidden_size,
            init_std=1.0 / math.sqrt(config.hidden_size),
        )

        # Positional embedding (Rotary)
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_heads,
            config.seq_len,
            config.rope_theta,
        )

        # Output head
        self.output_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = Linear(config.hidden_size, 2, bias=True)

        # Reasoner modules
        self.high_level_reasoner = ReasonerModule(config)
        self.low_level_reasoner = ReasonerModule(config)

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size), std=1)
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size), std=1)
        )

        # Q head special init (like in hrm_act_v1.py)
        with torch.no_grad():
            self.q_head.linear.weight.zero_()
            self.q_head.linear.bias.fill_(-5)

    def encode_inputs(self, inputs):
        """
        Encode input tokens.
        """
        # Convert input tokens to embeddings
        input_emb = self.input_embedding(inputs) * math.sqrt(self.config.hidden_size)
        return input_emb

    def should_halt(self, q_halt, q_continue, step, halt_min_steps, halt_max_steps):
        """
        ACT halting decision based on Q-values and step constraints.
        """
        # Check if we are at the maximum step, if yes, then halt
        # i.e. we don't have enough information to continue
        if step >= (halt_max_steps - 1):  # Use -1 to match 0-indexed steps
            return True
        # Check if we are past the minimum step and halt > continue
        if (step >= halt_min_steps) and (q_halt > q_continue):
            return True
        # Otherwise, it just means that we are still uncertain
        # i.e. we should continue reasoning
        return False

    def initial_hidden_states(self, batch_size, seq_len, device):
        high_level = (
            self.H_init.unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, seq_len, 1)
            .to(device)
        )
        low_level = (
            self.L_init.unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, seq_len, 1)
            .to(device)
        )
        return {
            "high_level": high_level,
            "low_level": low_level,
        }

    def forward(
        self,
        hidden_states,
        inputs,
    ):
        """
        Inner HRM forward pass with dynamic halting (ACT).
        """
        input_emb = self.encode_inputs(inputs)
        z_L = hidden_states["low_level"]
        z_H = hidden_states["high_level"]

        batch_size, seq_len, _ = z_H.shape

        # Prepare rotary embeddings for attention
        head_dim = self.config.hidden_size // self.config.num_heads
        num_heads = self.config.num_heads

        # Reshape input_emb for rotary embedding
        input_emb_flat = input_emb.view(batch_size * num_heads, seq_len, head_dim)
        cos_sin = self.rotary_emb(input_emb_flat)

        # Forward iterations with no gradients (except last step)
        with torch.no_grad():
            for _H_step in range(self.config.high_level_cycles):
                for _L_step in range(self.config.low_level_cycles):
                    # Only update z_L if not last H/L step
                    if not (
                        (_H_step == self.config.high_level_cycles - 1)
                        and (_L_step == self.config.low_level_cycles - 1)
                    ):
                        z_L = self.low_level_reasoner(
                            z_L, z_H + input_emb, cos_sin=cos_sin
                        )
                # Only update z_H if not last H step
                if not (_H_step == self.config.high_level_cycles - 1):
                    z_H = self.high_level_reasoner(z_H, z_L, cos_sin=cos_sin)

        # Final step with gradients (1-step grad approximation)
        z_L = self.low_level_reasoner(z_L, z_H + input_emb, cos_sin=cos_sin)
        z_H = self.high_level_reasoner(z_H, z_L, cos_sin=cos_sin)

        # Generate outputs: exclude first token (CLS)
        output_logits = self.output_head(z_H)[:, 1:]

        # Q head: use first token for Q-values
        q_logits = self.q_head(z_H[:, 0])
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]

        return {
            "output": output_logits,
            "hidden_states": {
                "high_level": z_H.detach(),
                "low_level": z_L.detach(),
            },
            "q_halt": q_halt,
            "q_continue": q_continue,
        }
