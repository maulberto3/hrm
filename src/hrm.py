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
from hrm_reasoner import ModelConfig, ReasonerModule


# --- Hierarchical Reasoning Model (HRM) ---
class HierarchicalReasonerModel(nn.Module):
    """
    Full HRM model: combines input network, low-level and high-level recurrent modules, output network, and ACT/Q-head.
    Implements hierarchical processing, temporal separation, and recurrent connectivity.
    Paper: "The HRM model consists of four learnable components: an input network f_I, a low-level recurrent module f_L, a high-level recurrent module f_H, and an output network f_O."
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(
            trunc_normal_init_(
                torch.empty(config.hidden_size), std=1.0 / math.sqrt(config.hidden_size)
            )
        )
        self.input_embedding = Embedding(
            config.vocab_size,
            config.hidden_size,
            init_std=1.0 / math.sqrt(config.hidden_size),
        )
        self.output_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.q_act_head = Linear(config.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_act_head.linear.weight.zero_()
            self.q_act_head.linear.bias.fill_(-5)
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_heads,
            config.seq_len + 1,
            config.rope_theta,
        )
        self.high_level_reasoner = ReasonerModule(config)
        self.low_level_reasoner = ReasonerModule(config)
        self.register_buffer(
            "H_init", trunc_normal_init_(torch.empty(config.hidden_size), std=1)
        )
        self.register_buffer(
            "L_init", trunc_normal_init_(torch.empty(config.hidden_size), std=1)
        )

    def initial_hidden_states(self, batch_size):
        """
        Returns initial hidden states z_H0, z_L0 for a batch.
        Paper: "The initial hidden states z_0 are initialized by sampling from a truncated normal distribution..."
        """
        device = self.cls_token.device
        return {
            "high_level": self.H_init.unsqueeze(0).repeat(batch_size, 1).to(device),
            "low_level": self.L_init.unsqueeze(0).repeat(batch_size, 1).to(device),
        }

    def encode_inputs(self, inputs):
        """
        Encode input tokens and prepend CLS token.
        Paper: "First, the input x is projected into a working representation x̃ by the input network..."
        """
        batch_size = inputs.size(0)
        input_emb = torch.cat(
            [
                self.cls_token.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1),
                self.input_embedding(inputs),
            ],
            dim=1,
        ) * math.sqrt(self.config.hidden_size)
        return input_emb

    def forward(self, hidden_states, inputs):
        """
        Implements the HRM forward pass:
        - Input embedding
        - Hierarchical recurrent cycles (low-level and high-level)
        - One-step gradient approximation (final step with grad, rest with torch.no_grad)
        - Output head and Q-head
        Paper: "At each timestep i, the L-module updates its state conditioned on its own previous state, the H-module's current state (which remains fixed throughout the cycle), and the input representation..."
        Paper: "The H-module only updates once per cycle (i.e., every T timesteps) using the L-module's final state at the end of that cycle..."
        Paper: "After N full cycles, a prediction ŷ is extracted from the hidden state of the H-module..."
        Paper: "A halting mechanism (detailed later in this section) determines whether the model should terminate, in which case ŷ will be used as the final prediction, or continue with an additional forward pass."
        Paper: "Deep supervision: multiple forward passes (segments), each with detached hidden state."
        Paper: "One-step gradient approximation: only backpropagate through final states, not full sequence."
        """
        input_emb = self.encode_inputs(inputs)
        high_level_state = hidden_states["high_level"]
        low_level_state = hidden_states["low_level"]

        # Fix: expand high_level_state to match input_emb shape
        _, seq_len_plus1, _ = input_emb.shape

        # Run reasoning cycles
        for cycle in range(
            self.config.high_level_cycles * self.config.low_level_cycles - 1
        ):
            # Only expand if high_level_state is 2D, ensure correct seq_len
            if high_level_state.dim() == 2:
                hl_expanded = high_level_state.unsqueeze(1).expand(
                    -1, seq_len_plus1, -1
                )
            elif high_level_state.shape[1] != seq_len_plus1:
                # If 3D but wrong seq_len, fix it
                hl_expanded = high_level_state[:, :1, :].expand(-1, seq_len_plus1, -1)
            else:
                hl_expanded = high_level_state
            low_level_state = self.low_level_reasoner(
                low_level_state,
                hl_expanded + input_emb,
                rotary_emb=self.rotary_emb,
            )
            if (cycle + 1) % self.config.low_level_cycles == 0:
                high_level_state = self.high_level_reasoner(
                    high_level_state, low_level_state, rotary_emb=self.rotary_emb
                )

        # Final step
        low_level_state = low_level_state.detach()
        high_level_state = high_level_state.detach()
        # Only expand if high_level_state is 2D, ensure correct seq_len
        if high_level_state.dim() == 2:
            hl_expanded = high_level_state.unsqueeze(1).expand(-1, seq_len_plus1, -1)
        elif high_level_state.shape[1] != seq_len_plus1:
            # If 3D but wrong seq_len, fix it
            hl_expanded = high_level_state[:, :1, :].expand(-1, seq_len_plus1, -1)
        else:
            hl_expanded = high_level_state
        low_level_state = self.low_level_reasoner(
            low_level_state,
            hl_expanded + input_emb,
            rotary_emb=self.rotary_emb,
        )
        high_level_state = self.high_level_reasoner(
            high_level_state, low_level_state, rotary_emb=self.rotary_emb
        )

        output_logits = self.output_head(high_level_state[:, 1:])
        q_act_logits = self.q_act_head(high_level_state[:, 0])
        return {
            "hidden_states": {
                "high_level": high_level_state.detach(),
                "low_level": low_level_state.detach(),
            },
            "output": output_logits,
            "q_act_halt": q_act_logits[:, 0],
            "q_act_continue": q_act_logits[:, 1],
        }


# --- Paper Reference ---
"""
This implementation follows the HRM architecture as described in the paper:
- Hierarchical processing: high-level and low-level modules, each with their own recurrent state.
- Temporal separation: distinct update frequencies for high-level (slow) and low-level (fast) modules.
- Recurrent connectivity: iterative refinement, deep supervision, and one-step gradient approximation.
- Adaptive computation time: Q-head predicts halt/continue, enabling dynamic reasoning depth.
- All transformer blocks use modern LLM enhancements: RoPE, SwiGLU, RMSNorm, bias-free linear layers, truncated normal initialization.

Key equations and mechanisms:
- x̃ = f_I(x; θ_I)
- z_L_i = f_L(z_L_{i-1}, z_H_{i-1}, x̃; θ_L)
- z_H_i = f_H(z_H_{i-1}, z_L_{i-1}; θ_H) (every T steps)
- ŷ = f_O(z_H_{N*T}; θ_O)
- Q-head: Q_hat = (Q_halt, Q_continue) for ACT
- Deep supervision: detach hidden state between segments
- One-step gradient: only backprop through final states
"""
