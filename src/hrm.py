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
        # Output heads
        self.output_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.q_learning_head = Linear(config.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_learning_head.linear.weight.zero_()
            self.q_learning_head.linear.bias.fill_(-5)
        # Reasoner modules
        self.high_level_reasoner = ReasonerModule(config)
        self.low_level_reasoner = ReasonerModule(config)

    def initial_hidden_states(self, batch_size, seq_len, device):
        """
        Returns initial hidden states z_H0, z_L0 for a batch, directly initialized to [batch_size, seq_len, hidden_size].
        Paper: "The initial hidden states z_0 are initialized by sampling from a truncated normal distribution..."
        """
        # Directly sample new tensors for each batch and sequence position
        high_level = trunc_normal_init_(
            torch.empty(batch_size, seq_len, self.config.hidden_size, device=device),
            std=1,
        )
        low_level = trunc_normal_init_(
            torch.empty(batch_size, seq_len, self.config.hidden_size, device=device),
            std=1,
        )
        return {
            "high_level": high_level,
            "low_level": low_level,
        }

    def encode_inputs(self, inputs):
        """
        Encode input tokens and prepend CLS token.
        Paper: "First, the input x is projected into a working representation x̃ by the input network..."
        """
        input_emb = self.input_embedding(inputs) * math.sqrt(self.config.hidden_size)
        return input_emb

    def forward(self, hidden_states, inputs, segment=0, max_segments=None):
        """
        Implements the HRM forward pass with ACT support:
        - Input embedding
        - Hierarchical recurrent cycles (low-level and high-level)
        - One-step gradient approximation (final step with grad, rest with torch.no_grad)
        - Output head and Q-head
        Paper: ACT implementation with Q-learning for adaptive halting
        """
        input_emb = self.encode_inputs(inputs)
        low_level_state = hidden_states["low_level"]
        high_level_state = hidden_states["high_level"]

        # Run reasoning cycles (temporal separation: high-level and low-level)
        for cycle in range(
            self.config.high_level_cycles * self.config.low_level_cycles - 1
        ):
            # Each cycle is a "segment" of reasoning
            low_level_state = self.low_level_reasoner(
                low_level_state,
                high_level_state + input_emb,
                rotary_emb=self.rotary_emb,
            )
            if (cycle + 1) % self.config.low_level_cycles == 0:
                high_level_state = self.high_level_reasoner(
                    high_level_state, low_level_state, rotary_emb=self.rotary_emb
                )

        # Final step with gradients (one-step gradient approximation)
        low_level_state = low_level_state.detach()
        high_level_state = high_level_state.detach()

        # One-step gradient approximation i.e. final loop
        low_level_state = self.low_level_reasoner(
            low_level_state, high_level_state + input_emb, rotary_emb=self.rotary_emb
        )
        high_level_state = self.high_level_reasoner(
            high_level_state, low_level_state, rotary_emb=self.rotary_emb
        )

        # Each segment produces its own prediction and Q-values
        # To experiment, toggle the following line:
        # output_logits = self.output_head(high_level_state[:, 1:])  # recommended: excludes CLS
        output_logits = self.output_head(high_level_state)  # includes CLS

        q_learning_logits = torch.sigmoid(self.q_learning_head(high_level_state[:, 0]))

        return {
            "hidden_states": {
                "high_level": high_level_state.detach(),
                "low_level": low_level_state.detach(),
            },
            "output": output_logits,
            "q_halt": q_learning_logits[:, 0],
            "q_continue": q_learning_logits[:, 1],
            "segment": segment,
        }

    def should_halt(self, q_halt, q_continue, segment, min_segments, max_segments):
        """
        ACT halting decision based on Q-values and segment constraints.
        """
        # The ACT mechanism adaptively decides how many segments to run
        if segment >= max_segments:
            return True
        if (segment >= min_segments) and (q_halt > q_continue):
            return True
        return False


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

# In machine learning terms, **segment** refers to a single "reasoning pass" or "iteration" of the model's hierarchical reasoning cycle.
# - In the HRM with ACT, the model can perform multiple segments per input, each segment representing a deeper or more deliberate reasoning step.
# - Each segment produces its own prediction and Q-values (halt/continue).
# - The ACT mechanism adaptively decides how many segments to run for each input, allowing the model to "think longer" for harder tasks and "halt early" for easier ones.
# - This is analogous to adaptive computation time, where the model dynamically chooses its "runtime" per input, similar to how humans may think quickly or slowly depending on task complexity.

# Yes, in HRM with ACT, the number of reasoning segments (loops) per input is **variable** and adaptively determined at runtime.
# - For each input, the model may perform a different number of reasoning cycles (segments), depending on the Q-head's halt/continue predictions and the ACT logic.
# - This means the computation time (number of passes through the reasoning loop) is **not fixed**—it can be longer for harder inputs and shorter for easier ones.
# - The adaptive halting mechanism allows the model to "think longer" when needed, just like humans do for complex tasks.
