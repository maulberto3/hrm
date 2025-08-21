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
from hrm_inner import HRMInner


class HierarchicalReasonerModel(nn.Module):
    """
    Full HRM model: combines input network, low-level and high-level recurrent modules, output network.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.inner = HRMInner(config)

    def initial_hidden_states(self, batch_size, seq_len, device):
        """
        Returns initial hidden states z_H0, z_L0 for a batch.
        """
        return self.inner.initial_hidden_states(batch_size, seq_len, device)

    def forward(self, hidden_states, inputs, **kwargs):
        """
        Implements the HRM forward pass.
        """
        outputs = self.inner(hidden_states, inputs, **kwargs)
        return outputs


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
# Yes, in HRM with ACT, the number of reasoning segments (loops) per input is **variable** and adaptively determined at runtime.
# - For each input, the model may perform a different number of reasoning cycles (segments), depending on the Q-head's halt/continue predictions and the ACT logic.
# - This means the computation time (number of passes through the reasoning loop) is **not fixed**—it can be longer for harder inputs and shorter for easier ones.
# - The adaptive halting mechanism allows the model to "think longer" when needed, just like humans do for complex tasks.
