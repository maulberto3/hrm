# Hierarchical Reasoning Model (HRM) implementation
# This code implements the HRM as described in the paper:
# - Hierarchical processing: high-level (abstract, slow) and low-level (detailed, fast) modules
# - Temporal separation: high-level cycles (N), low-level cycles (T)
# - Recurrent connectivity: iterative refinement, feedback, and deep supervision
# - Adaptive computation time (ACT): Q-head for halt/continue decisions

import torch
from torch import nn
from config import ModelConfig
from hrm_inner import HRMInner


class HierarchicalReasonerModel(nn.Module):
    """
    HRM model with ACT wrapper, matching the structure of HierarchicalReasoningModel_ACTV1.
    Handles ACT loop, halting logic, exploration, Q-targets, per-sequence step tracking, and batch data reset for halted sequences.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.inner = HRMInner(config)
        self.training = True  # Track training/eval mode

    def train(self, mode: bool = True):
        super().train(mode)
        self.training = mode
        self.inner.train(mode)
        return self

    def eval(self):
        super().eval()
        self.training = False
        self.inner.eval()
        return self

    def initial_hidden_states(self, batch_size, seq_len, device):
        return self.inner.initial_hidden_states(batch_size, seq_len, device)

    def forward(
        self,
        hidden_states,
        inputs,
        halt_max_steps=None,
        halt_exploration_prob=0.0,
        min_halt_steps=1,
        training=None,
    ):
        """
        ACT wrapper: runs the inner HRM model for up to halt_max_steps, halting adaptively.
        Adds exploration, Q-target computation, per-sequence step tracking, and batch reset.
        """
        # Use explicit training flag if provided, else use self.training
        if training is None:
            training = self.training

        batch_size = inputs.size(0)
        device = inputs.device
        max_steps = halt_max_steps or self.config.halt_max_steps
        exploration_prob = halt_exploration_prob if training else 0.0

        steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        outputs_list = []
        current_hidden_states = hidden_states
        current_inputs = inputs

        for step in range(max_steps):
            outputs = self.inner(current_hidden_states, current_inputs)
            outputs["step"] = step
            outputs_list.append(outputs)

            # Exploration only in training mode
            random_explore = (
                torch.rand(batch_size, device=device) < exploration_prob
                if training
                else torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
            random_min_steps = (
                torch.randint(
                    low=2, high=max_steps + 1, size=(batch_size,), device=device
                )
                if training
                else torch.full((batch_size,), max_steps, device=device)
            )
            must_continue = (steps < random_min_steps) & random_explore

            # Target Q-value computation (for Q-learning, only in training mode)
            if training and step < max_steps - 1:
                next_hidden_states = outputs["hidden_states"]
                next_outputs = self.inner(next_hidden_states, current_inputs)
                target_q_continue = torch.sigmoid(
                    torch.where(
                        steps + 1 >= max_steps,
                        next_outputs["q_halt"],
                        torch.maximum(
                            next_outputs["q_halt"], next_outputs["q_continue"]
                        ),
                    )
                )
                outputs["target_q_continue"] = target_q_continue

            # Batch data reset for halted sequences
            reset_flag = halted | (steps >= max_steps)
            if reset_flag.any():
                initial_states = self.initial_hidden_states(
                    batch_size, self.config.seq_len, device
                )
                for i in range(batch_size):
                    if reset_flag[i]:
                        for k in current_hidden_states:
                            current_hidden_states[k][i] = initial_states[k][i]

            # Halting logic (with exploration only in training)
            should_halt_batch = torch.tensor(
                [
                    self.inner.should_halt(
                        outputs["q_halt"][i],
                        outputs["q_continue"][i],
                        step,
                        min_halt_steps,
                        max_steps,
                    )
                    for i in range(batch_size)
                ],
                device=device,
                dtype=torch.bool,
            )
            should_halt_batch = should_halt_batch & (~must_continue)
            halted = halted | should_halt_batch

            steps = steps + (~halted).int()

            if halted.all():
                break

            current_hidden_states = outputs["hidden_states"]

        return outputs_list


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
