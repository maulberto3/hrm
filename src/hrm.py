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

    def forward(self, inputs):
        """
        Abstracted ACT wrapper: runs the inner HRM model for up to halt_max_steps (from config), halting adaptively.
        All ACT arguments and hidden state initialization are handled internally. User only provides inputs.
        Output logits from self.inner exclude the first token (CLS): shape [batch_size, seq_len-1, vocab_size].
        """
        # First, we extract necessary variables
        training = self.training
        batch_size = inputs.size(0)
        device = inputs.device
        max_steps = self.config.halt_max_steps
        exploration_prob = self.config.halt_exploration_prob if training else 0.0
        halt_min_steps = self.config.halt_min_steps

        # Then, we initialize tracking tensors and variables
        steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        outputs_list = []
        current_hidden_states = self.initial_hidden_states(
            batch_size, self.config.seq_len, device
        )
        current_inputs = inputs

        # And we start the ACT (Adaptive Computation Time) loop
        # where we iteratively refine the hidden states and outputs
        # In training, this involves computing Q-values network learning
        # and
        # In eval, we focus on exploitation, i.e. selecting the action with the highest Q-value
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
            # Continue reasoning even if halt reached, to encourage exploration
            # In eval, no forced exploration, just follow best Q-values
            must_continue = (steps < random_min_steps) & random_explore

            # Target Q-value computation (for Q-learning, only in training mode)
            if training and step < (max_steps - 1):
                next_hidden_states = outputs["hidden_states"]
                # Compute target Q-values for the next step by:
                # asking the model a new inner reasoning forward pass over same inputs
                # but with the fresh new hidden states (out from previous reasoning)
                # In Q-learning theory, this would be akin to bootstrapping
                # where we use the next state's Q-values to get the target for current state
                # Although we don't have a fixed separate target network for this task,
                # we can still compute the target Q-values using the next hidden states
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

            # Check if any sequence has halted or reached max steps
            # If so, we need to reset their hidden states, because
            # we want to ensure that each sequence starts fresh
            reset_flag = halted | (steps >= max_steps)
            if reset_flag.any():
                initial_states = self.initial_hidden_states(
                    batch_size, self.config.seq_len, device
                )
                # Avoid in-place operations that break gradient computation
                new_hidden_states = {}
                for k in current_hidden_states:
                    new_hidden_states[k] = current_hidden_states[k].clone()
                    for i in range(batch_size):
                        if reset_flag[i]:
                            new_hidden_states[k][i] = initial_states[k][i]
                current_hidden_states = new_hidden_states

            # Halting logic (with exploration only in training)
            should_halt_batch = torch.tensor(
                [
                    self.inner.should_halt(
                        outputs["q_halt"][i],
                        outputs["q_continue"][i],
                        step,
                        halt_min_steps,
                        max_steps,
                    )
                    for i in range(batch_size)
                ],
                device=device,
                dtype=torch.bool,
            )
            # Halting logic is applied, but exploration can override it.
            should_halt_batch = should_halt_batch & (~must_continue)
            halted = halted | should_halt_batch

            # Update steps
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
