from torch import nn
from hrm_building_blocks import Attention, SwiGLU, rms_norm


# --- Model Config ---
class ModelConfig:
    """
    Configuration for HRM, including all architectural hyperparameters.
    Paper: "Configuration for the hierarchical reasoning model."
    """

    def __init__(
        self,
        seq_len,
        vocab_size,
        high_level_cycles,
        low_level_cycles,
        num_layers,
        hidden_size,
        num_heads,
        expansion,
        norm_epsilon=1e-5,
        rope_theta=10000.0,
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        **kwargs,  # Accept and ignore extra keys
    ):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.high_level_cycles = high_level_cycles
        self.low_level_cycles = low_level_cycles
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.norm_epsilon = norm_epsilon
        self.rope_theta = rope_theta
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob


# --- Low-level and High-level Recurrent Modules (f_L, f_H) ---
class ReasoningBlock(nn.Module):
    """
    Transformer block (encoder-only, Llama-style) with attention, SwiGLU, RMSNorm.
    Used for both f_L (low-level) and f_H (high-level) modules.
    Paper: "Both low-level and high-level recurrent modules f_L and f_H are implemented using encoder-only Transformer blocks..."
    """

    def __init__(self, config):
        super().__init__()
        self.attention = Attention(
            dim=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
        )
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm_epsilon = config.norm_epsilon

    def forward(self, x, cos_sin):
        x = rms_norm(x + self.attention(x, cos_sin), epsilon=self.norm_epsilon)
        x = rms_norm(x + self.mlp(x), epsilon=self.norm_epsilon)
        return x


class ReasonerModule(nn.Module):
    """
    Stacked ReasoningBlocks: implements recurrent module f_L or f_H.
    Each module keeps its own hidden state (z_L, z_H).
    Element-wise addition merges multiple inputs (e.g., z_H + input_emb).
    Paper: "These modules take multiple inputs, and we use straightforward element-wise addition to combine them..."
    """

    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ReasoningBlock(config) for _ in range(config.num_layers)]
        )

    def forward(self, hidden_state, input_injection, cos_sin):
        # Element-wise addition: core HRM mechanism for combining inputs
        # Paper: "straightforward element-wise addition to combine them"
        x = hidden_state + input_injection

        # Process through reasoning blocks
        for i, block in enumerate(self.blocks):
            x = block(x, cos_sin)
        return x
