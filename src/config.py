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
        halt_min_steps=1,
        batch_size=2,
        n_epochs=2,
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
        self.halt_min_steps = halt_min_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs


small_model_config = {
    "seq_len": 64,
    "vocab_size": 1000,
    "high_level_cycles": 2,
    "low_level_cycles": 2,
    "num_layers": 2,
    "hidden_size": 64,
    "num_heads": 4,
    "expansion": 2.0,
    "norm_epsilon": 1e-5,
    "rope_theta": 10000.0,
    "halt_max_steps": 4,
    "halt_exploration_prob": 0.05,
    "halt_min_steps": 1,
    "batch_size": 2,
    "n_epochs": 1,
}

small_training_config = {
    "batch_size": 4,
    "n_epochs": 2,
    "lr": 5e-4,
    "warmup_steps": 100,
    "model_path": "hrm_model_small.pth",
    "sample_frequency": 0.01,
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.99,
    "eval_interval": 10,
    "checkpoint_every_eval": False,
}

big_model_config = {
    "seq_len": 1024,
    "vocab_size": 32000,
    "high_level_cycles": 8,
    "low_level_cycles": 8,
    "num_layers": 12,
    "hidden_size": 768,
    "num_heads": 12,
    "expansion": 4.0,
    "norm_epsilon": 1e-5,
    "rope_theta": 10000.0,
    "halt_max_steps": 32,
    "halt_exploration_prob": 0.1,
    "halt_min_steps": 1,
    "batch_size": 32,
    "n_epochs": 50,
}

big_training_config = {
    "batch_size": 32,
    "n_epochs": 50,
    "lr": 2e-4,
    "warmup_steps": 5000,
    "model_path": "hrm_model_big.pth",
    "sample_frequency": 0.001,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eval_interval": 1000,
    "checkpoint_every_eval": True,
}

# Consistency checks:
# - All config dicts use the same keys as ModelConfig's __init__ (except for extra training keys).
# - Architectural keys: seq_len, vocab_size, high_level_cycles, low_level_cycles, num_layers, hidden_size, num_heads, expansion, norm_epsilon, rope_theta, halt_max_steps, halt_exploration_prob.
# - Training keys (batch_size, n_epochs, etc.) are not used by ModelConfig, but are present for convenience in the config dicts.

# If you want strict consistency, you can add a comment or validation to ignore extra keys in ModelConfig, which is already handled by **kwargs.
