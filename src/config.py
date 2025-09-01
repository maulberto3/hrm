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
        norm_epsilon,
        rope_theta,
        halt_max_steps,
        halt_exploration_prob,
        halt_min_steps,
        # Generation parameters
        max_new_tokens,
        temperature,
        do_sample,
        top_p,
        max_length,
        **kwargs
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
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.max_length = max_length


small_config = {
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
    "max_new_tokens": 100,
    "temperature": 0.8,
    "do_sample": True,
    "top_p": 0.9,
    "max_length": 64,
    "cls_token": "[CLS]",
    "pad_token": "[PAD]",
    "eos_token": "[EOS]",
    "batch_size": 2,
    "n_epochs": 1,
    "generate_every": 5,
    "lr": 1e-4,
    "checkpoint_dir": "/home/maulb/hrm/data/",
}

big_config = {
    "seq_len": 256,
    "vocab_size": 32000,
    "high_level_cycles": 8,
    "low_level_cycles": 8,
    "num_layers": 6,
    "hidden_size": 504,
    "num_heads": 12,
    "expansion": 4.0,
    "norm_epsilon": 1e-5,
    "rope_theta": 10000.0,
    "halt_max_steps": 32,
    "halt_exploration_prob": 0.1,
    "halt_min_steps": 1,
    "max_new_tokens": 256,
    "temperature": 0.8,
    "do_sample": True,
    "top_p": 0.9,
    "max_length": 1024,
    "cls_token": "[CLS]",
    "pad_token": "[PAD]",
    "eos_token": "[EOS]",
    "batch_size": 16,
    "n_epochs": 50,
    "generate_every": 15,
    "lr": 1e-4,
    "checkpoint_dir": "/home/maulb/hrm/data/",
}
