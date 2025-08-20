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
    "batch_size": 32,  # Added for run.py
    "n_epochs": 50,  # Added for run.py
}

big_training_config = {
    "batch_size": 32,
    "n_epochs": 50,
    "lr": 2e-4,
    "warmup_steps": 5000,
    "clip_norm": 6.0,
    "model_path": "hrm_model_big.pth",
    "sample_frequency": 0.001,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eval_interval": 1000,
    "checkpoint_every_eval": True,
    "max_segments": 8,
    "halt_max_steps": 32,
    "halt_exploration_prob": 0.1,
}
