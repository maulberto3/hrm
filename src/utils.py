# Create model name from config
def create_model_name(config):
    """Create a unique model name based on key config parameters."""
    name_parts = [
        f"hrm",
        f"seq{config['seq_len']}",
        f"vocab{config['vocab_size']}",
        f"h{config['hidden_size']}",
        f"heads{config['num_heads']}",
        f"layers{config['num_layers']}",
        f"hcyc{config['high_level_cycles']}",
        f"lcyc{config['low_level_cycles']}",
        f"halt{config['halt_max_steps']}",
    ]
    return "_".join(name_parts)
