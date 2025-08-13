import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os
from transf_hrm import HierarchicalReasonerModel, generate_reasoning_text

def train_transf_hrm(tokenizer, model_config=None, training_config=None):

	# ARC training config matching
	default_model_config = {
		"batch_size": 768,
		"seq_len": 128,  # adjust as needed
		"num_layers": 4,  # inferred from H_layers/L_layers
		"num_heads": 8,
		"num_kv_heads": 8,
		"hidden_size": 512,
		"max_seq_len": 128,
		"vocab_size": len(tokenizer.get_vocab()),
		"dropout": 0.1,
		"expansion": 4.0,
		"rms_norm_eps": 1e-5,
		"rope_theta": 10000.0,
		"forward_dtype": "bfloat16",
		"halt_max_steps": 16,
		"halt_exploration_prob": 0.1,
		"H_layers": 4,
		"L_layers": 4,
		"puzzle_emb_ndim": 512,
		"pos_encodings": "rope",
		"num_puzzle_identifiers": 0,  # set if using puzzle identifiers
	}
	default_training_config = {
		"batch_size": 768,
		"n_epochs": 100000,
		"lr": 1e-4,
		"warmup_steps": 2000,
		"clip_norm": 6.0,
		"model_path": "textgen_model.pth",
		"sample_frequency": 0.0005,
		"weight_decay": 0.1,
		"beta1": 0.9,
		"beta2": 0.95,
		"eval_interval": 10000,
		"checkpoint_every_eval": True,
		"puzzle_emb_lr": 1e-2,
		"puzzle_emb_weight_decay": 0.1,
	}
	model_config = {**default_model_config, **(model_config or {})}
	training_config = {**default_training_config, **(training_config or {})}
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Instantiate model using ModelConfig from transf_hrm.py
	from transf_hrm import ModelConfig
	model_cfg = ModelConfig(
		seq_len=model_config["max_seq_len"],
		vocab_size=model_config["vocab_size"],
		high_level_cycles=model_config.get("H_layers", 4),
		low_level_cycles=model_config.get("L_layers", 4),
		num_layers=model_config["num_layers"],
		hidden_size=model_config["hidden_size"],
		num_heads=model_config["num_heads"],
		expansion=model_config["expansion"],
		norm_epsilon=model_config["rms_norm_eps"],
		rope_theta=model_config["rope_theta"],
		halt_max_steps=model_config["halt_max_steps"],
		halt_exploration_prob=model_config["halt_exploration_prob"]
	)
	model = HierarchicalReasonerModel(model_cfg).to(device)
	if hasattr(model, "forward_dtype"):
		model = model.to(dtype=model.forward_dtype)
	model = model.to(device)
	text = "\n".join(get_dataset_text())
	dataset = GutenbergDataset(text, tokenizer, seq_len=model_config["max_seq_len"])
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_config["batch_size"], shuffle=True)
	if os.path.exists(training_config["model_path"]):
		model.load_state_dict(torch.load(training_config["model_path"]))
		print(f"Loaded existing model from {training_config['model_path']}")
		return model
	optimizer = optim.AdamW(
		model.parameters(),
		lr=training_config["lr"],
		betas=(training_config["beta1"], training_config["beta2"]),
		weight_decay=training_config["weight_decay"]
	)
	loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[pad]"))
	warmup_scheduler = optim.lr_scheduler.LinearLR(
		optimizer, start_factor=0.01, end_factor=1.0, total_iters=training_config["warmup_steps"])
	cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
		optimizer, T_max=training_config["n_epochs"] * len(dataloader) - training_config["warmup_steps"], eta_min=0)
	scheduler = optim.lr_scheduler.SequentialLR(
		optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[training_config["warmup_steps"]])
	print(f"Training enhanced model for {training_config['n_epochs']} epochs")
	print(f"Steps per epoch: {len(dataloader)}")
	best_loss = float('inf')
	# Use initial_hidden_states from model
	for epoch in range(training_config["n_epochs"]):
		model.train()
		epoch_loss = 0
		progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{training_config['n_epochs']}")
		for x, y in progress_bar:
			x = x.to(device)
			y = y.to(device)
			optimizer.zero_grad()
			batch_size = x.size(0)
			hidden_states = model.initial_hidden_states(batch_size)
			# Forward pass
			outputs = model(hidden_states, x)
			logits = outputs["output"]
			loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), training_config["clip_norm"], error_if_nonfinite=True)
			optimizer.step()
			scheduler.step()
			epoch_loss += loss.item()
			progress_bar.set_postfix(loss=loss.item())
			if random() < training_config["sample_frequency"]:
				test_prompts = [
					"Once upon a time,",
					"We the people of the",
					"In the beginning was the",
				]
				print("\nGenerating sample texts:")
				for prompt in test_prompts:
					generated = generate_reasoning_text(model, tokenizer, prompt)
					print(f"\nPrompt: {prompt}")
					print(f"Generated: {generated}")
					print("-" * 80)
		avg_loss = epoch_loss / len(dataloader)
		print(f"Epoch {epoch+1}/{training_config['n_epochs']}; Avg loss: {avg_loss:.4f}")
		# Periodic evaluation and checkpointing
		if training_config.get("checkpoint_every_eval") and (epoch % training_config["eval_interval"] == 0):
			torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
		if avg_loss < best_loss:
			best_loss = avg_loss
			torch.save(model.state_dict(), training_config["model_path"])
	return model
