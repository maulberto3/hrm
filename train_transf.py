import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os

def train_transf(tokenizer, model_config=None, training_config=None):
	default_model_config = {
		"num_layers": 2,
		"num_heads": 2,
		"num_kv_heads": 2,
		"hidden_dim": 128,
		"max_seq_len": 128,
		"vocab_size": len(tokenizer.get_vocab()),
		"dropout": 0.1,
	}
	default_training_config = {
		"batch_size": 32,
		"n_epochs": 2,
		"lr": 0.0005,
		"warmup_steps": 2000,
		"clip_norm": 6.0,
		"model_path": "textgen_model.pth",
		"sample_frequency": 0.0005,
	}
	model_config = {**default_model_config, **(model_config or {})}
	training_config = {**default_training_config, **(training_config or {})}
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = SimpleTextGenerationModel(**model_config).to(device)
	use_mask = True
	text = "\n".join(get_dataset_text())
	dataset = GutenbergDataset(text, tokenizer, seq_len=model_config["max_seq_len"])
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_config["batch_size"], shuffle=True)
	if os.path.exists(training_config["model_path"]):
		model.load_state_dict(torch.load(training_config["model_path"]))
		print(f"Loaded existing model from {training_config['model_path']}")
		return model
	optimizer = optim.AdamW(model.parameters(), lr=training_config["lr"])
	loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[pad]"))
	warmup_scheduler = optim.lr_scheduler.LinearLR(
		optimizer, start_factor=0.01, end_factor=1.0, total_iters=training_config["warmup_steps"])
	cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
		optimizer, T_max=training_config["n_epochs"] * len(dataloader) - training_config["warmup_steps"], eta_min=0)
	scheduler = optim.lr_scheduler.SequentialLR(
		optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[training_config["warmup_steps"]])
	print(f"Training simple model for {training_config['n_epochs']} epochs")
	print(f"Steps per epoch: {len(dataloader)}")
	best_loss = float('inf')
	for epoch in range(training_config["n_epochs"]):
		model.train()
		epoch_loss = 0
		progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{training_config['n_epochs']}")
		for x, y in progress_bar:
			x = x.to(device)
			y = y.to(device)
			optimizer.zero_grad()
			if use_mask:
				mask = create_causal_mask(x.shape[1], device)
				logits = model(x, mask.unsqueeze(0))
			else:
				logits = model(x)
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
					generated = generate_text(model, tokenizer, prompt)
					print(f"\nPrompt: {prompt}")
					print(f"Generated: {generated}")
					print("-" * 80)
		avg_loss = epoch_loss / len(dataloader)
		print(f"Epoch {epoch+1}/{training_config['n_epochs']}; Avg loss: {avg_loss:.4f}")
		if avg_loss < best_loss:
			best_loss = avg_loss
			torch.save(model.state_dict(), training_config["model_path"])
	return model
