import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import time
import argparse
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path

def bytes_to_gb(bytes):
	return bytes / (1024 ** 3)

def mean_pooling(model_output, attention_mask):
	"""Average pooling for sentence embeddings calculation"""
	token_embeddings = model_output.last_hidden_state
	input_mask_expanded = (
		attention_mask.unsqueeze(-1)
		.expand(token_embeddings.size())
		.float()
	)
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
		input_mask_expanded.sum(1), min=1e-9
	)

def measure_model_performance(model_name, precision=None, input_text="The quick brown fox"):
	"""Benchmark embedding model performance across key metrics"""
	
	# Check if CUDA is available
	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	# Initialize memory tracking
	start_mem = 0
	if device == "cuda":
		torch.cuda.reset_peak_memory_stats()
		start_mem = torch.cuda.memory_allocated()
	
	# Load tokenizer first to check model config
	print(f"Loading tokenizer for {model_name}...")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	
	# Determine model's default precision
	config = AutoConfig.from_pretrained(model_name)
	if precision is None:
		precision = getattr(config, "torch_dtype", "fp16")
		precision = str(precision).split(".")[-1]  # Convert torch dtype to string format
	
	# Configure model loading parameters
	load_params = {
		"device_map": "auto" if device == "cuda" else "cpu",
		"trust_remote_code": True
	}
	
	if precision == "8bit":
		load_params["load_in_8bit"] = True
	elif precision == "4bit":
		load_params["load_in_4bit"] = True
	else:
		load_params["torch_dtype"] = torch.float16 if precision == "fp16" else torch.float32

	# Load model with progress monitoring
	print(f"Loading {model_name} in {precision} precision...")
	model = AutoModel.from_pretrained(model_name, **load_params)
	
	# Calculate model memory footprint
	if device == "cuda":
		model_mem = torch.cuda.memory_allocated() - start_mem
		print(f"Model loaded. VRAM usage: {bytes_to_gb(model_mem):.2f}GB")
	else:
		model_mem = 0  # CPU memory tracking not implemented
		print("Model loaded on CPU")

	# Tokenize input
	inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
	inputs = {k: v.to(device) for k, v in inputs.items()}

	# Warmup run
	print("Performing warmup run...")
	with torch.no_grad():
		model(**inputs)

	# Performance measurement
	if device == "cuda":
		torch.cuda.synchronize()
	start_time = time.time()
	
	with torch.no_grad():
		outputs = model(**inputs)
	
	if device == "cuda":
		torch.cuda.synchronize()
	elapsed = time.time() - start_time
	
	# Get embeddings
	embeddings = outputs.last_hidden_state.mean(dim=1)
	
	# Save embeddings to file
	os.makedirs("./results", exist_ok=True)
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	embedding_file = f"./results/embeddings_{model_name.replace('/', '_')}_{timestamp}.npy"
	np.save(embedding_file, embeddings.cpu().numpy())
	print(f"Embeddings saved to {embedding_file}")
	
	# Calculate metrics
	embedding_size = embeddings.shape[1]
	peak_mem = bytes_to_gb(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
	
	return {
		"model": model_name,
		"vram_usage": bytes_to_gb(model_mem) if device == "cuda" else 0,
		"peak_vram": peak_mem,
		"inference_time": elapsed,
		"embedding_size": embedding_size,
		"precision": precision,
		"device": device,
		"embedding_file": embedding_file,
		"timestamp": timestamp
	}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True, help="Hugging Face model identifier")
	parser.add_argument("--precision", choices=["fp32", "fp16", "8bit", "4bit"], default=None)
	parser.add_argument("--input", default="The quick brown fox", help="Input text")
	parser.add_argument("--accelerator", choices=["cpu", "nvidia"], default="nvidia")
	parser.add_argument("--benchmark_mode", choices=["llm", "embedding", "image", "all"], default="all")
	
	args = parser.parse_args()
	
	results = measure_model_performance(
		args.model,
		args.precision,
		args.input
	)
	
	# Output JSON marker for easy parsing
	print("\n@@@BENCHMARK_RESULTS_START@@@")
	print(json.dumps(results))
	print("@@@BENCHMARK_RESULTS_END@@@")

	# Create results directory if it doesn't exist
	results_dir = Path("./results")
	results_dir.mkdir(exist_ok=True)
	
	# Create a clean model name for the filename (remove slashes)
	clean_model_name = args.model.replace("/", "_")
	
	# Get precision value to use in filename (might be None from args)
	precision_str = results["precision"]
	
	# Create filename
	filename = f"perf_embedding_{clean_model_name}_{precision_str}_{results['timestamp']}.json"
	file_path = results_dir / filename
	
	# Add results to the file
	with open(file_path, "w") as f:
		json.dump(results, f, indent=2)
	
	print(f"\nBenchmark results saved to {file_path}")
	
	# Also save to a cumulative results file that combines all runs
	cumulative_file = results_dir / f"perf_embedding_{clean_model_name}_{precision_str}_cumulative.json"
	
	# Load existing data if file exists
	cumulative_data = []
	if cumulative_file.exists():
		try:
			with open(cumulative_file, "r") as f:
				cumulative_data = json.load(f)
				if not isinstance(cumulative_data, list):
					cumulative_data = [cumulative_data]
		except json.JSONDecodeError:
			# If file exists but is not valid JSON, start fresh
			cumulative_data = []
	
	# Add new results
	cumulative_data.append(results)
	
	# Write back the combined results
	with open(cumulative_file, "w") as f:
		json.dump(cumulative_data, f, indent=2)
	
	print(f"Results also appended to cumulative file: {cumulative_file}")
