import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import argparse
import json
import os
import datetime
from pathlib import Path

def bytes_to_gb(bytes):
	return bytes / (1024 ** 3)

def measure_model_performance(model_name, precision=None, input_text="The quick brown fox", max_tokens=2048, batch_size=1):
	"""Benchmark LLM performance across key metrics"""
	
	# Check if CUDA is available
	device_type = "cuda" if torch.cuda.is_available() else "cpu"
	
	# Initialize memory tracking
	start_mem = 0
	if device_type == "cuda":
		torch.cuda.reset_peak_memory_stats()
		start_mem = torch.cuda.memory_allocated()

	print("Device:", device_type)

	if device_type == "cuda":
		device = "cuda:0"
		print("Device id:", device)
	
	# Load tokenizer first to check model config
	print(f"Loading tokenizer for {model_name}...")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	
	# Ensure pad token is set
	if tokenizer.pad_token is None:
		if tokenizer.eos_token is not None:
			tokenizer.pad_token = tokenizer.eos_token
			print("Setting pad_token to eos_token")
		else:
			# Add a padding token if needed
			tokenizer.add_special_tokens({'pad_token': '[PAD]'})
			print("Added [PAD] token to tokenizer")
	
	# Determine model's default precision and max tokens
	config = AutoConfig.from_pretrained(model_name)
	if precision is None:
		# Default to fp16 if torch_dtype is not specified in config
		precision = getattr(config, "torch_dtype", "fp16")
		precision = str(precision).split(".")[-1]  # Convert torch dtype to string format
	
	if max_tokens is None:
		# Use model's max length or default to 2048 if not specified
		max_tokens = getattr(config, "max_position_embeddings", 2048)
		print(f"Using model's maximum context length: {max_tokens}")
	
	# Configure model loading parameters
	load_params = {
		"device_map": "auto" if device_type == "cuda" else "cpu",
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
	model = AutoModelForCausalLM.from_pretrained(model_name, **load_params)
	
	# Resize token embeddings if we added new tokens
	if len(tokenizer) != model.config.vocab_size:
		model.resize_token_embeddings(len(tokenizer))
	
	# Apply optimizations when available
	if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
		# Enable Flash Attention if available
		model.config.attn_implementation = "flash_attention_2"
		print("Enabled Flash Attention 2")
	
	# Apply torch.compile for PyTorch 2.0+ if available
	if hasattr(torch, "compile") and device_type == "cuda":
		try:
			model = torch.compile(model)
			print("Applied torch.compile() for faster inference")
		except Exception as e:
			print(f"Could not apply torch.compile(): {e}")
	
	# Calculate model memory footprint
	if device_type == "cuda":
		model_mem = torch.cuda.memory_allocated() - start_mem
		print(f"Model loaded. VRAM usage: {bytes_to_gb(model_mem):.2f}GB")
	else:
		model_mem = 0  # CPU memory tracking not implemented
		print("Model loaded on CPU")

	# Warmup run to ensure everything is loaded
	print("Performing warmup run...")
	inputs = tokenizer(input_text, return_tensors="pt", padding=True)
	inputs = {k: v.to(model.device) for k, v in inputs.items()}
	with torch.no_grad():
		model.generate(**inputs, max_new_tokens=5)
	print("Warmup complete.")
	
	# Precise timing for tokenization
	if device_type == "cuda":
		torch.cuda.synchronize()
	tokenization_start = time.time()
	
	# Handle batch size properly
	batch_texts = [input_text] * batch_size
	inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)
	inputs = {k: v.to(model.device) for k, v in inputs.items()}
	
	if device_type == "cuda":
		torch.cuda.synchronize()
	tokenization_time = time.time() - tokenization_start
	input_token_count = inputs["input_ids"].size(1)
	input_tokens_per_sec = input_token_count / tokenization_time if tokenization_time > 0 else 0
	print(f"Input tokenization: {tokenization_time:.6f}s for {input_token_count} tokens ({input_tokens_per_sec:.2f} tokens/sec)")
	
	# Start generation with timing using optimized generate method
	if device_type == "cuda":
		torch.cuda.synchronize()
	
	# Record start time
	generation_start = time.time()
	
	with torch.no_grad():
		generate_kwargs = {
			"max_new_tokens": max_tokens,
			"do_sample": False,        # Deterministic for benchmarking
			"use_cache": True,         # Enable KV caching
			"pad_token_id": tokenizer.pad_token_id,
			"return_dict_in_generate": True,
			"output_scores": True      # Needed for token-by-token timing
		}
		
		outputs = model.generate(**inputs, **generate_kwargs)
	
	if device_type == "cuda":
		torch.cuda.synchronize()
	generation_end = time.time()
	
	# Calculate metrics
	generated_ids = outputs.sequences
	generated_text = tokenizer.batch_decode(
		generated_ids[:, inputs["input_ids"].shape[1]:], 
		skip_special_tokens=True
	)[0]  # Take first batch item for display
	
	generated_tokens = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]

	print("-" * 10)
	print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
	print("-" * 10)
	
	# Calculate per-token generation times
	total_generation_time = generation_end - generation_start
	
	# Account for batch size in the tokens per second calculation
	total_generated_tokens = generated_tokens * batch_size
	output_tokens_per_sec = total_generated_tokens / total_generation_time if total_generation_time > 0 else 0
	print(f"Output generation: {total_generation_time:.6f}s for {total_generated_tokens} tokens total ({output_tokens_per_sec:.2f} tokens/sec)")
	
	peak_mem = bytes_to_gb(torch.cuda.max_memory_allocated()) if device_type == "cuda" else 0
	
	# Save detailed token times for analysis
	token_timing_details = {
		"input_tokenization_time": tokenization_time,
		"input_token_count": input_token_count,
		"input_tokens_per_sec": input_tokens_per_sec,
		"output_token_count": generated_tokens,
		"total_output_tokens": total_generated_tokens,
		"output_tokens_per_sec": output_tokens_per_sec,
		"total_generation_time": total_generation_time,
		"batch_size": batch_size
	}
	
	return {
		"model": model_name,
		"vram_usage": bytes_to_gb(model_mem) if device_type == "cuda" else 0,
		"peak_vram": peak_mem,
		"input_tokens": input_token_count,
		"output_tokens": generated_tokens,
		"total_output_tokens": total_generated_tokens,
		"total_tokens": input_token_count + generated_tokens,
		"input_tokenization_time": tokenization_time,
		"input_tokens_per_sec": input_tokens_per_sec,
		"generation_time": total_generation_time,
		"output_tokens_per_sec": output_tokens_per_sec,
		"precision": precision,
		"device": device_type,
		"generated_text": generated_text,
		"token_timing_details": token_timing_details,
		"batch_size": batch_size
	}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True, help="Hugging Face model identifier")
	parser.add_argument("--precision", choices=["fp32", "fp16", "8bit", "4bit"], default="fp16")
	parser.add_argument("--input", default="The quick brown fox", help="Input prompt")
	parser.add_argument("--max_tokens", type=int, default=2048)
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--accelerator", choices=["cpu", "nvidia"], default="nvidia")
	parser.add_argument("--benchmark_mode", choices=["llm", "embedding", "image", "all"], default="all")
	
	args = parser.parse_args()
	
	results = measure_model_performance(
		args.model,
		args.precision,
		args.input,
		args.max_tokens,
		args.batch_size
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
	
	# Generate timestamp for filename
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	
	# Create filename
	filename = f"perf_llm_{clean_model_name}_{args.precision}_{timestamp}.json"
	file_path = results_dir / filename
	
	# Add results to the file
	with open(file_path, "w") as f:
		json.dump(results, f, indent=2)
	
	print(f"\nBenchmark results saved to {file_path}")
	
	# Also save to a cumulative results file that combines all runs
	cumulative_file = results_dir / f"perf_llm_{clean_model_name}_{args.precision}_cumulative.json"
	
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
	
	# Add new results with timestamp
	results["timestamp"] = timestamp
	cumulative_data.append(results)
	
	# Write back the combined results
	with open(cumulative_file, "w") as f:
		json.dump(cumulative_data, f, indent=2)
	
	print(f"Results also appended to cumulative file: {cumulative_file}")