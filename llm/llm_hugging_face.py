import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
import time
import argparse
import json

def bytes_to_gb(bytes):
	return bytes / (1024 ** 3)

def measure_model_performance(model_name, precision=None, input_text="The quick brown fox", max_tokens=None, batch_size=1):
	"""Benchmark LLM performance across key metrics"""
	
	
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
	model = AutoModelForCausalLM.from_pretrained(model_name, **load_params)
	
	# Calculate model memory footprint
	if device == "cuda":
		model_mem = torch.cuda.memory_allocated() - start_mem
		print(f"Model loaded. VRAM usage: {bytes_to_gb(model_mem):.2f}GB")
	else:
		model_mem = 0  # CPU memory tracking not implemented
		print("Model loaded on CPU")

	# Create text generation pipeline
	generator = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer
	)

	# Warmup run
	print("Performing warmup run...")
	generator(input_text, max_new_tokens=10, batch_size=batch_size)

	# Performance measurement
	if device == "cuda":
		torch.cuda.synchronize()
	start_time = time.time()
	
	with torch.no_grad():
		outputs = generator(
			input_text,
			max_new_tokens=max_tokens,
			batch_size=batch_size,
			pad_token_id=tokenizer.eos_token_id
		)
	
	if device == "cuda":
		torch.cuda.synchronize()
	elapsed = time.time() - start_time
	
	# Calculate metrics
	generated_tokens = len(tokenizer.encode(outputs[0]['generated_text']))
	tokens_per_sec = generated_tokens / elapsed
	
	peak_mem = bytes_to_gb(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
	
	return {
		"vram_usage": bytes_to_gb(model_mem) if device == "cuda" else 0,
		"peak_vram": peak_mem,
		"total_tokens": generated_tokens,
		"inference_time": elapsed,
		"tokens_per_sec": tokens_per_sec,
		"precision": precision,
		"device": device
	}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True, help="Hugging Face model identifier")
	parser.add_argument("--precision", choices=["fp32", "fp16", "8bit", "4bit"], default="fp16")
	parser.add_argument("--input", default="The quick brown fox", help="Input prompt")
	parser.add_argument("--max_tokens", type=int, default=50)
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
	
	# Add model name to results
	results["model"] = args.model
	
	# Output JSON marker for easy parsing
	print("\n@@@BENCHMARK_RESULTS_START@@@")
	print(json.dumps(results))
	print("@@@BENCHMARK_RESULTS_END@@@")
