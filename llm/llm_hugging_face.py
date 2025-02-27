import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import argparse
import json

def bytes_to_gb(bytes):
	return bytes / (1024 ** 3)

def measure_model_performance(model_name, precision=None, input_text="The quick brown fox", max_tokens=None, batch_size=1):
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
	inputs = tokenizer(input_text, return_tensors="pt", padding=True)
	inputs = {k: v.to(model.device) for k, v in inputs.items()}
	if device_type == "cuda":
		torch.cuda.synchronize()
	tokenization_time = time.time() - tokenization_start
	input_token_count = inputs["input_ids"].size(1)
	input_tokens_per_sec = input_token_count / tokenization_time if tokenization_time > 0 else 0
	print(f"Input tokenization: {tokenization_time:.6f}s for {input_token_count} tokens ({input_tokens_per_sec:.2f} tokens/sec)")
	
	# Alternative approach for token timing - generate tokens one by one
	token_times = []
	generated_ids = inputs["input_ids"].clone()
	attention_mask = inputs["attention_mask"].clone()
	
	# Start generation with timing
	if device_type == "cuda":
		torch.cuda.synchronize()
	
	# Record start time
	generation_start = time.time()
	token_times.append(generation_start)
	
	with torch.no_grad():
		for _ in range(max_tokens):
			# Generate one token at a time
			if device_type == "cuda":
				torch.cuda.synchronize()
			
			token_start = time.time()
			outputs = model(
				input_ids=generated_ids,
				attention_mask=attention_mask
			)
			next_token_logits = outputs.logits[:, -1, :]
			next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
			
			if device_type == "cuda":
				torch.cuda.synchronize()
			
			token_times.append(time.time())
			
			# Add the predicted token to the sequence
			generated_ids = torch.cat([generated_ids, next_token], dim=1)
			# Extend attention mask for the new token
			attention_mask = torch.cat([
				attention_mask, 
				torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
			], dim=1)
			
			# Stop if we hit the EOS token
			if next_token.item() == tokenizer.eos_token_id:
				break
	
	if device_type == "cuda":
		torch.cuda.synchronize()
	generation_end = time.time()
	
	# Calculate metrics
	generated_text = tokenizer.decode(generated_ids[0][input_token_count:], skip_special_tokens=True)
	generated_tokens = len(generated_ids[0]) - input_token_count
	
	# Calculate per-token generation times
	total_generation_time = generation_end - generation_start
	output_tokens_per_sec = generated_tokens / total_generation_time if total_generation_time > 0 else 0
	print(f"Output generation: {total_generation_time:.6f}s for {generated_tokens} tokens ({output_tokens_per_sec:.2f} tokens/sec)")
	
	# Calculate per-token times (differential)
	token_generation_times = []
	for i in range(1, len(token_times)):
		token_generation_times.append(token_times[i] - token_times[i-1])
	
	peak_mem = bytes_to_gb(torch.cuda.max_memory_allocated()) if device_type == "cuda" else 0
	
	# Save detailed token times for analysis
	token_timing_details = {
		"input_tokenization_time": tokenization_time,
		"input_token_count": input_token_count,
		"input_tokens_per_sec": input_tokens_per_sec,
		"output_token_count": generated_tokens,
		"output_tokens_per_sec": output_tokens_per_sec,
		"total_generation_time": total_generation_time,
	}
	
	return {
		"model": model_name,
		"vram_usage": bytes_to_gb(model_mem) if device_type == "cuda" else 0,
		"peak_vram": peak_mem,
		"input_tokens": input_token_count,
		"output_tokens": generated_tokens,
		"total_tokens": input_token_count + generated_tokens,
		"input_tokenization_time": tokenization_time,
		"input_tokens_per_sec": input_tokens_per_sec,
		"generation_time": total_generation_time,
		"output_tokens_per_sec": output_tokens_per_sec,
		"precision": precision,
		"device": device_type,
		"generated_text": generated_text,
		"token_timing_details": token_timing_details
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
	
	# Output JSON marker for easy parsing
	print("\n@@@BENCHMARK_RESULTS_START@@@")
	print(json.dumps(results))
	print("@@@BENCHMARK_RESULTS_END@@@")