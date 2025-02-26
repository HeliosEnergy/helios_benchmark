import torch
from transformers import AutoModel, AutoTokenizer
import time
import argparse
import json

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

def measure_embedding_performance(model_name, precision=None, input_text="The quick brown fox", batch_size=1):
    """Benchmark embedding model performance across key metrics"""
    
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
        precision = str(precision).split(".")[-1]
    
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
        model_mem = 0
        print("Model loaded on CPU")

    # Prepare input batches
    print(f"Preparing batch size {batch_size}...")
    inputs = tokenizer(
        [input_text] * batch_size,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Warmup run
    print("Performing warmup run...")
    with torch.no_grad():
        model_output = model(**inputs)
        embeddings = mean_pooling(model_output, inputs["attention_mask"])

    # Performance measurement
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        model_output = model(**inputs)
        embeddings = mean_pooling(model_output, inputs["attention_mask"])
    
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    # Calculate metrics
    embedding_dim = embeddings.shape[1]
    embeddings_per_sec = batch_size / elapsed
    peak_mem = bytes_to_gb(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    
    return {
        "vram_usage": bytes_to_gb(model_mem),
        "peak_vram": peak_mem,
        "total_embeddings": batch_size,
        "inference_time": elapsed,
        "embeddings_per_sec": embeddings_per_sec,
        "embedding_dimension": embedding_dim,
        "precision": precision,
        "device": device
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model identifier")
    parser.add_argument("--accelerator", choices=["cpu", "nvidia"], default="nvidia")
    parser.add_argument("--benchmark_mode", choices=["llm", "embedding", "image", "all"], default="all")
    parser.add_argument("--precision", choices=["fp32", "fp16", "8bit", "4bit"], default="fp16")
    parser.add_argument("--input", default="The quick brown fox", help="Input text")
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    
    results = measure_embedding_performance(
        args.model,
        args.precision,
        args.input,
        args.batch_size
    )
    
    # Add model name to results
    results["model"] = args.model
    
    # Output JSON marker for easy parsing
    print("\n@@@BENCHMARK_RESULTS_START@@@")
    print(json.dumps(results))
    print("@@@BENCHMARK_RESULTS_END@@@")
