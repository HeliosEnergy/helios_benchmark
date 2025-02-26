import torch
from diffusers import StableDiffusionPipeline
import time
import argparse
import json
from PIL import Image
import io
import os
from datetime import datetime

def bytes_to_gb(bytes):
    return bytes / (1024 ** 3)

def measure_model_performance(model_name, precision=None, input_prompt="A scenic landscape", 
                             num_inference_steps=10, batch_size=1, width=1024, height=1024):
    """Benchmark image generation model performance across key metrics"""

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Adjust precision based on device
    if device == "cpu" and precision == "fp16":
        print("Warning: fp16 not supported on CPU. Switching to fp32.")
        precision = "fp32"
    
    torch_dtype = torch.float16 if precision == "fp16" else torch.float32
    print(f"Using precision: {precision} ({torch_dtype})")
    print(f"Image resolution: {width}x{height}")

    # Initialize memory tracking
    start_mem = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

    # Load model with specified precision
    print(f"Loading model {model_name}...")
    
    try:
        model = StableDiffusionPipeline.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
            safety_checker=None  # Disable safety checker for benchmarking
        )
        model = model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {
            "error": str(e),
            "model": model_name,
            "device": device,
            "precision": precision
        }

    # Calculate model memory footprint
    if device == "cuda":
        model_mem = torch.cuda.memory_allocated() - start_mem
        print(f"Model loaded. VRAM usage: {bytes_to_gb(model_mem):.2f}GB")
    else:
        model_mem = 0  # CPU memory tracking not implemented
        print("Model loaded on CPU")

    # Warmup run
    print("Performing warmup run...")
    try:
        _ = model(input_prompt, num_inference_steps=5, width=width, height=height)
    except Exception as e:
        print(f"Error during warmup: {e}")
        return {
            "error": str(e),
            "model": model_name,
            "device": device,
            "precision": precision
        }

    # Performance measurement
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    try:
        with torch.no_grad():
            outputs = model(
                [input_prompt] * batch_size,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height
            )
    except Exception as e:
        print(f"Error during inference: {e}")
        return {
            "error": str(e),
            "model": model_name,
            "device": device,
            "precision": precision
        }

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # Calculate metrics
    peak_mem = bytes_to_gb(torch.cuda.max_memory_allocated()) if device == "cuda" else 0

    # Create results directory if it doesn't exist
    os.makedirs("./results", exist_ok=True)
    
    # Save one of the generated images to verify output
    try:
        image = outputs.images[0]
        filename = f"./results/generated_image_{model_name.replace('/', '_')}_{width}x{height}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image.save(filename)
        print(f"Image saved to {filename}")
    except Exception as e:
        print(f"Error saving image: {e}")

    return {
        "model": model_name,
        "vram_usage": bytes_to_gb(model_mem) if device == "cuda" else 0,
        "peak_vram": peak_mem,
        "inference_time": elapsed,
        "precision": precision,
        "device": device,
        "steps": num_inference_steps,
        "resolution": f"{width}x{height}",
        "image_path": filename,
        "prompt": input_prompt
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model identifier")
    parser.add_argument("--accelerator", choices=["cpu", "nvidia"], default="nvidia")
    parser.add_argument("--benchmark_mode", choices=["llm", "embedding", "image", "all"], default="all")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--input", default="A scenic landscape", help="Input prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=512, help="Image height in pixels")

    args = parser.parse_args()

    results = measure_model_performance(
        args.model,
        args.precision,
        args.input,
        args.num_inference_steps,
        args.batch_size,
        args.width,
        args.height
    )

    # Output JSON marker for easy parsing
    print("\n@@@BENCHMARK_RESULTS_START@@@")
    print(json.dumps(results))
    print("@@@BENCHMARK_RESULTS_END@@@")
