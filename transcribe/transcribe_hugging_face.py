import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoFeatureExtractor
import time
import argparse
import json
import os
import numpy as np
from datetime import datetime
import librosa

def bytes_to_gb(bytes):
    return bytes / (1024 ** 3)

def measure_model_performance(model_name, precision=None, audio_path=None, max_length=None, return_timestamps=False):
    """Benchmark speech transcription model performance across key metrics"""
    
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
    
    # Default to a sample audio if none provided
    if audio_path is None:
        # Use a built-in sample or a reasonable default
        print("No audio file provided, using sample audio")
        audio_path = "sample.wav"  # Update this with appropriate default
    
    if not os.path.exists(audio_path):
        return {
            "error": f"Audio file not found: {audio_path}",
            "model": model_name
        }
    
    # Determine model's default precision
    if precision is None:
        precision = "fp16" if device_type == "cuda" else "fp32"
    
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

    # Load processor first
    print(f"Loading processor for {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Load model with progress monitoring
    print(f"Loading {model_name} in {precision} precision...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **load_params)
    
    # Calculate model memory footprint
    if device_type == "cuda":
        model_mem = torch.cuda.memory_allocated() - start_mem
        print(f"Model loaded. VRAM usage: {bytes_to_gb(model_mem):.2f}GB")
    else:
        model_mem = 0  # CPU memory tracking not implemented
        print("Model loaded on CPU")

    # Load and preprocess audio
    print(f"Loading audio file: {audio_path}")
    audio_loading_start = time.time()
    audio_input, sample_rate = librosa.load(audio_path, sr=16000)
    audio_duration = len(audio_input) / sample_rate
    audio_loading_time = time.time() - audio_loading_start
    print(f"Audio loaded. Duration: {audio_duration:.2f}s, Loading time: {audio_loading_time:.2f}s")
    
    # Prepare inputs
    preprocessing_start = time.time()
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    preprocessing_time = time.time() - preprocessing_start
    
    # Warmup run to ensure everything is loaded
    print("Performing warmup run...")
    with torch.no_grad():
        model.generate(**inputs, max_length=20)
    print("Warmup complete.")
    
    # Transcription with timing
    if device_type == "cuda":
        torch.cuda.synchronize()
    
    transcription_start = time.time()
    generate_kwargs = {
        "max_length": max_length if max_length else 448
    }
    
    if return_timestamps:
        generate_kwargs["return_timestamps"] = return_timestamps
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    
    if device_type == "cuda":
        torch.cuda.synchronize()
    
    transcription_time = time.time() - transcription_start
    
    # Decode outputs
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)
    
    # Calculate metrics
    real_time_factor = transcription_time / audio_duration if audio_duration > 0 else float("inf")
    peak_mem = bytes_to_gb(torch.cuda.max_memory_allocated()) if device_type == "cuda" else 0
    
    # Save transcription to file
    os.makedirs("./results", exist_ok=True)
    transcription_file = f"./results/transcription_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write(transcription[0])
    print(f"Transcription saved to {transcription_file}")
    
    return {
        "model": model_name,
        "vram_usage": bytes_to_gb(model_mem) if device_type == "cuda" else 0,
        "peak_vram": peak_mem,
        "audio_duration": audio_duration,
        "audio_loading_time": audio_loading_time,
        "preprocessing_time": preprocessing_time,
        "transcription_time": transcription_time,
        "real_time_factor": real_time_factor,
        "precision": precision,
        "device": device_type,
        "transcription": transcription[0][:500] + ("..." if len(transcription[0]) > 500 else ""),  # Truncate if too long
        "transcription_file": transcription_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model identifier (e.g., openai/whisper-base)")
    parser.add_argument("--precision", choices=["fp32", "fp16", "8bit", "4bit"], default=None)
    parser.add_argument("--audio", help="Path to audio file to transcribe")
    parser.add_argument("--max_length", type=int, help="Maximum token length for transcription")
    parser.add_argument("--return_timestamps", action="store_true", help="Return word-level timestamps")
    parser.add_argument("--accelerator", choices=["cpu", "nvidia"], default="nvidia")
    parser.add_argument("--benchmark_mode", choices=["llm", "embedding", "image", "transcription", "all"], default="all")
    
    args = parser.parse_args()
    
    results = measure_model_performance(
        args.model,
        args.precision,
        args.audio,
        args.max_length,
        args.return_timestamps
    )
    
    # Output JSON marker for easy parsing
    print("\n@@@BENCHMARK_RESULTS_START@@@")
    print(json.dumps(results))
    print("@@@BENCHMARK_RESULTS_END@@@")
