from dotenv import load_dotenv
import os
import sys
from io import StringIO
import json
import re

import subprocess

original_mode = os.getenv('BENCHMARK_MODE')
# load env
load_dotenv()

benchmark_mode = ""
if original_mode == "" or original_mode == None or original_mode == "None":
	benchmark_mode = os.getenv('BENCHMARK_MODE')
	if benchmark_mode == "" or benchmark_mode == None or benchmark_mode == "None":
		raise ValueError("BENCHMARK_MODE is not set")
else:
	benchmark_mode = original_mode

accelerator_type = ""
try:
	nvcc_version = os.popen('nvcc --version').read()
	if "12.8" in nvcc_version:
		accelerator_type = "nvidia"
	else:
		#TODO: AMD ROCm
		accelerator_type = "cpu"
except Exception as e:
	# If nvcc command is not found or fails, default to CPU
	nvcc_version = "not found"
	accelerator_type = "cpu"

print(f"Benchmark mode: {benchmark_mode} (accelerator type: {accelerator_type})")



llm_mode_enabled = benchmark_mode == "llm" or \
	benchmark_mode == "text-only" or \
	benchmark_mode == "all"
embedding_mode_enabled = benchmark_mode == "embedding" or \
	benchmark_mode == "text-only" or \
	benchmark_mode == "all"
image_mode_enabled = benchmark_mode == "image" or \
	benchmark_mode == "all"


if llm_mode_enabled:
	original_argv = sys.argv
	
	for model_name in [ 
		"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
		#"meta-llama/Llama-3.2-3B-Instruct",
	]:

		for prompt in [
			"Hello, world!",
			"What is the capital of France?",
			"Write me a python script which calculates the fibonacci sequence up to 1000000. Each number must be printed, you need to use recursion.",
			"Write me an essay about Julius Caesar's time in Gaul, but in the style of a carribean pirate.",
		]:
			command = [
				os.getenv('PYTHON_CMD'),
				'llm/llm_hugging_face.py',
				'--accelerator', accelerator_type,
				'--benchmark_mode', benchmark_mode,
				'--model', model_name,
				'--precision', 'fp16',
				'--input', prompt,
				'--batch_size', '1'
			]
			
			# Use subprocess to capture output in real-time
			process = subprocess.Popen(
				command,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				bufsize=1
			)
			
			# Collect all output for later JSON parsing
			full_output = []
			
			# Print output in real-time
			print(f"\nRunning benchmark for model: {model_name}")
			print("-" * 80)
			for line in iter(process.stdout.readline, ''):
				print(line, end='')  # Print in real-time
				full_output.append(line)
				
			process.stdout.close()
			process.wait()
			
			# Join all output lines for JSON parsing
			process_output = ''.join(full_output)
			
			# Extract JSON results using markers
			start_marker = "@@@BENCHMARK_RESULTS_START@@@"
			end_marker = "@@@BENCHMARK_RESULTS_END@@@"
			
			if start_marker in process_output and end_marker in process_output:
				json_str = process_output.split(start_marker)[1].split(end_marker)[0].strip()
				try:
					results = json.loads(json_str)
					print("\nParsed Results:")
					print(json.dumps(results, indent=2))
				except json.JSONDecodeError as e:
					print(f"Error parsing results JSON: {e}")
			
			print(f"LLM '{model_name}' completed")
			print("-" * 80)

if image_mode_enabled:
	original_argv = sys.argv
	
	for model_name in [
		"stabilityai/stable-diffusion-2-1",
	]:
		for resolution, num_inference_steps in [
			["256", 50],
			["512", 20],
			["1024", 10],
			["2048", 10]
		]:
			for prompt in [
				"A beautiful mountain landscape",
				"A serene beach with a sunset",
				"A bustling cityscape at night",
				"A serene forest with a waterfall",
				"A serene forest with a waterfall",
			]:
				command = [
					os.getenv('PYTHON_CMD'),
					'image/image_hugging_face.py',
					'--accelerator', accelerator_type,
					'--benchmark_mode', benchmark_mode,
					'--model', model_name,
					'--input', prompt,
					'--num_inference_steps', str(num_inference_steps),
					'--width', resolution,
					'--height', resolution
				]
				process = subprocess.Popen(
					command,
					stdout=subprocess.PIPE,
					stderr=subprocess.STDOUT,
					text=True,
					bufsize=1
				)

				full_output = []
				for line in iter(process.stdout.readline, ''):
					print(line, end='')
					full_output.append(line)

				process.stdout.close()
				process.wait()

				process_output = ''.join(full_output)
				
				start_marker = "@@@BENCHMARK_RESULTS_START@@@"
				end_marker = "@@@BENCHMARK_RESULTS_END@@@"

				if start_marker in process_output and end_marker in process_output:
					json_str = process_output.split(start_marker)[1].split(end_marker)[0].strip()
					try:
						results = json.loads(json_str)
						print("\nParsed Results:")
						print(json.dumps(results, indent=2))
					except json.JSONDecodeError as e:
						print(f"Error parsing results JSON: {e}")

				print(f"Image '{model_name}' completed")
				print("-" * 80)

if embedding_mode_enabled:
	original_argv = sys.argv
	
	for model_name in [
		"sentence-transformers/all-MiniLM-L6-v2",
	]:

		hamlet_text = open("datasets/text/hamlet.txt", "r").read()
		command = [
			os.getenv('PYTHON_CMD'),
			'embedding/embed_hugging_face.py',
			'--accelerator', accelerator_type,
			'--benchmark_mode', benchmark_mode,
			'--model', model_name,
			'--input', hamlet_text,
		]
		process = subprocess.Popen(
			command,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			bufsize=1
		)

		full_output = []
		for line in iter(process.stdout.readline, ''):
			print(line, end='')
			full_output.append(line)

		process.stdout.close()
		process.wait()

		process_output = ''.join(full_output)
		
		start_marker = "@@@BENCHMARK_RESULTS_START@@@"
		end_marker = "@@@BENCHMARK_RESULTS_END@@@"

		if start_marker in process_output and end_marker in process_output:
			json_str = process_output.split(start_marker)[1].split(end_marker)[0].strip()
			try:
				results = json.loads(json_str)
				print("\nParsed Results:")
				print(json.dumps(results, indent=2))
			except json.JSONDecodeError as e:
				print(f"Error parsing results JSON: {e}")

		print(f"Embedding '{model_name}' completed")
		print("-" * 80)
		

