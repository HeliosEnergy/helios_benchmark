# helios_benchmark
AI benchmarking for GPUs

Results written to ./results

# Local Setup

## Intel/CPU
```
# Note, if you don't want to reinstall BNBs dependencies, append the `--no-deps` flag!
pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'
```

# Docker
DockerHub: https://hub.docker.com/repository/docker/heliosai/helios-benchmark/general

### CPU

###### Locally
```
docker compose --profile cpu up --build
```

### NVIDIA
Uses CUDA 12.8 for testing.

##### Locally
```
docker compose --profile nvidia up --build
```



# Scripts

### Embed Compare
Vector similiary search to test results manually.
```
python ./scripts/embed_compare_similitary.py ./results/embeddings_sentence-transformers_all-MiniLM-L6-v2_hamlet_20250301_101440.npy ./results/embeddings_sentence-transformers_all-MiniLM-L6-v2_romeo_and_juliet_20250301_101443.npy 
```