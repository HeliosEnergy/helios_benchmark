# helios_benchmark
AI benchmarking for GPUs

Results written to ./results

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