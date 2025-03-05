# Use the official NVIDIA CUDA image compatible with RTX 4090
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    sudo \
    build-essential \
    pkg-config \
    wget \
    gnupg \
    software-properties-common \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 --version && python3 -m pip --version && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install newer version of CMake from Kitware's repository
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake

# Clone the vLLM repository
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm

# Set working directory
WORKDIR /opt/vllm

# Create and activate a virtual environment
RUN python3 -m venv /opt/vllm/venv
ENV PATH="/opt/vllm/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-cuda.txt && \
    pip install setuptools_scm

# Build vLLM
RUN python3 setup.py install

# Set the entrypoint to keep the container running
CMD ["bash", "-c", "while true; do sleep 1000; done"]
