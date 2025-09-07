FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Basic setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim ca-certificates python3 python3-pip python3-dev build-essential cmake \
    libtinfo-dev zlib1g-dev libcurl4-openssl-dev libssl-dev \
    llvm-dev clang cython3 python-is-python3 libedit-dev libxml2-dev \
    automake make gcc g++ graphviz sudo apt-utils locales \
    && rm -rf /var/lib/apt/lists/*

# Setup locale
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Set timezone
ENV TZ="America/Sao_Paulo"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install PyTorch (CUDA 12.4 support via pip nightly wheels)
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install TVM v0.13.0
WORKDIR /opt
RUN git clone -b v0.13.0 --recursive https://github.com/apache/tvm tvm
WORKDIR /opt/tvm/build

# Download TVM config for CUDA support
RUN wget https://raw.githubusercontent.com/lac-dcc/DropletSearch/main/docker/cuda/config.cmake

# Build TVM with CUDA support
RUN cmake .. \
    -DUSE_CUDA=ON \
    -DUSE_LLVM=ON \
    -DUSE_METAL=OFF \
    && make -j$(nproc)

# Set TVM environment variables
ENV PYTHONPATH=/opt/tvm/python:${PYTHONPATH}
ENV TVM_HOME=/opt/tvm
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.4/compat/

# Install Python deps for TVM & benchmarks
RUN pip install numpy==1.24.4 decorator psutil scipy tornado attrs pytest mypy orderedset Pillow \
    typing_extensions cloudpickle synr mxnet transformers onnx==1.15.0
RUN pip install "xgboost>=1.1.0"

# Setup working directory for Bennu
WORKDIR /root/bennu/

# Add convenience aliases and environment setup
RUN echo "export PYTHONPATH=/opt/tvm/python:\$PYTHONPATH" >> ~/.bashrc && \
    echo "export TVM_HOME=/opt/tvm" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.4/compat/" >> ~/.bashrc && \
    echo "alias python=python3" >> ~/.bashrc && \
    echo "alias pip=pip3" >> ~/.bashrc

# Add build and run instructions as comments
# Build: docker build -t bennu-tvm:cuda12.4 .
# Run: docker run --gpus all -it --rm -v $(pwd):/root/bennu bennu-tvm:cuda12.4
# Test TVM: python3 -c "import tvm; print('TVM version:', tvm.__version__)"
# Test PyTorch: python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

ENTRYPOINT ["/bin/bash"]