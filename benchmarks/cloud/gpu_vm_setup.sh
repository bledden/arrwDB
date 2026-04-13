#!/bin/bash
# GPU VM Setup — installs CUDA, conda, faiss-gpu-cuvs, and arrwDB deps
set -e

echo "=== arrwDB GPU VM Setup ==="

# Check for NVIDIA GPU
if ! nvidia-smi &>/dev/null; then
    echo "NVIDIA driver not detected. Installing..."
    sudo apt-get update
    sudo apt-get install -y nvidia-driver-550
    echo ""
    echo "Driver installed. You MUST reboot now:"
    echo "  sudo reboot"
    echo "Then re-run this script after reboot."
    exit 0
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install system deps
sudo apt-get update
sudo apt-get install -y build-essential curl git python3.11 python3.11-venv python3.11-dev wget

# Install Miniconda (needed for faiss-gpu-cuvs)
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
fi

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Create conda env with faiss-gpu-cuvs
echo "Creating conda environment with faiss-gpu-cuvs..."
conda create -n arrwdb-gpu python=3.11 -y
conda activate arrwdb-gpu

conda install -c pytorch -c nvidia -c rapidsai -c conda-forge \
    faiss-gpu-cuvs numpy h5py -y

# Install arrwDB Python deps (skip Rust build — not needed for GPU benchmark)
cd ~/arrwDB
pip install requests

echo ""
echo "=== GPU Setup Complete ==="
echo ""
echo "To run GPU benchmarks:"
echo "  eval \"\$(~/miniconda3/bin/conda shell.bash hook)\""
echo "  conda activate arrwdb-gpu"
echo "  cd ~/arrwDB"
echo "  chmod +x benchmarks/cloud/launch_gpu_bench.sh"
echo "  ./benchmarks/cloud/launch_gpu_bench.sh"
