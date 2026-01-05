#!/bin/bash
# VM Setup Script - Run this on the GCP VM after SSH
set -e

echo "=== arrwDB VM Setup ==="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y build-essential curl git python3.11 python3.11-venv python3.11-dev

# Install Rust
echo "Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Setup Python virtual environment
echo "Setting up Python environment..."
cd ~/arrwDB
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install maturin numpy

# Install arrwDB dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install arrwDB package
if [ -d "packages/arrwdb" ]; then
    pip install -e packages/arrwdb
fi

# Build Rust HNSW
echo "Building Rust HNSW..."
cd rust/indexes
maturin build --release
pip install target/wheels/*.whl
cd ~/arrwDB

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the benchmark:"
echo "  cd ~/arrwDB"
echo "  source venv/bin/activate"
echo "  ./benchmarks/cloud/run_cloud_benchmark.sh"
