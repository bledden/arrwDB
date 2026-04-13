#!/bin/bash
# Create GCP GPU VM for CAGRA benchmarks
#
# Uses NVIDIA T4 (16GB VRAM) — sufficient for 1M vectors up to 1024d
# Cost: ~$0.35/hr on-demand
#
# Usage:
#   export GCP_PROJECT_ID=your-project-id
#   ./gcp_gpu_setup.sh
#
set -e

PROJECT_ID="${GCP_PROJECT_ID:-}"
ZONE="${GCP_ZONE:-us-east1-c}"
INSTANCE_NAME="arrwdb-gpu-benchmark"
MACHINE_TYPE="n1-highmem-8"  # 8 vCPU, 52GB RAM
ACCELERATOR="nvidia-tesla-t4"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: Set GCP_PROJECT_ID environment variable"
    exit 1
fi

echo "=== arrwDB GPU Benchmark Setup ==="
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Machine: $MACHINE_TYPE + $ACCELERATOR"
echo ""

gcloud config set project "$PROJECT_ID"

echo "Creating GPU VM..."
gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$ACCELERATOR,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True" \
    --tags=arrwdb-benchmark

echo "Waiting for instance..."
sleep 30

EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "Instance IP: $EXTERNAL_IP"

echo ""
echo "=== VM Created ==="
echo ""
echo "Next steps:"
echo "1. SSH in:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "2. Install NVIDIA drivers (if not auto-installed):"
echo "   sudo apt-get install -y nvidia-driver-550"
echo "   sudo reboot"
echo ""
echo "3. Clone repo and run GPU setup:"
echo "   git clone https://github.com/bledden/arrwDB.git"
echo "   cd arrwDB"
echo "   chmod +x benchmarks/cloud/gpu_vm_setup.sh"
echo "   ./benchmarks/cloud/gpu_vm_setup.sh"
echo ""
echo "4. Launch GPU benchmarks:"
echo "   ./benchmarks/cloud/launch_gpu_bench.sh"
