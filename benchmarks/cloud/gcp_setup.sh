#!/bin/bash
# GCP Cloud Benchmark Setup for arrwDB
# Run this locally to create and configure a GCP VM for benchmarking

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
ZONE="${GCP_ZONE:-us-central1-b}"
INSTANCE_NAME="arrwdb-benchmark-highmem"
MACHINE_TYPE="n2-highmem-16"  # 16 vCPU, 128GB RAM - matches competition hardware

if [ -z "$PROJECT_ID" ]; then
    echo "Error: Set GCP_PROJECT_ID environment variable"
    echo "  export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

echo "=== arrwDB GCP Benchmark Setup ==="
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Machine: $MACHINE_TYPE (16 vCPU, 128GB)"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not installed"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

# Create VM instance
echo "Creating VM instance..."
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --tags=arrwdb-benchmark

# Wait for instance to be ready
echo "Waiting for instance to be ready..."
sleep 30

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "Instance IP: $EXTERNAL_IP"

# Copy setup script to VM
echo "Copying setup script to VM..."
gcloud compute scp benchmarks/cloud/vm_setup.sh $INSTANCE_NAME:~/vm_setup.sh --zone=$ZONE

# Copy arrwDB code to VM
echo "Copying arrwDB code to VM..."
gcloud compute scp --recurse . $INSTANCE_NAME:~/arrwDB --zone=$ZONE \
    --exclude=".git" \
    --exclude="venv" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude="data" \
    --exclude="target"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. SSH into the VM:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "2. Run the setup script:"
echo "   chmod +x ~/vm_setup.sh && ~/vm_setup.sh"
echo ""
echo "3. Run the benchmark:"
echo "   cd ~/arrwDB && ./benchmarks/cloud/run_cloud_benchmark.sh"
echo ""
echo "4. When done, delete the VM:"
echo "   gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
