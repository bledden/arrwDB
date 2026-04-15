#!/bin/bash
# AWS r6i.16xlarge benchmark — matches ann-benchmarks.com hardware
#
# This runs arrwDB on the same hardware used by ann-benchmarks.com:
#   r6i.16xlarge: Intel Xeon Ice Lake, 64 vCPU, 512GB RAM
#   Single-threaded search, hyperthreading disabled
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - SSH key pair created in the target region
#
# Usage:
#   export AWS_REGION=us-east-1
#   export AWS_KEY_NAME=your-key-pair
#   ./aws_benchmark.sh
#
set -e

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="r6i.16xlarge"
KEY_NAME="${AWS_KEY_NAME:-}"
# Find latest Ubuntu 22.04 AMI for the target region
AMI=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text 2>/dev/null)

if [ -z "$AMI" ] || [ "$AMI" = "None" ]; then
    echo "Error: Could not find Ubuntu 22.04 AMI in $REGION"
    exit 1
fi
echo "AMI: $AMI"

if [ -z "$KEY_NAME" ]; then
    echo "Error: Set AWS_KEY_NAME environment variable"
    exit 1
fi

echo "=== arrwDB AWS Benchmark ==="
echo "Region: $REGION"
echo "Instance: $INSTANCE_TYPE (Intel Xeon Ice Lake, 64 vCPU, 512GB)"
echo "Key: $KEY_NAME"
echo ""
echo "Estimated cost: ~\$4/hr, ~\$8 total for full benchmark"
echo ""

# Create instance
echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --instance-type "$INSTANCE_TYPE" \
    --image-id "$AMI" \
    --key-name "$KEY_NAME" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance: $INSTANCE_ID"
echo "Waiting for running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Public IP: $PUBLIC_IP"
echo ""
echo "=== Instance Ready ==="
echo ""
echo "Next steps:"
echo "1. SSH in:"
echo "   ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "2. Run setup:"
echo "   git clone https://github.com/bledden/arrwDB.git"
echo "   cd arrwDB"
echo "   sudo apt-get update && sudo apt-get install -y build-essential python3.11 python3.11-venv python3.11-dev"
echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
echo "   source ~/.cargo/env"
echo "   python3.11 -m venv venv && source venv/bin/activate"
echo "   pip install maturin numpy h5py"
echo "   cd rust/indexes && maturin build --release"
echo "   pip install target/wheels/*.whl"
echo ""
echo "3. Run benchmark:"
echo "   cd ~/arrwDB"
echo "   python benchmarks/vectordbbench/run_fast_hnsw_benchmark.py \\"
echo "     --dataset all --M 48 --ef-construction 400 --fast-only \\"
echo "     --output-dir /tmp/aws_results/"
echo ""
echo "4. When done, terminate:"
echo "   aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
