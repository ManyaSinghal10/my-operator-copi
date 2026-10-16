#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

cd "$(dirname "$0")"

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing requirements..."


echo "Checking Jetson environment..."

if [ -e /etc/nv_tegra_release ]; then
    echo "Jetson detected."

    PYTHON_VERSION=$(python -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

    if [ "$PYTHON_VERSION" != "3.10" ]; then
        echo "Error: Expected Python 3.10, found $PYTHON_VERSION"
        exit 1
    fi

    echo "Installing system dependencies for PyTorch..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-pip \
        libopenblas-dev

    echo "Installing compatible NumPy..."
    python -m pip install "numpy==1.26.4"

    echo "Installing Jetson-compatible PyTorch..."
    python -m pip install \
        --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
        "torch==2.8.0"

    echo "Verifying PyTorch GPU availability..."
    python -c '
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access the GPU")
'

    echo "Installing remaining application dependencies..."
    grep -v '^torch$' backend/requirements.txt |
        python -m pip install --extra-index-url https://pypi.org/simple/ -r /dev/stdin

else
    echo "Non-Jetson system detected."
    python -m pip install -r backend/requirements.txt
fi

echo "========================================="
echo "Setup complete!"
if [ -n "$JETSON_DETECTED" ]; then
    echo "Jetson detected: install the JetPack-matched PyTorch wheel manually —"
    echo "see the comment above 'torch' in backend/requirements.txt for the URL."
fi
echo "Run ./download_model.sh to download the model."
echo "Run ./start.sh to start the servers."
echo "========================================="
