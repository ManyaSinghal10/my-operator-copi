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
# Note: this file isn't hash-pinned (no per-package --hash entries), so
# --require-hashes is intentionally not used here.
if [ -e /etc/nv_tegra_release ]; then
    # Jetson (JetPack): PyPI's `torch` has no Tegra CUDA support, so it's
    # stripped here in favor of NVIDIA's JetPack-matched wheel (see the
    # comment above `torch` in backend/requirements.txt for the install URL).
    echo "Jetson host detected; excluding plain PyPI torch from install."
    grep -v '^torch$' backend/requirements.txt | pip install --extra-index-url https://pypi.org/simple/ -r /dev/stdin
    JETSON_DETECTED=1
else
    pip install --extra-index-url https://pypi.org/simple/ -r backend/requirements.txt
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
