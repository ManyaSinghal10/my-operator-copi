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

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

echo "Downloading IndicConformer STT and MMS-TTS models from Hugging Face..."
echo "This is a one-time step; the backend runs fully offline afterward."

python3 backend/prefetch_models.py

echo "========================================="
echo "STT/TTS model download complete!"
echo "You can now start the server with ./start.sh"
echo "========================================="
