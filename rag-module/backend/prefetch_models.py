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

"""One-time download of all IndicConformer STT and MMS-TTS models into the local
Hugging Face cache, so the backend never needs internet access at runtime."""

from server import (
    INDICCONFORMER_MODELS,
    MMS_TTS_MODELS,
    get_indicconformer_model,
    get_mms_tts_model,
    get_whisper_model,
)


def main():
    print("[STT] Fetching Whisper model (English)")
    get_whisper_model()

    for lang in INDICCONFORMER_MODELS:
        print(f"[STT] Fetching IndicConformer model: {lang}")
        get_indicconformer_model(lang)

    for lang in MMS_TTS_MODELS:
        print(f"[TTS] Fetching MMS-TTS model: {lang}")
        get_mms_tts_model(lang)

    print("All STT/TTS models are cached locally.")


if __name__ == "__main__":
    main()
