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

"""One-time export of the MMS-TTS (VITS) checkpoints to ONNX.

This is the only step that needs PyTorch, `transformers` and `onnx`; run it on a
machine that has them (and internet or a warm Hugging Face cache). The resulting
`backend/tts_onnx/<lang>/` folders are all the server needs at runtime, together
with `onnxruntime` and `numpy`.

    python3 backend/export_mms_tts_onnx.py            # all languages
    python3 backend/export_mms_tts_onnx.py hi kn      # selected languages
"""

import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_ONNX_DIR = os.path.join(BASE_DIR, "tts_onnx")

MMS_TTS_MODELS = {
    "en": "facebook/mms-tts-eng",
    "hi": "facebook/mms-tts-hin",
    "mr": "facebook/mms-tts-mar",
    "kn": "facebook/mms-tts-kan",
    "ta": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
    "bn": "facebook/mms-tts-ben",
    "gu": "facebook/mms-tts-guj",
}


def _patch_relative_embeddings():
    """Make VITS relative-position slicing trace-safe.

    The stock `_get_relative_embeddings` uses Python `max()` on the sequence
    length, which the ONNX tracer would freeze to the example input's length.
    This version is mathematically identical (zero-padded window gather) but
    built from tensor ops so it works for any input length.
    """
    import torch
    from transformers.models.vits import modeling_vits

    def _get_relative_embeddings(self, relative_embeddings, length):
        window = self.window_size
        offsets = torch.arange(2 * length - 1, device=relative_embeddings.device) - (length - 1)
        idx = offsets + window
        valid = ((idx >= 0) & (idx <= 2 * window)).to(relative_embeddings.dtype)
        gathered = relative_embeddings[:, idx.clamp(0, 2 * window)]
        return gathered * valid.view(1, -1, 1)

    modeling_vits.VitsAttention._get_relative_embeddings = _get_relative_embeddings


def export_language(lang):
    import torch
    from transformers import VitsModel, AutoTokenizer

    model_id = MMS_TTS_MODELS[lang]
    out_dir = os.path.join(TTS_ONNX_DIR, lang)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[export] {lang}: loading {model_id}")
    model = VitsModel.from_pretrained(model_id).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    class Wrapper(torch.nn.Module):
        def __init__(self, vits):
            super().__init__()
            self.vits = vits

        def forward(self, input_ids):
            attention_mask = torch.ones_like(input_ids)
            return self.vits(input_ids=input_ids, attention_mask=attention_mask).waveform

    # Example input built from real vocab symbols (a fixed English word would be
    # filtered to nothing by non-Latin tokenizers). Length is dynamic in the graph.
    vocab = tokenizer.get_vocab()
    symbols = [i for tok, i in sorted(vocab.items(), key=lambda kv: kv[1]) if len(tok) == 1][:12]
    blanked = [0] * (len(symbols) * 2 + 1)
    blanked[1::2] = symbols
    sample = torch.tensor([blanked], dtype=torch.long)
    onnx_path = os.path.join(out_dir, "model.onnx")
    with torch.no_grad():
        torch.onnx.export(
            Wrapper(model),
            (sample,),
            onnx_path,
            input_names=["input_ids"],
            output_names=["waveform"],
            dynamic_axes={"input_ids": {1: "seq"}, "waveform": {1: "samples"}},
            opset_version=17,
            dynamo=False,
        )

    # Everything the runtime tokenizer needs (no `transformers` at runtime).
    vocab_src = tokenizer.get_vocab()
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_src, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "sampling_rate": int(model.config.sampling_rate),
                "add_blank": bool(tokenizer.add_blank),
                "normalize": bool(tokenizer.normalize),
                "pad_id": int(vocab_src.get("<pad>", 0)),
            },
            f,
        )
    print(f"[export] {lang}: wrote {out_dir}")
    return model, tokenizer, onnx_path


def verify_language(lang, model, tokenizer, onnx_path):
    """Run the ONNX graph on several lengths and compare to PyTorch."""
    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    texts = {
        "en": ["hi", "this machine requires high voltage for functioning"],
        "hi": ["नमस्ते", "यह मशीन चलने के लिए उच्च वोल्टेज की आवश्यकता है"],
        "mr": ["नमस्कार", "या यंत्राला चालण्यासाठी उच्च व्होल्टेज लागते"],
        "kn": ["ನಮಸ್ಕಾರ", "ಈ ಯಂತ್ರಕ್ಕೆ ಹೆಚ್ಚಿನ ವೋಲ್ಟೇಜ್ ಬೇಕು"],
        "ta": ["வணக்கம்", "இந்த இயந்திரத்திற்கு உயர் மின்னழுத்தம் தேவை"],
        "te": ["నమస్కారం", "ఈ యంత్రానికి అధిక వోల్టేజ్ అవసరం"],
        "bn": ["নমস্কার", "এই যন্ত্রটির জন্য উচ্চ ভোল্টেজ প্রয়োজন"],
        "gu": ["નમસ્તે", "આ મશીનને ઉચ્ચ વોલ્ટેજની જરૂર છે"],
    }[lang]
    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            ref = model(**tokenizer(text, return_tensors="pt")).waveform.squeeze().numpy()
        out = sess.run(None, {"input_ids": ids.numpy().astype(np.int64)})[0].squeeze()
        ratio = len(out) / max(len(ref), 1)
        print(
            f"[verify] {lang} len(ids)={ids.shape[1]:3d} "
            f"torch={len(ref)} onnx={len(out)} ratio={ratio:.2f} "
            f"rms_torch={np.sqrt(np.mean(ref ** 2)):.3f} rms_onnx={np.sqrt(np.mean(out ** 2)):.3f}"
        )


def main():
    langs = sys.argv[1:] or list(MMS_TTS_MODELS)
    unknown = [l for l in langs if l not in MMS_TTS_MODELS]
    if unknown:
        raise SystemExit(f"Unknown language(s): {unknown}. Known: {list(MMS_TTS_MODELS)}")

    _patch_relative_embeddings()
    for lang in langs:
        model, tokenizer, onnx_path = export_language(lang)
        verify_language(lang, model, tokenizer, onnx_path)


if __name__ == "__main__":
    main()
