import json
import sys
import numpy as np
import librosa
import onnxruntime as ort

from huggingface_hub import hf_hub_download


REPO = "sulabhkatiyar/indicconformer-120m-onnx"


LANGUAGES = {
    "hi": "Hindi",
    "kn": "Kannada",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}


def load_model(language):

    if language not in LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}\n"
            f"Supported: {list(LANGUAGES.keys())}"
        )

    print(f"\nLoading IndicConformer: {LANGUAGES[language]} ({language})")

    model_path = hf_hub_download(
        REPO,
        f"{language}/model.onnx"
    )

    vocab_path = hf_hub_download(
        REPO,
        f"{language}/vocab.json"
    )

    print("Model:", model_path)
    print("Vocabulary:", vocab_path)

    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    session = ort.InferenceSession(
        model_path,
        providers=providers
    )

    print("Execution providers:")
    print(session.get_providers())

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return session, vocab


def preprocess_audio(audio_path):

    print("\nLoading audio...")

    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    print("Sample rate:", sr)
    print("Samples:", len(audio))
    print("Duration:", len(audio) / sr)

    # Pre-emphasis
    audio_pe = np.concatenate([
        audio[:1],
        audio[1:] - 0.97 * audio[:-1]
    ])

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio_pe,
        sr=16000,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmin=0,
        fmax=8000,
        norm="slaney",
        power=2.0,
    )

    # Log mel
    log_mel = np.log(
        mel + 2 ** -24
    ).astype(np.float32)

    # Normalize
    mean = log_mel.mean(
        axis=1,
        keepdims=True
    )

    std = log_mel.std(
        axis=1,
        ddof=1,
        keepdims=True
    ) + 1e-5

    log_mel = (
        log_mel - mean
    ) / std

    # [1, 80, T]
    mel_batch = log_mel[
        np.newaxis,
        :,
        :
    ].astype(np.float32)

    mel_length = np.array(
        [mel_batch.shape[2]],
        dtype=np.int64
    )

    return mel_batch, mel_length


def ctc_decode(logits, vocab):

    # logits:
    # [batch, time, vocab + blank]

    token_ids = np.argmax(
        logits,
        axis=-1
    )[0]

    blank_id = len(vocab)

    decoded_tokens = []

    previous = None

    for token_id in token_ids:

        # CTC blank
        if token_id == blank_id:
            previous = None
            continue

        # Remove repeated tokens
        if token_id == previous:
            continue

        if token_id < len(vocab):
            decoded_tokens.append(
                vocab[token_id]
            )

        previous = token_id

    text = "".join(decoded_tokens)

    # SentencePiece-style word boundary
    text = text.replace("▁", " ")

    return text.strip()


def transcribe(audio_path, language):

    session, vocab = load_model(language)

    mel, length = preprocess_audio(
        audio_path
    )

    print("\nRunning inference...")

    inputs = {
        "audio_signal": mel,
        "length": length,
    }

    outputs = session.run(
        None,
        inputs
    )

    logits = outputs[0]

    print("Logits shape:", logits.shape)

    text = ctc_decode(
        logits,
        vocab
    )

    return text


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python test_indicconformer.py "
            "<audio.wav> <language>"
        )

        print("\nExample:")
        print(
            "python test_indicconformer.py "
            "kannada.wav kn"
        )

        sys.exit(1)

    audio_file = sys.argv[1]
    language = sys.argv[2]

    result = transcribe(
        audio_file,
        language
    )

    print("\n")
    print("=" * 60)
    print("INDICCONFORMER RESULT")
    print("=" * 60)
    print("Language :", LANGUAGES[language])
    print("Audio    :", audio_file)
    print("Text     :", result)
    print("=" * 60)