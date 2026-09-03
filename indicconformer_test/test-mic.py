import sounddevice as sd
import numpy as np
import onnx_asr

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

MODEL_ID = "OpenVoiceOS/ai4bharat-indicconformer-kn-onnx"


print("Loading IndicConformer Kannada model...")

model = onnx_asr.load_model(MODEL_ID)

print("Model loaded.")
print()
print(f"Speak for {RECORD_SECONDS} seconds...")
print("Recording...")

audio = sd.rec(
    int(RECORD_SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)

sd.wait()

print("Recording finished.")

audio = audio.squeeze()

print()
print("========== AUDIO ==========")
print(f"Samples : {len(audio)}")
print(f"Duration: {len(audio) / SAMPLE_RATE:.2f} sec")
print(f"Min    : {audio.min():.4f}")
print(f"Max    : {audio.max():.4f}")
print(f"RMS    : {np.sqrt(np.mean(audio ** 2)):.4f}")
print("============================")
print()

print("Running IndicConformer...")

result = model.recognize(audio)

print()
print("========== RESULT ==========")
print(result)
print("============================")