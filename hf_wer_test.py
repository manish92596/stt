

# pip install datasets jiwer pandas soundfile



# nohup python kathbath_noisy.py > result_ASR-Benchmarking-Dataset_kathbath_noisy.log 2>&1 &



# tail -f result_ASR-Benchmarking-Dataset_kathbath_noisy.log

# pip install torchcodec



# pkill -f hf_wer_test.py










# from datasets import load_dataset
# from jiwer import wer
# import subprocess
# import json

# # Load dataset
# dataset = load_dataset(
#     "RinggAI/ASR-Benchmarking-Dataset",
#     "commonvoice",
#     split="eval"
# )

# # Optional: test only first N samples
# MAX_SAMPLES = 50

# total_wer = 0

# for i, row in enumerate(dataset):

#     if i >= MAX_SAMPLES:
#         break

#     audio_path = row["audio"]["path"]

#     reference = row["original_reference"]

#     print(f"\nSample {i+1}")
#     print("Reference:", reference)

#     # Call your STT API
#     result = subprocess.run([
#         "curl", "-s", "-X", "POST",
#         "http://localhost:6006/transcribe",
#         "-H", "api-key: any-key-works",
#         "-H", "model-id: /home/models/zero-stt-hinglish-ct2",
#         "-H", "language: hi",
#         "-F", f"audio_file=@{audio_path}"
#     ], capture_output=True, text=True)

#     try:
#         response = json.loads(result.stdout)
#         prediction = response.get("transcription", "")
#     except:
#         prediction = ""

#     print("Prediction:", prediction)

#     error = wer(reference, prediction)

#     total_wer += error

#     print(f"WER: {error * 100:.2f}%")

# avg_wer = (total_wer / MAX_SAMPLES) * 100

# print("\n" + "="*50)
# print(f"Average WER: {avg_wer:.2f}%")
# print("="*50)



























from datasets import load_dataset, Audio
from jiwer import wer
import subprocess
import json

# 1. Load dataset
dataset = load_dataset(
    "RinggAI/ASR-Benchmarking-Dataset",
    "commonvoice",
    split="eval"
)

# 2. CRITICAL FIX: Disable torchcodec auto-decoding to keep row["audio"]["path"] working
dataset = dataset.cast_column("audio", Audio(decode=False))

# Optional: test only first N samples
MAX_SAMPLES = 50

total_wer = 0

for i, row in enumerate(dataset):

    if i >= MAX_SAMPLES:
        break

    # This line now works flawlessly!
    audio_path = row["audio"]["path"]

    reference = row["original_reference"]

    print(f"\nSample {i+1}")
    print("Reference:", reference)

    # Call your STT API
    result = subprocess.run([
        "curl", "-s", "-X", "POST",
        "https://jarvislabs.net",
        "-H", "api-key: any-key-works",
        "-H", "model-id: /home/models/zero-stt-hinglish-ct2",
        "-H", "language: hi",
        "-F", f"audio_file=@{audio_path}"
    ], capture_output=True, text=True)

    try:
        response = json.loads(result.stdout)
        prediction = response.get("transcription", "")
    except:
        prediction = ""

    print("Prediction:", prediction)

    error = wer(reference, prediction)

    total_wer += error

    print(f"WER: {error * 100:.2f}%")

avg_wer = (total_wer / MAX_SAMPLES) * 100

print("\n" + "="*50)
print(f"Average WER: {avg_wer:.2f}%")
print("="*50)






























from datasets import load_dataset, Audio
from jiwer import wer
import subprocess
import json
import os
import tempfile
import soundfile as sf

# 1. Load dataset — decode=True to get actual audio arrays
dataset = load_dataset(
    "RinggAI/ASR-Benchmarking-Dataset",
    "commonvoice",
    split="eval"
)
dataset = dataset.cast_column("audio", Audio(decode=True, sampling_rate=16000))

MAX_SAMPLES = 50
total_wer = 0
failed = 0

STT_URL = "https://481d1d4174781.notebooksn.jarvislabs.net/transcribe"  # Use localhost if running on same machine
                                               # or your external URL if remote

for i, row in enumerate(dataset):
    if i >= MAX_SAMPLES:
        break

    audio_array = row["audio"]["array"]
    sample_rate = row["audio"]["sampling_rate"]
    reference = row["original_reference"]

    print(f"\nSample {i+1}")
    print("Reference:", reference)

    # Write audio to a real temp file curl can access
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio_array, sample_rate)

    try:
        result = subprocess.run([
            "curl", "-s", "--max-time", "30",  # 30s timeout
            "-X", "POST", STT_URL,
            "-H", "api-key: any-key-works",
            "-H", "model-id: /home/models/zero-stt-hinglish-ct2",
            "-H", "language: hi",
            "-F", f"audio_file=@{tmp_path}"
        ], capture_output=True, text=True)

        response = json.loads(result.stdout)
        prediction = response.get("transcription", "").strip()

        if not prediction:
            print("WARNING: Empty transcription returned")
            print("stderr:", result.stderr)
            failed += 1
            continue

    except (json.JSONDecodeError, Exception) as e:
        print(f"ERROR on sample {i+1}: {e}")
        print("stdout:", result.stdout[:200])
        failed += 1
        continue
    finally:
        os.unlink(tmp_path)  # Clean up temp file

    print("Prediction:", prediction)
    error = wer(reference, prediction)
    total_wer += error
    print(f"WER: {error * 100:.2f}%")

successful = MAX_SAMPLES - failed
avg_wer = (total_wer / successful * 100) if successful > 0 else float('inf')

print("\n" + "="*50)
print(f"Processed: {successful}/{MAX_SAMPLES} samples ({failed} failed)")
print(f"Average WER: {avg_wer:.2f}%")
print("="*50)









































from datasets import load_dataset, Audio
from jiwer import wer
import subprocess
import json
import os
import tempfile
import soundfile as sf

# 1. Load dataset
dataset = load_dataset(
    "RinggAI/ASR-Benchmarking-Dataset",
    "commonvoice",
    split="eval"
)
dataset = dataset.cast_column("audio", Audio(decode=True, sampling_rate=16000))

total_wer = 0
failed = 0
total_samples = len(dataset)

STT_URL = "https://481d1d4174781.notebooksn.jarvislabs.net/transcribe"

for i, row in enumerate(dataset):
    audio_array = row["audio"]["array"]
    sample_rate = row["audio"]["sampling_rate"]
    reference = row["original_reference"]

    print(f"\nSample {i+1}/{total_samples}")
    print("Reference:", reference)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio_array, sample_rate)

    try:
        result = subprocess.run([
            "curl", "-s", "--max-time", "30",
            "-X", "POST", STT_URL,
            "-H", "api-key: any-key-works",
            "-H", "model-id: /home/models/zero-stt-hinglish-ct2",
            "-H", "language: hi",
            "-F", f"audio_file=@{tmp_path}"
        ], capture_output=True, text=True)

        response = json.loads(result.stdout)
        prediction = response.get("transcription", "").strip()

        if not prediction:
            print("WARNING: Empty transcription returned")
            print("stderr:", result.stderr)
            failed += 1
            continue

    except (json.JSONDecodeError, Exception) as e:
        print(f"ERROR on sample {i+1}: {e}")
        print("stdout:", result.stdout[:200])
        failed += 1
        continue
    finally:
        os.unlink(tmp_path)

    print("Prediction:", prediction)
    error = wer(reference, prediction)
    total_wer += error
    print(f"WER: {error * 100:.2f}%")

successful = i + 1 - failed
avg_wer = (total_wer / successful * 100) if successful > 0 else float('inf')

print("\n" + "="*50)
print(f"Processed: {successful}/{total_samples} samples ({failed} failed)")
print(f"Average WER: {avg_wer:.2f}%")
print("="*50)




































from datasets import load_dataset, Audio
from jiwer import wer
import subprocess
import json
import os
import tempfile
import soundfile as sf

# 1. Load dataset
dataset = load_dataset(
    "RinggAI/ASR-Benchmarking-Dataset",
    "kathbath",
    split="eval"
)
dataset = dataset.cast_column("audio", Audio(decode=True, sampling_rate=16000))

total_wer = 0
failed = 0
total_samples = len(dataset)

STT_URL = "https://481d1d4174781.notebooksn.jarvislabs.net/transcribe"

for i, row in enumerate(dataset):
    audio_array = row["audio"]["array"]
    sample_rate = row["audio"]["sampling_rate"]
    reference = row["original_reference"]

    print(f"\nSample {i+1}/{total_samples}")
    print("Reference:", reference)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio_array, sample_rate)

    try:
        result = subprocess.run([
            "curl", "-s", "--max-time", "30",
            "-X", "POST", STT_URL,
            "-H", "api-key: any-key-works",
            "-H", "model-id: /home/models/zero-stt-hinglish-ct2",
            "-H", "language: hi",
            "-F", f"audio_file=@{tmp_path}"
        ], capture_output=True, text=True)

        response = json.loads(result.stdout)
        prediction = response.get("transcription", "").strip()

        if not prediction:
            print("WARNING: Empty transcription returned")
            print("stderr:", result.stderr)
            failed += 1
            continue

    except (json.JSONDecodeError, Exception) as e:
        print(f"ERROR on sample {i+1}: {e}")
        print("stdout:", result.stdout[:200])
        failed += 1
        continue
    finally:
        os.unlink(tmp_path)

    print("Prediction:", prediction)
    error = wer(reference, prediction)
    total_wer += error
    print(f"WER: {error * 100:.2f}%")

successful = i + 1 - failed
avg_wer = (total_wer / successful * 100) if successful > 0 else float('inf')

print("\n" + "="*50)
print(f"Processed: {successful}/{total_samples} samples ({failed} failed)")
print(f"Average WER: {avg_wer:.2f}%")
print("="*50)