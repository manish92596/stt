






# pip install jiwer

import subprocess
import json
import sys
from jiwer import wer

# Define your test cases — add your own audio files and expected transcriptions
test_cases = [
    {
        "audio": "/home/demo.wav",
        "expected": "जी हमारी team आप से contact करके demo arrange करेगी क्या मैं आपकी किसी और तरह से help कर सकती हूँ"  # what you EXPECT the model to say
    },
    # Add more test cases:
    {"audio": "/home/1.wav", "expected": "हाँ madam, बोलो।"},
    {"audio": "/home/2.wav", "expected": "आधे घंटे से"},
    {"audio": "/home/3.wav", "expected": "Control Tower से बोल रहे हैं, ट्रक नंबर T1234 के ड्राइवर से बात हो रही है?"},
    {"audio": "/home/a1.wav", "expected": "Rest कर रहा हूँ अभी"},
    {"audio": "/home/a2.wav", "expected": "वाहन कब से रुका हुआ है?"},
    {"audio": "/home/a3.wav", "expected": "पैंतालीस मिनट से"},
    {"audio": "/home/a4.wav", "expected": "MP Nagar"},
    {"audio": "/home/a5.wav", "expected": "नहीं तो ग्राहक तक पहुंचने में और delay हो सकता है।"},
    {"audio": "/home/a6.wav", "expected": "नहीं नहीं ये सब तो सही है।"},
    {"audio": "/home/a7.wav", "expected": "जी मैं अभी eleventh में हूं।"},
    {"audio": "/home/a8.wav", "expected": "अ जी मैं अभी PCM पढ़ रहा हूँ।"},
    {"audio": "/home/a9.wav", "expected": "बिल्कुल। तो कब convenient रहेगा? कल morning या शाम?"},
    {"audio": "/home/a10.wav", "expected": "जी परसो morning 10:00 AM सही है।"},
    {"audio": "/home/a11.wav", "expected": "9993337118"}
]

def transcribe(audio_path):
    result = subprocess.run([
        "curl", "-s", "-X", "POST", "http://localhost:6006/transcribe",
        "-H", "api-key: any-key-works",
        "-H", f"model-id: /home/models/zero-stt-hinglish-ct2",
        "-H", "language: hi",
        "-F", f"audio_file=@{audio_path}"
    ], capture_output=True, text=True)
    
    try:
        data = json.loads(result.stdout)
        return data.get("transcription", "")
    except:
        print(f"Error: {result.stdout}")
        return ""

total_wer = 0
results = []

for i, test in enumerate(test_cases):
    print(f"\nTest {i+1}: {test['audio']}")
    
    hypothesis = transcribe(test['audio'])
    reference = test['expected']
    
    error_rate = wer(reference, hypothesis)
    total_wer += error_rate
    
    results.append({
        "audio": test['audio'],
        "expected": reference,
        "got": hypothesis,
        "wer": round(error_rate * 100, 2)
    })
    
    print(f"Expected : {reference}")
    print(f"Got      : {hypothesis}")
    print(f"WER      : {error_rate * 100:.2f}%")

avg_wer = (total_wer / len(test_cases)) * 100
print(f"\n{'='*50}")
print(f"Average WER: {avg_wer:.2f}%")
print(f"{'='*50}")  





