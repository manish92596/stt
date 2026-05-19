import os
from huggingface_hub import snapshot_download
import whisper_s2t
from whisper_s2t.backends.ctranslate2.model import BEST_ASR_CONFIG

CACHE = {}
BEST_ASR_CONFIG['word_timestamps'] = True

def get_model(repo_id: str):
    if repo_id in CACHE:
        return CACHE[repo_id]

    if os.path.isdir(repo_id):
        local_dir = repo_id
        print(f"Loading model from local path: {local_dir}")
    else:
        local_dir = f"./models/{repo_id.replace('/', '_')}"
        os.makedirs(local_dir, exist_ok=True)
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        print(f"Downloading model {repo_id}...")
        if hf_token:
            snapshot_download(repo_id=repo_id, local_dir=local_dir, token=hf_token)
        else:
            snapshot_download(repo_id=repo_id, local_dir=local_dir)

    print("Loading CTranslate2 model...")
    model = whisper_s2t.load_model(
        local_dir,
        backend='CTranslate2',
        device='cuda',
        compute_type='float16',
        n_mels=128,
        asr_options=BEST_ASR_CONFIG
    )
    CACHE[repo_id] = model
    print(f"Model ready.")
    return model