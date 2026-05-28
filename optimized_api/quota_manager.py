import os
import time
from typing import Dict, Any

REFRESH_INTERVAL_SECONDS = 5 * 60
REFRESH_INTERVAL_ALL_MODELS_SECONDS = 60 * 5
FLUSH_INTERVAL_SECONDS = 60 * 5

class GPUQuotaManager:
    def __init__(self):
        self.user_caches = {}
        self.user_last_usage_record = {}
        self.user_pending_usage = {}
        self.all_existing_models = []
        self.all_existing_models_last_update = None
        self.users_data = {}
        self.api_base_url = os.getenv("API_BASE_URL", "")
        self.private_secured_key = os.getenv("PRIVATE_SECURED_KEY", "")
        self.batchers = {}

    def can_transcribe(self, api_key: str, duration_ms: int) -> bool:
        if api_key not in self.user_caches:
            if api_key in self.users_data:
                self.user_caches[api_key] = self.users_data[api_key]
            else:
                self.user_caches[api_key] = {
                    "available_milliseconds": 999999999,
                    "models": [
                        {"model_id": m, "hugging_face_repo_id": m}
                        for m in [
                            "/home/inference/models/zero-stt-hinglish-ct2",
                            "/home/models/zero-stt-hinglish-ct2",
                            "shunyalabs/zero-stt-hinglish",
                            "openai/whisper-large-v3-turbo",
                            "openai/whisper-large-v3",
                            "openai/whisper-medium",
                            "openai/whisper-small",
                        ]
                    ]
                }
        return self.user_caches[api_key]["available_milliseconds"] >= duration_ms

    def consume_quota(self, api_key: str, duration_ms: int):
        if api_key in self.user_caches:
            self.user_caches[api_key]["available_milliseconds"] -= duration_ms

    def record_usage(self, api_key: str, cost: float, duration_seconds: float, latency_ms: int, requested_at: float, fallbacked: bool = False, streaming: bool = False):
        if api_key not in self.user_pending_usage:
            self.user_pending_usage[api_key] = []
        self.user_pending_usage[api_key].append({
            "cost": cost,
            "duration_seconds": duration_seconds,
            "infer_latency": latency_ms,
            "requested_at": requested_at,
            "fallbacked": fallbacked,
            "streaming": streaming
        })

    def _flush_usage_records(self):
        pass

    def get_available_milliseconds(self, api_key: str) -> int:
        if api_key not in self.user_caches:
            if api_key in self.users_data:
                self.user_caches[api_key] = self.users_data[api_key]
            else:
                return 999999999
        return self.user_caches[api_key].get("available_milliseconds", 0)

    def get_all_existing_models(self) -> list:
        return self.all_existing_models

    def get_batcher(self, repo_id: str):
        if repo_id in self.batchers:
            return self.batchers[repo_id]
        from model_manager import get_model
        from batcher import CoalescingBatcher
        model = get_model(repo_id)
        self.batchers[repo_id] = CoalescingBatcher(model)
        return self.batchers[repo_id]