"""Stream a raw PCM file at real-time pace to the /stream WebSocket endpoint
and print every transcription event the server returns.

Generate the PCM first, e.g.:
    ffmpeg -y -i demo.wav -f s16le -acodec pcm_s16le -ac 1 -ar 16000 demo_16k.pcm

Then run:
    python ws_file_test.py \
        --url ws://localhost:6006/stream \
        --pcm /home/inference/demo_16k.pcm \
        --sample-rate 16000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets


async def stream_pcm(args: argparse.Namespace) -> None:
    headers = [
        ("api-key", args.api_key),
        ("model-id", args.model_id),
        ("sample-rate", str(args.sample_rate)),
        ("vad-threshold", str(args.vad_threshold)),
        ("min-silence-duration", str(args.min_silence_duration)),
        ("stt-interval-ms", str(args.stt_interval_ms)),
        ("language", args.language),
    ]

    bytes_per_sample = 2  # int16 mono
    chunk_bytes = int(args.sample_rate * bytes_per_sample * (args.chunk_ms / 1000.0))
    chunk_interval = args.chunk_ms / 1000.0

    with open(args.pcm, "rb") as f:
        pcm = f.read()
    total_seconds = len(pcm) / (args.sample_rate * bytes_per_sample)
    print(f"PCM loaded: {len(pcm)} bytes = {total_seconds:.2f}s "
          f"@ {args.sample_rate} Hz | chunk={chunk_bytes}B every {args.chunk_ms}ms")

    async with websockets.connect(args.url, max_size=None, additional_headers=headers) as ws:
        print(f"Connected to {args.url}")

        async def sender() -> None:
            t0 = time.time()
            for offset in range(0, len(pcm), chunk_bytes):
                chunk = pcm[offset:offset + chunk_bytes]
                await ws.send(chunk)
                # pace at real time so server-side VAD behaves like a live mic
                target = t0 + (offset + chunk_bytes) / (args.sample_rate * bytes_per_sample)
                sleep_for = target - time.time()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            print("[sender] PCM fully sent, holding for trailing silence...")
            await asyncio.sleep(2.0)
            try:
                await ws.send("close")
            except Exception:
                pass

        async def receiver() -> None:
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        print(f"[recv] {len(raw)} bytes (binary)")
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"[recv-text] {raw}")
                        continue
                    print(f"[recv] {msg}")
            except websockets.exceptions.ConnectionClosed as exc:
                print(f"[recv] connection closed: {exc.code} {exc.reason}")

        await asyncio.gather(sender(), receiver(), return_exceptions=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:6006/stream")
    parser.add_argument("--pcm", required=True, help="path to raw int16 mono PCM file")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-ms", type=int, default=100, help="size of each WS send")
    parser.add_argument("--api-key", default="any-key-works")
    parser.add_argument("--model-id", default="/home/inference/models/zero-stt-hinglish-ct2")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--vad-threshold", type=float, default=0.3)
    parser.add_argument("--min-silence-duration", type=float, default=0.3)
    parser.add_argument("--stt-interval-ms", type=int, default=200)
    args = parser.parse_args()
    asyncio.run(stream_pcm(args))


if __name__ == "__main__":
    main()
