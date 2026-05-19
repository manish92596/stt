import asyncio, time, tempfile, os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
MIN_MS = 300
MIN_LEN = int((MIN_MS / 1000) * SAMPLE_RATE * BYTES_PER_SAMPLE)
MAX_BATCH = 32
MAX_DELAY_MS = 5.0
CONCURRENCY = 50
_executor = ThreadPoolExecutor(CONCURRENCY)

def bytes_to_tempfile(audio_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name

class CoalescingBatcher:
    def __init__(self, model):
        self.model = model
        self.pending = []
        self.lock = asyncio.Lock()
        self.flush_task = None

    async def enqueue(self, audio: bytes, language: str = "hi"):
        fut = asyncio.get_running_loop().create_future()
        async with self.lock:
            self.pending.append((audio, fut, language))
            if not self.flush_task or self.flush_task.done():
                self.flush_task = asyncio.create_task(self._flush())
        return await fut

    async def _flush(self):
        await asyncio.sleep(MAX_DELAY_MS / 1000)
        while True:
            async with self.lock:
                if not self.pending:
                    self.flush_task = None
                    return
                batch = self.pending[:MAX_BATCH]
                del self.pending[:len(batch)]

            tmp_files = [bytes_to_tempfile(x[0]) for x in batch]

            loop = asyncio.get_running_loop()
            t0 = time.time()
            print([x[2] for x in batch])
            try:
                try:
                    out = await loop.run_in_executor(_executor, partial(
                        self.model.transcribe, tmp_files,
                        lang_codes=[x[2] for x in batch],
                        tasks=['transcribe'] * len(tmp_files),
                        initial_prompts=[None] * len(tmp_files),
                        batch_size=min(24, len(tmp_files))
                    ))
                except Exception as e:
                    for _, fut, _ in batch:
                        if not fut.done():
                            fut.set_exception(e)
                    continue
            finally:
                for f in tmp_files:
                    try:
                        os.unlink(f)
                    except:
                        pass

            t1 = (time.time() - t0) * 1000
            for i, (_, f, _) in enumerate(batch):
                print(f"Batch {i} result: {out[i]}")
                text = out[i][0].get('text', '').strip() if isinstance(out[i][0], dict) else ''
                avg_logprob = out[i][0].get('avg_logprob', 0)
                word_timestamps = out[i][0].get('word_timestamps', [])
                good_prob = any(word['prob'] > 0.80 for word in word_timestamps)
                if not f.done():
                    f.set_result((text, t1, len(tmp_files), avg_logprob, good_prob))