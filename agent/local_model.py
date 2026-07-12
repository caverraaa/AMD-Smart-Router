"""Gemma-2-2B-it via llama-cpp, CPU-only, serialized behind a lock.

llama.cpp contexts are not thread-safe: exactly one generation runs at a
time; the Fireworks worker pool is unaffected. Gemma has no system role,
so callers pass a single merged user message. Generation time is bounded
by construction via small max_tokens caps (no kill-timer exists in
llama-cpp) — the speed spike validates the worst case fits 30 s/response.
"""
import os
import threading

DEFAULT_MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH", "/app/models/gemma-2-2b-it-Q4_K_M.gguf")
LOCAL_MAX_TOKENS = 160
CLASSIFY_MAX_TOKENS = 8


class LocalModel:
    def __init__(self, path=DEFAULT_MODEL_PATH, llama_factory=None, n_threads=2):
        if llama_factory is None:
            from llama_cpp import Llama  # lazy: not installed in the v3 image
            llama_factory = Llama
        self._lock = threading.Lock()
        self._llm = llama_factory(model_path=path, n_ctx=2048,
                                  n_threads=n_threads, verbose=False)

    def generate(self, user_text, max_tokens=LOCAL_MAX_TOKENS):
        with self._lock:
            out = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": user_text}],
                max_tokens=max_tokens, temperature=0.0)
        return (out["choices"][0]["message"]["content"] or "").strip()

    def classify(self, prompt, categories):
        instruction = ("Classify this task. Answer with exactly one word from: "
                       + ", ".join(categories) + ".\nTask: " + prompt[:500]
                       + "\nCategory:")
        word = self.generate(instruction, max_tokens=CLASSIFY_MAX_TOKENS)
        word = word.lower().strip(" .:\n\"'")
        return word if word in categories else None
