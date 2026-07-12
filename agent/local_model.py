"""Qwen3-1.7B via llama-cpp, CPU-only, serialized behind a lock.

llama.cpp contexts are not thread-safe: exactly one generation runs at a
time; the Fireworks worker pool is unaffected.
Generation time is bounded by construction via small max_tokens caps (no
kill-timer exists in llama-cpp) — the speed spike validates the worst
case fits 30 s/response.

Qwen3 is a hybrid-thinking model: it will emit <think>...</think>
reasoning blocks unless told not to. Non-thinking mode is enforced at
both layers — switched at the source by sending the `/no_think` soft
switch as a dedicated SYSTEM message (Qwen3 supports a system role,
unlike Gemma), and stripped as defense-in-depth from the raw output in
case the switch is ignored or the block is left unclosed.

Measured (v7 spike): appending " /no_think" to the user message also
suppresses thinking, but on this GGUF it can trigger a greedy-decode
degeneracy on some prompts — the model emits runs of U+202F/whitespace
until the token cap instead of stopping. Placing "/no_think" as its own
system message suppresses thinking without triggering the degeneracy,
so a whitespace-run guard below is kept as a second line of defense.
"""
import os
import re
import threading
import time

DEFAULT_MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH", "/app/models/Qwen3-1.7B-Q4_K_M.gguf")
LOCAL_MAX_TOKENS = 160
CLASSIFY_MAX_TOKENS = 8
LOCAL_CATEGORY_MAX_TOKENS = {
    "sentiment": 160, "ner": 160, "factual": 160,
    "summarisation": 256, "math": 256,
    "logic": 320, "code_debug": 320, "code_gen": 320,
}
# Worst-case lock wait + generation reserve; bounds the budget handed to the
# lock/generation call itself. agent.main imports this constant directly
# (single source of truth) to gate whether the lane is even attempted.
# TODO: recalibrated by the v6 spike (Task 2).
LOCAL_WORST_SECONDS = 30.0

# Defense-in-depth: strip any <think>...</think> block the model emits
# despite the soft switch.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Defense-in-depth: greedy Q4 decoding can degenerate into runs of
# whitespace (including U+202F) that continue until the token cap; no
# legitimate answer contains 6+ consecutive whitespace characters, so
# truncate at the first such run.
_DEGEN_RE = re.compile(r"[\s ]{6,}")


class LocalModel:
    def __init__(self, path=DEFAULT_MODEL_PATH, llama_factory=None, n_threads=2):
        if llama_factory is None:
            from llama_cpp import Llama  # lazy: not installed in the v3 image
            llama_factory = Llama
        self._lock = threading.Lock()
        self._llm = llama_factory(model_path=path, n_ctx=2048,
                                  n_threads=n_threads, verbose=False)

    def generate(self, user_text, max_tokens=LOCAL_MAX_TOKENS, deadline=None):
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": user_text},
        ]
        if deadline is not None:
            budget = deadline - time.monotonic() - 5.0  # 5s reserve for the answer write
            if budget <= 0:
                return ""
            if not self._lock.acquire(timeout=budget):
                return ""
            try:
                out = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens, temperature=0.0)
            finally:
                self._lock.release()
        else:
            with self._lock:
                out = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens, temperature=0.0)
        raw = (out["choices"][0]["message"]["content"] or "").strip()
        text = _THINK_RE.sub("", raw).strip()
        if text.startswith("<think>"):
            # Unclosed think block slipped through: treat as no usable
            # answer and let the caller fall back to the cloud path.
            return ""
        m = _DEGEN_RE.search(text)
        text = text[:m.start()].strip() if m else text
        return text

    def classify(self, prompt, categories, deadline=None):
        instruction = ("Classify this task. Answer with exactly one word from: "
                       + ", ".join(categories) + ".\nTask: " + prompt[:500]
                       + "\nCategory:")
        word = self.generate(instruction, max_tokens=CLASSIFY_MAX_TOKENS, deadline=deadline)
        word = word.lower().strip(" .:\n\"'")
        return word if word in categories else None
