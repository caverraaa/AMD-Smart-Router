"""Gemma-2-2B-it via llama-cpp, CPU-only, serialized behind a lock.

llama.cpp contexts are not thread-safe: exactly one generation runs at a
time; the Fireworks worker pool is unaffected. Gemma has no system role,
so callers pass a single merged user message. Generation time is bounded
by construction via small max_tokens caps (no kill-timer exists in
llama-cpp) — the speed spike validates the worst case fits 30 s/response.
"""
import os
import threading
import time

try:
    from agent.code_tools import solve_code_task
    from agent.local_summary import (
        fuse_lossless_summary,
        validate_lossless_summary,
    )
    from agent.math_tools import solve_math
    from agent.local_validators import (
        build_ner_prompt,
        is_ner_request,
        repair_shortened_date_spans,
        repair_trailing_descriptors,
        supports_local_ner_request,
        validate_sentiment_answer,
        validate_logic_answer,
        validate_ner_answer,
    )
    from agent.local_tools import solve_assignment_logic
except ImportError:  # executed with /app/agent on sys.path
    from code_tools import solve_code_task
    from local_summary import fuse_lossless_summary, validate_lossless_summary
    from math_tools import solve_math
    from local_validators import (build_ner_prompt, is_ner_request,
                                  repair_shortened_date_spans,
                                  repair_trailing_descriptors,
                                  supports_local_ner_request,
                                  validate_sentiment_answer,
                                  validate_logic_answer, validate_ner_answer)
    from local_tools import solve_assignment_logic

DEFAULT_MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH", "/app/models/gemma-2-2b-it-Q4_K_M.gguf")
LOCAL_MAX_TOKENS = 160
CLASSIFY_MAX_TOKENS = 8
# Worst-case lock wait + generation reserve. Mirrors agent.main.LOCAL_WORST_SECONDS —
# that copy gates whether the lane is even attempted; this one bounds the budget
# handed to the lock/generation call itself.
LOCAL_WORST_SECONDS = 30.0


class LocalModel:
    def __init__(self, path=DEFAULT_MODEL_PATH, llama_factory=None, n_threads=2):
        if llama_factory is None:
            from llama_cpp import Llama  # lazy: not installed in the v3 image
            llama_factory = Llama
        self._lock = threading.Lock()
        self._llm = llama_factory(model_path=path, n_ctx=2048,
                                  n_threads=n_threads, verbose=False)

    def _generate_once(self, user_text, max_tokens, deadline=None):
        if deadline is not None:
            budget = deadline - time.monotonic() - 5.0  # 5s reserve for the answer write
            if budget <= 0:
                return ""
            if not self._lock.acquire(timeout=budget):
                return ""
            try:
                out = self._llm.create_chat_completion(
                    messages=[{"role": "user", "content": user_text}],
                    max_tokens=max_tokens, temperature=0.0)
            finally:
                self._lock.release()
        else:
            with self._lock:
                out = self._llm.create_chat_completion(
                    messages=[{"role": "user", "content": user_text}],
                    max_tokens=max_tokens, temperature=0.0)
        return (out["choices"][0]["message"]["content"] or "").strip()

    def answer(self, user_text, category=None, max_tokens=LOCAL_MAX_TOKENS,
               deadline=None):
        """Category-aware local answer with fail-closed validation.

        ``generate`` auto-detects NER for compatibility. Logic and summary use
        deterministic tools, sentiment has a structural output validator, and
        invalid NER gets one local repair attempt. Any rejected answer becomes
        ``""`` so the existing caller falls back to Fireworks.
        """
        if category == "logic":
            solved = solve_assignment_logic(user_text)
            checked = validate_logic_answer(user_text, solved)
            return checked.answer if checked.valid else ""
        if category == "math":
            return solve_math(user_text)
        if category in ("code_debug", "code_gen"):
            return solve_code_task(user_text, category)
        if category == "summarisation":
            fused = fuse_lossless_summary(user_text)
            return fused if validate_lossless_summary(user_text, fused) else ""
        if category == "sentiment":
            generated = self._generate_once(user_text, max_tokens, deadline)
            checked = validate_sentiment_answer(generated)
            return checked.answer if checked.valid else ""
        if category == "ner" or (category is None and is_ner_request(user_text)):
            if not supports_local_ner_request(user_text):
                return ""
            # Keep the already judge-tested merged NER prompt and add only a
            # narrow format/exact-span suffix. A full replacement prompt made
            # Gemma-2B confuse ORG and LOCATION on passing golden examples.
            first = self._generate_once(
                build_ner_prompt(user_text), max_tokens, deadline)
            first = repair_trailing_descriptors(first)
            first = repair_shortened_date_spans(user_text, first)
            checked = validate_ner_answer(user_text, first)
            if checked.valid:
                return checked.answer
            if deadline is not None and time.monotonic() + 5.0 >= deadline:
                return ""
            repair_prompt = build_ner_prompt(
                user_text, previous_answer=first, issues=checked.issues)
            repaired = self._generate_once(repair_prompt, max_tokens, deadline)
            repaired = repair_trailing_descriptors(repaired)
            repaired = repair_shortened_date_spans(user_text, repaired)
            checked = validate_ner_answer(user_text, repaired)
            return checked.answer if checked.valid else ""
        return self._generate_once(user_text, max_tokens, deadline)

    def generate(self, user_text, max_tokens=LOCAL_MAX_TOKENS, deadline=None):
        return self.answer(user_text, max_tokens=max_tokens, deadline=deadline)

    def classify(self, prompt, categories, deadline=None):
        instruction = ("Classify this task. Answer with exactly one word from: "
                       + ", ".join(categories) + ".\nTask: " + prompt[:500]
                       + "\nCategory:")
        word = self._generate_once(
            instruction, max_tokens=CLASSIFY_MAX_TOKENS, deadline=deadline)
        word = word.lower().strip(" .:\n\"'")
        return word if word in categories else None
