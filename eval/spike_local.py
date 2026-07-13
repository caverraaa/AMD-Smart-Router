"""Go/no-go: local model on 2 CPU threads. Run under taskset -c 0,1.

GO requires: load <= 40s AND worst answer <= 25s (at the largest per-category cap),
AND no <think> blocks in cleaned output (Qwen3 non-thinking enforcement works).
"""
import time

MODEL = "models/Qwen3-1.7B-Q4_K_M.gguf"
PROMPTS = [
    ("sentiment/160", 160, "Classify the sentiment of this review and justify in one sentence: "
     "The checkout was seamless but the product arrived cracked."),
    ("ner/160", 160, "Extract all named entities and their types, one per line: On 14 February 2025, "
     "Dr. Amara Okafor of the European Space Agency presented at MIT in Cambridge."),
    ("summ/256", 256, "Summarize the following in exactly one sentence: Electric vehicle adoption "
     "accelerated sharply this decade as battery costs fell below $100 per kilowatt-hour, charging "
     "networks expanded along major highways, and several countries announced 2035 deadlines for "
     "ending combustion-engine sales."),
    ("code_gen/320", 320, "Write a Python function that merges overlapping intervals from a list of "
     "[start, end] pairs, handles nested and duplicate intervals, and runs in O(n log n)."),
]

import sys
sys.path.insert(0, ".")
from agent.local_model import LocalModel  # uses /no_think + strip  # noqa: E402

t0 = time.monotonic()
lm = LocalModel(path=MODEL)
load_s = time.monotonic() - t0
print(f"load: {load_s:.1f}s")

worst = 0.0
think_leak = False
for name, cap, prompt in PROMPTS:
    t = time.monotonic()
    text = lm.generate(prompt, max_tokens=cap)
    dt = time.monotonic() - t
    worst = max(worst, dt)
    leaked = "<think>" in text
    think_leak = think_leak or leaked
    print(f"{name}: {dt:.1f}s, {len(text)} chars, think_leak={leaked} -> {text[:90]!r}")

ok = load_s <= 40 and worst <= 25 and not think_leak
print(f"VERDICT: {'GO' if ok else 'NO-GO'} (load {load_s:.1f}s, worst {worst:.1f}s, think_leak={think_leak})")
