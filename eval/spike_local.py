"""Go/no-go: Gemma-2-2B-it Q4_K_M on 2 CPU threads. Run under taskset -c 0,1.

GO requires: load <= 40s AND slowest answer <= 25s.
"""
import time

MODEL = "models/gemma-2-2b-it-Q4_K_M.gguf"
PROMPTS = [
    ("sentiment", "Classify the sentiment of this review and justify in one sentence: "
                  "The checkout was seamless but the product arrived cracked."),
    ("ner", "Extract all named entities and their types, one per line: On 14 February 2025, "
            "Dr. Amara Okafor of the European Space Agency presented at MIT in Cambridge."),
]

t0 = time.monotonic()
from llama_cpp import Llama  # noqa: E402
llm = Llama(model_path=MODEL, n_ctx=2048, n_threads=2, verbose=False)
load_s = time.monotonic() - t0
print(f"load: {load_s:.1f}s")

worst = 0.0
for name, prompt in PROMPTS:
    t = time.monotonic()
    out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                     max_tokens=160, temperature=0.0)
    dt = time.monotonic() - t
    worst = max(worst, dt)
    text = out["choices"][0]["message"]["content"].strip()
    print(f"{name}: {dt:.1f}s, {len(text)} chars -> {text[:100]!r}")

print(f"VERDICT: {'GO' if load_s <= 40 and worst <= 25 else 'NO-GO'} "
      f"(load {load_s:.1f}s, worst answer {worst:.1f}s)")
