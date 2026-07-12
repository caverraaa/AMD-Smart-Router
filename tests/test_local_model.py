import threading
import time

from agent.local_model import CLASSIFY_MAX_TOKENS, LOCAL_MAX_TOKENS, LOCAL_WORST_SECONDS, LocalModel


class FakeLlama:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def create_chat_completion(self, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.005)  # real inference takes time; without this, unlocked calls never overlap
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        with self._lock:
            self.active -= 1
        return {"choices": [{"message": {"content": reply}}]}


def make(replies):
    fake = FakeLlama(replies)
    lm = LocalModel(path="unused.gguf", llama_factory=lambda **kw: fake)
    return lm, fake


def test_generate_returns_stripped_text():
    lm, fake = make(["  Positive — praises battery.  "])
    assert lm.generate("classify this") == "Positive — praises battery."
    call = fake.calls[0]
    assert call["max_tokens"] == LOCAL_MAX_TOKENS
    assert call["messages"] == [{"role": "user", "content": "classify this"}]  # no system role


def test_generate_custom_cap():
    lm, fake = make(["x"])
    lm.generate("p", max_tokens=64)
    assert fake.calls[0]["max_tokens"] == 64


def test_ner_valid_first_answer_is_canonicalized_without_repair():
    lm, fake = make([
        "Maria Sanchez PERSON\nFireworks AI ORG\nBerlin LOCATION\nlast March DATE"
    ])
    prompt = ("Extract all named entities and their types from: Maria Sanchez joined "
              "Fireworks AI in Berlin last March.")
    answer = lm.generate(prompt)

    assert answer.endswith("last March — DATE")
    assert len(fake.calls) == 1
    assert fake.calls[0]["messages"][0]["content"].startswith(prompt)
    assert "Use exactly one pair per line" in fake.calls[0]["messages"][0]["content"]


def test_ner_shortened_date_is_repaired_deterministically():
    lm, fake = make([
        "Maria Sanchez — PERSON\nFireworks AI — ORG\nBerlin — LOCATION\nMarch — DATE",
    ])
    prompt = ("Extract all named entities and their types from: Maria Sanchez joined "
              "Fireworks AI in Berlin last March.")
    answer = lm.answer(prompt, category="ner")

    assert answer.endswith("last March — DATE")
    assert len(fake.calls) == 1


def test_ner_invalid_structure_gets_one_model_repair_attempt():
    lm, fake = make([
        "Here are the entities:\nMaria Sanchez — PERSON",
        "Maria Sanchez — PERSON\nFireworks AI — ORG\nBerlin — LOCATION\nlast March — DATE",
    ])
    prompt = ("Extract all named entities and their types from: Maria Sanchez joined "
              "Fireworks AI in Berlin last March.")
    answer = lm.answer(prompt, category="ner")

    assert answer.endswith("last March — DATE")
    assert len(fake.calls) == 2
    repair = fake.calls[1]["messages"][0]["content"]
    assert "Repair the NER answer" in repair
    assert "use one entity and type per line" in repair


def test_ner_second_invalid_answer_fails_closed_for_cloud_fallback():
    lm, fake = make(["London — LOCATION", "London — LOCATION"])
    prompt = "Extract all named entities and their types from: Berlin last March."
    assert lm.generate(prompt) == ""
    assert len(fake.calls) == 2


def test_unsupported_ner_format_fails_closed_without_local_generation():
    lm, fake = make([])
    prompt = "Extract all named entities as JSON from: Maria joined Acme."
    assert lm.answer(prompt, category="ner") == ""
    assert fake.calls == []


def test_supported_logic_is_solved_without_model_generation():
    lm, fake = make([])
    prompt = (
        "Answer in English. Three friends, Sam, Jo, and Lee, each own a "
        "different pet: cat, dog, bird. Sam does not own the bird. Jo owns "
        "the dog. Who owns the cat?\n\n"
        "Answer first; briefly verify every constraint."
    )
    assert lm.answer(prompt, category="logic") == "Sam owns the cat."
    assert fake.calls == []


def test_unsupported_logic_fails_closed_without_model_generation():
    lm, fake = make([])
    prompt = (
        "Three students, Ali, Ben, and Cho, each have a favorite color: red, "
        "blue, or green. Ali does not like green. Ben likes red. Who likes blue?"
    )
    assert lm.answer(prompt, category="logic") == ""
    assert fake.calls == []


def test_supported_summary_is_fused_without_model_generation():
    lm, fake = make([])
    source = (
        "Remote work changed office planning and reduced commuting for many staff. "
        "Companies adopted flexible schedules and hired from a wider geographic area. "
        "Employees gained autonomy but reported weaker boundaries between work and home."
    )
    prompt = (
        "Answer in English. Summarize the following in exactly one sentence: "
        + source
        + "\n\nMatch requested format and length exactly. No preamble."
    )

    assert lm.answer(prompt, category="summarisation") == source.replace(". ", "; ")
    assert fake.calls == []


def test_unsupported_summary_fails_closed_without_model_generation():
    lm, fake = make([])
    prompt = "Summarize this in two bullets: One fact. Another fact."
    assert lm.answer(prompt, category="summarisation") == ""
    assert fake.calls == []


def test_sentiment_answer_is_structurally_validated_and_canonicalized():
    lm, fake = make([
        "Mixed.\n\nReason: Great battery life but a fragile screen."
    ])
    answer = lm.answer(
        "Classify the sentiment of this review: Great battery, fragile screen.",
        category="sentiment",
    )
    assert answer == "Mixed. Reason: Great battery life but a fragile screen."
    assert len(fake.calls) == 1


def test_invalid_sentiment_answer_fails_closed_after_one_generation():
    lm, fake = make(["Mixed."])
    assert lm.answer(
        "Classify the sentiment of this review: Great battery, fragile screen.",
        category="sentiment",
    ) == ""
    assert len(fake.calls) == 1


def test_classify_valid_word():
    lm, _ = make(["sentiment"])
    assert lm.classify("is this good or bad?", ("sentiment", "ner")) == "sentiment"


def test_classify_normalizes_punctuation_and_case():
    lm, _ = make([" Sentiment. "])
    assert lm.classify("x", ("sentiment", "ner")) == "sentiment"


def test_classify_garbage_returns_none():
    lm, _ = make(["I think this could be several things"])
    assert lm.classify("x", ("sentiment", "ner")) is None


def test_classify_uses_small_token_cap():
    lm, fake = make(["sentiment"])
    lm.classify("x", ("sentiment",))
    assert fake.calls[0]["max_tokens"] == CLASSIFY_MAX_TOKENS


def test_concurrent_calls_serialize():
    lm, fake = make(["a"] * 8)
    threads = [threading.Thread(target=lm.generate, args=("p",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake.max_active == 1  # the lock never admits two generations at once


def test_local_worst_seconds_constant():
    assert LOCAL_WORST_SECONDS == 30.0


def test_generate_returns_empty_when_lock_held_past_deadline():
    lm, fake = make(["first", "second"])

    t = threading.Thread(target=lm.generate, args=("first",))
    t.start()
    result = lm.generate("second", deadline=time.monotonic() + 0.05)
    t.join()

    assert result == ""
    assert len(fake.calls) == 1  # only the thread's call reached the llm


def test_generate_returns_empty_when_deadline_already_passed():
    lm, fake = make(["a"])
    result = lm.generate("p", deadline=time.monotonic() - 1)
    assert result == ""
    assert fake.calls == []
