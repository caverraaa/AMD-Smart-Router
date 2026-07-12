import threading
import time

from agent.local_model import (
    CLASSIFY_MAX_TOKENS,
    LOCAL_CATEGORY_MAX_TOKENS,
    LOCAL_MAX_TOKENS,
    LOCAL_WORST_SECONDS,
    LocalModel,
)


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
    assert call["messages"] == [
        {"role": "user", "content": "classify this /no_think"}
    ]  # no system role, no_think switch appended


def test_generate_appends_no_think_suffix():
    lm, fake = make(["ok"])
    lm.generate("some prompt")
    assert fake.calls[0]["messages"][0]["content"] == "some prompt /no_think"


def test_generate_strips_think_block():
    lm, _ = make(["<think>reasoning here</think>The answer"])
    assert lm.generate("p") == "The answer"


def test_generate_passes_through_when_no_think_block():
    lm, _ = make(["plain answer, no think tags"])
    assert lm.generate("p") == "plain answer, no think tags"


def test_generate_returns_empty_on_unclosed_think_block():
    lm, _ = make(["<think>still reasoning"])
    assert lm.generate("p") == ""


def test_generate_custom_cap():
    lm, fake = make(["x"])
    lm.generate("p", max_tokens=64)
    assert fake.calls[0]["max_tokens"] == 64


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


def test_local_category_max_tokens_has_exactly_eight_categories():
    assert LOCAL_CATEGORY_MAX_TOKENS == {
        "sentiment": 160,
        "ner": 160,
        "factual": 160,
        "summarisation": 256,
        "math": 256,
        "logic": 320,
        "code_debug": 320,
        "code_gen": 320,
    }


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
