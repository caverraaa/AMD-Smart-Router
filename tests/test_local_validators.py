from agent.local_validators import (
    build_ner_prompt,
    extract_ner_source,
    find_date_mentions,
    is_ner_request,
    parse_ner_answer,
    repair_shortened_date_spans,
    repair_trailing_descriptors,
    supports_local_ner_request,
    validate_ner_answer,
)


MERGED = (
    "Answer in English. Extract all named entities and their types from: "
    "Maria Sanchez joined Fireworks AI in Berlin last March.\n\n"
    "List each entity with its type, including relative dates such as 'next month'."
)


def test_extract_source_excludes_appended_constraint_example():
    assert extract_ner_source(MERGED) == (
        "Maria Sanchez joined Fireworks AI in Berlin last March."
    )
    assert find_date_mentions(extract_ner_source(MERGED)) == ("last March",)


def test_ner_request_detection_is_conservative():
    assert is_ner_request(MERGED)
    assert not is_ner_request("Classify this task as sentiment, ner, or factual.")
    assert not is_ner_request("Explain how named-entity recognition works.")


def test_local_ner_rejects_custom_formats_and_unsupported_types():
    assert not supports_local_ner_request(
        "Extract all named entities as JSON from: Maria joined Acme.")
    assert not supports_local_ner_request(
        "Extract PRODUCT and EVENT entities from: Pixel launched at I/O.")
    assert supports_local_ner_request(
        "Extract all named entities and their types from: JSON Corp hired Maria.")


def test_parser_accepts_compact_formats_and_normalizes_types():
    parsed = parse_ner_answer(
        "Maria Sanchez PERSON\nFireworks AI - ORGANIZATION\nBerlin: LOC\nlast March — DATE"
    )
    assert [(item.text, item.type) for item in parsed] == [
        ("Maria Sanchez", "PERSON"),
        ("Fireworks AI", "ORG"),
        ("Berlin", "LOCATION"),
        ("last March", "DATE"),
    ]


def test_validator_canonicalizes_valid_answer():
    result = validate_ner_answer(
        MERGED,
        "Maria Sanchez PERSON\nFireworks AI ORG\nBerlin LOCATION\nlast March DATE",
    )
    assert result.valid
    assert result.answer.endswith("last March — DATE")


def test_validator_rejects_shortened_relative_date():
    result = validate_ner_answer(
        MERGED,
        "Maria Sanchez — PERSON\nFireworks AI — ORG\nBerlin — LOCATION\nMarch — DATE",
    )
    assert not result.valid
    assert "copy date span `last March` exactly" in result.issues


def test_safe_deterministic_repair_restores_unique_date_prefix():
    repaired = repair_shortened_date_spans(
        MERGED,
        "Maria Sanchez PERSON\nFireworks AI ORG\nBerlin LOCATION\nMarch DATE",
    )
    result = validate_ner_answer(MERGED, repaired)
    assert result.valid
    assert result.answer.endswith("last March — DATE")


def test_trailing_descriptor_repair_preserves_internal_connectors():
    repaired = repair_trailing_descriptors(
        "Tesla factory — ORG\nMuseum of Modern Art — ORG\n"
        "Bank of America — ORG\nRio de Janeiro — LOCATION\nlast summer — DATE"
    )
    assert repaired.splitlines() == [
        "Tesla — ORG",
        "Museum of Modern Art — ORG",
        "Bank of America — ORG",
        "Rio de Janeiro — LOCATION",
        "last summer — DATE",
    ]


def test_validator_rejects_hallucinated_entity_and_extra_prose():
    hallucinated = validate_ner_answer(MERGED, "London — LOCATION")
    prose = validate_ner_answer(MERGED, "Here are the entities:\nBerlin — LOCATION")
    assert not hallucinated.valid
    assert "`London` is not an exact source span" in hallucinated.issues
    assert not prose.valid


def test_prompt_requires_exact_spans_and_repair_includes_feedback():
    first = build_ner_prompt(MERGED)
    repair = build_ner_prompt(
        MERGED, "March — DATE", ("copy date span `last March` exactly",))
    assert first.startswith(MERGED)
    assert "Use exactly one pair per line" in first
    assert "Previous answer:" in repair
    assert "copy date span `last March` exactly" in repair
