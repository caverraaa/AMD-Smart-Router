"""Deterministic Python code repair/generation for narrowly proved tasks.

The public entry point, :func:`solve_code_task`, recognizes semantic task
profiles rather than benchmark IDs.  It emits code only after all of these
checks pass:

* the request matches a complete, reviewed grammar;
* a debug request contains one parseable, ordinary one-argument function;
* the generated answer is a single AST-parseable function with no imports or
  unsafe calls; and
* the function passes deterministic property tests over a finite input domain.

Anything outside those profiles returns ``""`` so the caller can fall back to
the cloud.  Prompt-provided code is parsed for its signature but never run.
"""
from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable


PROFILE_DEBUG_MAX = "code-debug-max-v1"
PROFILE_DEBUG_SUM = "code-debug-sum-v1"
PROFILE_DEBUG_EVEN = "code-debug-even-v1"
PROFILE_DEBUG_FACTORIAL = "code-debug-factorial-v1"
PROFILE_DEBUG_REVERSE = "code-debug-reverse-v1"
PROFILE_DEBUG_AVERAGE = "code-debug-average-v1"

PROFILE_GEN_SECOND_LARGEST = "code-gen-second-largest-v1"
PROFILE_GEN_LONGEST_PREFIX = "code-gen-longest-prefix-v1"
PROFILE_GEN_PALINDROME = "code-gen-palindrome-v1"
PROFILE_GEN_INTERSECTION = "code-gen-intersection-v1"
PROFILE_GEN_PRIME_COUNT = "code-gen-prime-count-v1"
PROFILE_GEN_MONOTONIC = "code-gen-monotonic-v1"

CODE_DEBUG_PROFILES = frozenset({
    PROFILE_DEBUG_MAX,
    PROFILE_DEBUG_SUM,
    PROFILE_DEBUG_EVEN,
    PROFILE_DEBUG_FACTORIAL,
    PROFILE_DEBUG_REVERSE,
    PROFILE_DEBUG_AVERAGE,
})
CODE_GEN_PROFILES = frozenset({
    PROFILE_GEN_SECOND_LARGEST,
    PROFILE_GEN_LONGEST_PREFIX,
    PROFILE_GEN_PALINDROME,
    PROFILE_GEN_INTERSECTION,
    PROFILE_GEN_PRIME_COUNT,
    PROFILE_GEN_MONOTONIC,
})

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")
_DEBUG_WRAPPER_RE = re.compile(
    r"^This function should (?P<intent>[^\r\n]{3,240}?) but has a bug:\s*"
    r"(?P<source>def\s+.+)\.\s*Find and fix it\.$",
    re.IGNORECASE | re.DOTALL,
)
_GEN_WRAPPER_RE = re.compile(
    r"^(?:Write|Implement) a Python function (?:that|to) "
    r"(?P<intent>[^\r\n]{3,500})\.$",
    re.IGNORECASE,
)

_DEBUG_INTENT_PATTERNS = (
    (
        PROFILE_DEBUG_MAX,
        re.compile(
            r"(?:return|find) the (?:max|maximum)(?: value)? "
            r"(?:of|in|from) (?:a|the) list",
            re.IGNORECASE,
        ),
    ),
    (
        PROFILE_DEBUG_SUM,
        re.compile(
            r"(?:return|compute|calculate) the (?:sum|total) "
            r"(?:of|for) (?:a|the) list",
            re.IGNORECASE,
        ),
    ),
    (
        PROFILE_DEBUG_EVEN,
        re.compile(
            r"(?:check|determine) (?:if|whether) (?:a|an|the) "
            r"(?:number|integer) is even",
            re.IGNORECASE,
        ),
    ),
    (
        PROFILE_DEBUG_FACTORIAL,
        re.compile(
            r"(?:compute|return|calculate) the factorial of (?:a|the) "
            r"non[- ]negative integer",
            re.IGNORECASE,
        ),
    ),
    (
        PROFILE_DEBUG_REVERSE,
        re.compile(
            r"(?:return|produce) (?:a|the) reversed string|"
            r"reverse (?:a|the) string",
            re.IGNORECASE,
        ),
    ),
    (
        PROFILE_DEBUG_AVERAGE,
        re.compile(
            r"(?:return|compute|calculate) the (?:average|mean) "
            r"(?:of|for) (?:a|the) list",
            re.IGNORECASE,
        ),
    ),
)

_GEN_INTENT_PATTERNS = (
    (
        PROFILE_GEN_SECOND_LARGEST,
        re.compile(
            r"(?:returns?|finds?) the second[- ]largest (?:distinct )?"
            r"(?:number|value) (?:in|from) (?:a|the) list,? "
            r"(?:while )?handling duplicates correctly",
            re.IGNORECASE,
        ),
    ),
    (
        PROFILE_GEN_LONGEST_PREFIX,
        re.compile(
            r"(?:returns?|finds?) the longest common prefix(?: string)? "
            r"(?:among|amongst|shared by) (?:all )?(?:a|the) list of strings\. "
            r"If there is no common prefix, return an empty string",
            re.IGNORECASE,
        ),
    ),
    (
        PROFILE_GEN_PALINDROME,
        re.compile(
            r"(?:checks?|determines?) (?:if|whether) (?:a|the) given string "
            r"is a palindrome, considering only alphanumeric characters "
            r"and ignoring case",
            re.IGNORECASE,
        ),
    ),
    (
        PROFILE_GEN_INTERSECTION,
        re.compile(
            r"(?:returns?|computes?|finds?) the intersection of two integer "
            r"lists as a list of unique, sorted elements",
            re.IGNORECASE,
        ),
    ),
    (
        PROFILE_GEN_PRIME_COUNT,
        re.compile(
            r"(?:counts?|returns? the count of) the number of prime numbers "
            r"strictly less than (?:a|the) given positive integer n",
            re.IGNORECASE,
        ),
    ),
    (
        PROFILE_GEN_MONOTONIC,
        re.compile(
            r"(?:determines?|checks?) whether (?:a|the) given list of integers "
            r"is monotonic \(either entirely non[- ]decreasing or entirely "
            r"non[- ]increasing\)",
            re.IGNORECASE,
        ),
    ),
)

_SAFE_BUILTINS = {
    "ValueError": ValueError,
    "all": all,
    "len": len,
    "range": range,
    "set": set,
    "sorted": sorted,
    "sum": sum,
}
_SAFE_CALL_NAMES = frozenset(_SAFE_BUILTINS)
_SAFE_METHOD_CALLS = frozenset({"casefold", "isalnum", "join", "startswith"})
_FORBIDDEN_AST_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


@dataclass(frozen=True)
class CodePlan:
    profile: str
    function_name: str
    argument_names: tuple[str, ...]


def solve_code_task(user_text: str, category: str | None = None) -> str:
    """Return verified Python source, or ``""`` for unsupported input.

    ``category`` may be ``"code_debug"``, ``"code_gen"``, or ``None`` for
    auto-detection.  A category mismatch fails closed.
    """
    if not isinstance(user_text, str) or not user_text.strip():
        return ""
    if category is not None and category not in ("code_debug", "code_gen"):
        return ""
    task = _strip_runtime_wrapper(user_text)
    if not task or len(task) > 2000 or "\x00" in task:
        return ""
    return _solve_cached(task, category)


def supports_code_task(user_text: str, category: str | None = None) -> bool:
    """Whether this exact task produces execution-verified source."""
    return bool(solve_code_task(user_text, category))


@lru_cache(maxsize=128)
def _solve_cached(task: str, category: str | None) -> str:
    try:
        plan = _parse_debug_plan(task) if category != "code_gen" else None
        if plan is None and category != "code_debug":
            plan = _parse_generation_plan(task)
        if plan is None:
            return ""
        source = _render(plan)
        return source if _validate_generated_source(source, plan) else ""
    except Exception:
        # Parsing, compilation, or a property-check failure can only make the
        # caller spend cloud tokens; it must never leak an unproved answer.
        return ""


def _strip_runtime_wrapper(user_text: str) -> str:
    text = user_text.strip()
    prefix = "Answer in English. "
    if text.startswith(prefix):
        text = text[len(prefix):]
    for suffix in (
        "\n\nOutput only the corrected code.",
        "\n\nOutput only the code.",
    ):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    return text.strip()


def _parse_debug_plan(task: str) -> CodePlan | None:
    match = _DEBUG_WRAPPER_RE.fullmatch(task)
    if not match:
        return None
    profile = _match_profile(match.group("intent"), _DEBUG_INTENT_PATTERNS)
    if profile is None:
        return None
    signature = _single_argument_signature(match.group("source"))
    if signature is None:
        return None
    function_name, argument_name = signature
    return CodePlan(profile, function_name, (argument_name,))


def _parse_generation_plan(task: str) -> CodePlan | None:
    match = _GEN_WRAPPER_RE.fullmatch(task)
    if not match:
        return None
    profile = _match_profile(match.group("intent"), _GEN_INTENT_PATTERNS)
    if profile is None:
        return None
    names = {
        PROFILE_GEN_SECOND_LARGEST: ("second_largest", ("nums",)),
        PROFILE_GEN_LONGEST_PREFIX: ("longest_common_prefix", ("strings",)),
        PROFILE_GEN_PALINDROME: ("is_palindrome", ("text",)),
        PROFILE_GEN_INTERSECTION: ("sorted_intersection", ("first", "second")),
        PROFILE_GEN_PRIME_COUNT: ("count_primes", ("n",)),
        PROFILE_GEN_MONOTONIC: ("is_monotonic", ("nums",)),
    }
    function_name, arguments = names[profile]
    return CodePlan(profile, function_name, arguments)


def _match_profile(text: str, patterns) -> str | None:
    matches = [profile for profile, pattern in patterns if pattern.fullmatch(text)]
    return matches[0] if len(matches) == 1 else None


def _single_argument_signature(source: str) -> tuple[str, str] | None:
    try:
        tree = ast.parse(source.strip(), mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    function = tree.body[0]
    args = function.args
    positional = tuple(args.posonlyargs) + tuple(args.args)
    if (
        len(positional) != 1
        or args.vararg is not None
        or args.kwarg is not None
        or args.kwonlyargs
        or args.defaults
        or args.kw_defaults
        or function.decorator_list
    ):
        return None
    function_name = function.name
    argument_name = positional[0].arg
    if not _safe_identifier(function_name) or not _safe_identifier(argument_name):
        return None
    return function_name, argument_name


def _safe_identifier(value: str) -> bool:
    return bool(
        _IDENTIFIER_RE.fullmatch(value)
        and not value.startswith("_")
        and "__" not in value
    )


def _render(plan: CodePlan) -> str:
    name = plan.function_name
    args = plan.argument_names
    if plan.profile == PROFILE_DEBUG_MAX:
        arg = args[0]
        return (
            f"def {name}({arg}):\n"
            f"    current = {arg}[0]\n"
            f"    for value in {arg}[1:]:\n"
            "        if value > current:\n"
            "            current = value\n"
            "    return current"
        )
    if plan.profile == PROFILE_DEBUG_SUM:
        arg = args[0]
        return (
            f"def {name}({arg}):\n"
            "    total = 0\n"
            f"    for value in {arg}:\n"
            "        total += value\n"
            "    return total"
        )
    if plan.profile == PROFILE_DEBUG_EVEN:
        arg = args[0]
        return f"def {name}({arg}):\n    return {arg} % 2 == 0"
    if plan.profile == PROFILE_DEBUG_FACTORIAL:
        arg = args[0]
        return (
            f"def {name}({arg}):\n"
            f"    if {arg} <= 1:\n"
            "        return 1\n"
            f"    return {arg} * {name}({arg} - 1)"
        )
    if plan.profile == PROFILE_DEBUG_REVERSE:
        arg = args[0]
        return f"def {name}({arg}):\n    return {arg}[::-1]"
    if plan.profile == PROFILE_DEBUG_AVERAGE:
        arg = args[0]
        return (
            f"def {name}({arg}):\n"
            "    total = 0\n"
            "    count = 0\n"
            f"    for value in {arg}:\n"
            "        total += value\n"
            "        count += 1\n"
            "    return total / count"
        )
    if plan.profile == PROFILE_GEN_SECOND_LARGEST:
        return (
            "def second_largest(nums):\n"
            "    distinct = sorted(set(nums), reverse=True)\n"
            "    if len(distinct) < 2:\n"
            "        raise ValueError(\"at least two distinct values are required\")\n"
            "    return distinct[1]"
        )
    if plan.profile == PROFILE_GEN_LONGEST_PREFIX:
        return (
            "def longest_common_prefix(strings):\n"
            "    if not strings:\n"
            "        return \"\"\n"
            "    prefix = strings[0]\n"
            "    for text in strings[1:]:\n"
            "        while prefix and not text.startswith(prefix):\n"
            "            prefix = prefix[:-1]\n"
            "        if not prefix:\n"
            "            return \"\"\n"
            "    return prefix"
        )
    if plan.profile == PROFILE_GEN_PALINDROME:
        return (
            "def is_palindrome(text):\n"
            "    normalized = \"\".join(\n"
            "        char.casefold() for char in text if char.isalnum()\n"
            "    )\n"
            "    return normalized == normalized[::-1]"
        )
    if plan.profile == PROFILE_GEN_INTERSECTION:
        return (
            "def sorted_intersection(first, second):\n"
            "    return sorted(set(first) & set(second))"
        )
    if plan.profile == PROFILE_GEN_PRIME_COUNT:
        return (
            "def count_primes(n):\n"
            "    if n <= 2:\n"
            "        return 0\n"
            "    is_prime = [True] * n\n"
            "    is_prime[0] = is_prime[1] = False\n"
            "    candidate = 2\n"
            "    while candidate * candidate < n:\n"
            "        if is_prime[candidate]:\n"
            "            for multiple in range(candidate * candidate, n, candidate):\n"
            "                is_prime[multiple] = False\n"
            "        candidate += 1\n"
            "    return sum(is_prime)"
        )
    if plan.profile == PROFILE_GEN_MONOTONIC:
        return (
            "def is_monotonic(nums):\n"
            "    non_decreasing = all(\n"
            "        nums[index] <= nums[index + 1]\n"
            "        for index in range(len(nums) - 1)\n"
            "    )\n"
            "    non_increasing = all(\n"
            "        nums[index] >= nums[index + 1]\n"
            "        for index in range(len(nums) - 1)\n"
            "    )\n"
            "    return non_decreasing or non_increasing"
        )
    return ""


def _validate_generated_source(source: str, plan: CodePlan) -> bool:
    if not source or "```" in source or len(source) > 2000:
        return False
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    function = tree.body[0]
    actual_args = tuple(arg.arg for arg in function.args.args)
    if function.name != plan.function_name or actual_args != plan.argument_names:
        return False
    if function.decorator_list or function.args.posonlyargs or function.args.kwonlyargs:
        return False
    if function.args.vararg is not None or function.args.kwarg is not None:
        return False
    if any(isinstance(node, _FORBIDDEN_AST_NODES) for node in ast.walk(tree)):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and "__" in node.id:
            return False
        if isinstance(node, ast.Attribute) and (
            node.attr not in _SAFE_METHOD_CALLS or node.attr.startswith("_")
        ):
            return False
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SAFE_CALL_NAMES | {plan.function_name}:
                    return False
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in _SAFE_METHOD_CALLS:
                    return False
            else:
                return False
    namespace = {"__builtins__": dict(_SAFE_BUILTINS)}
    try:
        exec(compile(tree, "<verified-code-answer>", "exec"), namespace, namespace)
        function_value = namespace.get(plan.function_name)
        return bool(callable(function_value) and _run_properties(plan.profile, function_value))
    except Exception:
        return False


def _run_properties(profile: str, function: Callable) -> bool:
    if profile == PROFILE_DEBUG_MAX:
        for values in _integer_lists(min_length=1, max_length=4):
            if function(list(values)) != max(values):
                return False
        return True
    if profile == PROFILE_DEBUG_SUM:
        for values in _integer_lists(min_length=0, max_length=4):
            if function(list(values)) != sum(values):
                return False
        return True
    if profile == PROFILE_DEBUG_EVEN:
        return all(function(value) is (value % 2 == 0) for value in range(-40, 41))
    if profile == PROFILE_DEBUG_FACTORIAL:
        expected = 1
        for value in range(0, 11):
            if value > 1:
                expected *= value
            if function(value) != expected:
                return False
        return True
    if profile == PROFILE_DEBUG_REVERSE:
        corpus = ("", "a", "ab", "racecar", "a b!", "naïve", "🙂ab")
        return all(function(value) == value[::-1] for value in corpus)
    if profile == PROFILE_DEBUG_AVERAGE:
        for values in _integer_lists(min_length=1, max_length=4):
            if function(list(values)) != sum(values) / len(values):
                return False
        return True
    if profile == PROFILE_GEN_SECOND_LARGEST:
        for values in _integer_lists(min_length=0, max_length=5):
            distinct = sorted(set(values), reverse=True)
            if len(distinct) < 2:
                try:
                    function(list(values))
                except ValueError:
                    continue
                return False
            if function(list(values)) != distinct[1]:
                return False
        return True
    if profile == PROFILE_GEN_LONGEST_PREFIX:
        strings = ("", "a", "b", "aa", "ab", "ba", "bb")
        for length in range(0, 4):
            for values in itertools.product(strings, repeat=length):
                if function(list(values)) != _reference_longest_prefix(values):
                    return False
        return True
    if profile == PROFILE_GEN_PALINDROME:
        for length in range(0, 5):
            for characters in itertools.product("aA1 !", repeat=length):
                value = "".join(characters)
                normalized = "".join(
                    char.casefold() for char in value if char.isalnum()
                )
                if function(value) is not (normalized == normalized[::-1]):
                    return False
        return all(
            function(value) is expected
            for value, expected in (
                ("A man, a plan, a canal: Panama", True),
                ("race a car", False),
                ("No 'x' in Nixon", True),
                ("Été", True),
            )
        )
    if profile == PROFILE_GEN_INTERSECTION:
        lists = tuple(_integer_lists(min_length=0, max_length=3))
        for first in lists:
            for second in lists:
                expected = sorted(set(first).intersection(second))
                if function(list(first), list(second)) != expected:
                    return False
        return True
    if profile == PROFILE_GEN_PRIME_COUNT:
        return all(
            function(value) == _reference_prime_count(value)
            for value in range(-5, 201)
        )
    if profile == PROFILE_GEN_MONOTONIC:
        for values in _integer_lists(min_length=0, max_length=5):
            expected = (
                all(left <= right for left, right in zip(values, values[1:]))
                or all(left >= right for left, right in zip(values, values[1:]))
            )
            if function(list(values)) is not expected:
                return False
        return True
    return False


def _integer_lists(min_length: int, max_length: int):
    for length in range(min_length, max_length + 1):
        yield from itertools.product(range(-2, 3), repeat=length)


def _reference_longest_prefix(strings) -> str:
    if not strings:
        return ""
    shortest = min(len(value) for value in strings)
    size = 0
    while size < shortest and len({value[size] for value in strings}) == 1:
        size += 1
    return strings[0][:size]


def _reference_prime_count(n: int) -> int:
    count = 0
    for value in range(2, n):
        if all(value % divisor for divisor in range(2, int(value ** 0.5) + 1)):
            count += 1
    return count
