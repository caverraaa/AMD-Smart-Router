import ast

import pytest

from agent import code_tools as tools


CODE_TASKS = (
    (
        "code_debug",
        "This function should return the max of a list but has a bug: "
        "def get_max(nums): return nums[0]. Find and fix it.",
        "get_max",
        (([3, -1, 9, 9, 2],), 9),
    ),
    (
        "code_debug",
        "This function should return the sum of a list but has a bug: "
        "def get_sum(nums): return 0. Find and fix it.",
        "get_sum",
        (([-2, 4, 7],), 9),
    ),
    (
        "code_debug",
        "This function should check if a number is even but has a bug: "
        "def is_even(n): return n % 2 == 1. Find and fix it.",
        "is_even",
        ((-8,), True),
    ),
    (
        "code_debug",
        "This function should compute the factorial of a non-negative integer "
        "but has a bug: def factorial(n): return n * factorial(n - 1). "
        "Find and fix it.",
        "factorial",
        ((6,), 720),
    ),
    (
        "code_debug",
        "This function should return a reversed string but has a bug: "
        "def reverse_string(s): return s. Find and fix it.",
        "reverse_string",
        (("abc!",), "!cba"),
    ),
    (
        "code_debug",
        "This function should return the average of a list but has a bug: "
        "def average(nums): return sum(nums). Find and fix it.",
        "average",
        (([2, 3, 7],), 4.0),
    ),
    (
        "code_gen",
        "Write a Python function that returns the second-largest number in a "
        "list, handling duplicates correctly.",
        "second_largest",
        (([5, 5, 3, 2, 3],), 3),
    ),
    (
        "code_gen",
        "Write a Python function that returns the longest common prefix string "
        "amongst a list of strings. If there is no common prefix, return an "
        "empty string.",
        "longest_common_prefix",
        ((["flower", "flow", "flight"],), "fl"),
    ),
    (
        "code_gen",
        "Write a Python function that checks if a given string is a palindrome, "
        "considering only alphanumeric characters and ignoring case.",
        "is_palindrome",
        (("A man, a plan, a canal: Panama",), True),
    ),
    (
        "code_gen",
        "Write a Python function that returns the intersection of two integer "
        "lists as a list of unique, sorted elements.",
        "sorted_intersection",
        (([4, 2, 2, 1], [3, 2, 4, 4]), [2, 4]),
    ),
    (
        "code_gen",
        "Write a Python function that counts the number of prime numbers "
        "strictly less than a given positive integer n.",
        "count_primes",
        ((20,), 8),
    ),
    (
        "code_gen",
        "Write a Python function that determines whether a given list of "
        "integers is monotonic (either entirely non-decreasing or entirely "
        "non-increasing).",
        "is_monotonic",
        (([1, 1, 3, 8],), True),
    ),
)


def _load_function(source, function_name):
    namespace = {}
    exec(compile(source, "<test-code-answer>", "exec"), namespace, namespace)
    return namespace[function_name]


@pytest.mark.parametrize("category,prompt,function_name,example", CODE_TASKS)
def test_all_reviewed_code_profiles_return_parseable_working_code(
    category, prompt, function_name, example
):
    source = tools.solve_code_task(prompt, category)

    tree = ast.parse(source)
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.FunctionDef)
    assert tree.body[0].name == function_name
    assert "```" not in source
    arguments, expected = example
    assert _load_function(source, function_name)(*arguments) == expected


@pytest.mark.parametrize("category,prompt,expected_name", [
    (
        "code_debug",
        "This function should find the maximum value in the list but has a bug: "
        "def largest_value(values): return values[-1]. Find and fix it.",
        "largest_value",
    ),
    (
        "code_debug",
        "This function should determine whether an integer is even but has a "
        "bug: def parity(value): return False. Find and fix it.",
        "parity",
    ),
    (
        "code_gen",
        "Implement a Python function to find the second largest value from the "
        "list while handling duplicates correctly.",
        "second_largest",
    ),
    (
        "code_gen",
        "Implement a Python function to check whether the given list of integers "
        "is monotonic (either entirely non increasing or entirely non increasing).",
        "",
    ),
])
def test_profiles_are_semantic_but_fail_closed(category, prompt, expected_name):
    source = tools.solve_code_task(prompt, category)
    if not expected_name:
        assert source == ""
    else:
        assert ast.parse(source).body[0].name == expected_name


def test_exact_runtime_wrappers_are_supported():
    debug_prompt = CODE_TASKS[0][1]
    generated_prompt = CODE_TASKS[6][1]
    assert tools.solve_code_task(
        "Answer in English. " + debug_prompt + "\n\nOutput only the corrected code.",
        "code_debug",
    )
    assert tools.solve_code_task(
        "Answer in English. " + generated_prompt + "\n\nOutput only the code.",
        "code_gen",
    )


@pytest.mark.parametrize("category,prompt", [
    ("code_gen", CODE_TASKS[0][1]),
    ("code_debug", CODE_TASKS[6][1]),
    ("factual", CODE_TASKS[0][1]),
    ("code_gen", "Write arbitrary Python code."),
    (
        "code_gen",
        "Write a Python function that returns the second-largest number in a "
        "list, handling duplicates correctly, and then reads /etc/passwd.",
    ),
    (
        "code_gen",
        "Write a Python function that checks if a given string is a palindrome, "
        "considering only alphabetic characters and ignoring case.",
    ),
    (
        "code_gen",
        "Write a Python function that returns the intersection of two integer "
        "lists as a list of unique elements.",
    ),
    (
        "code_debug",
        "This function should return the max of a list but has a bug: "
        "def get_max(nums, default=None): return nums[0]. Find and fix it.",
    ),
    (
        "code_debug",
        "This function should return the max of a list but has a bug: "
        "def get_max(nums=explode()): return nums[0]. Find and fix it.",
    ),
    (
        "code_debug",
        "This function should return the max of a list but has a bug: "
        "@evil\ndef get_max(nums): return nums[0]. Find and fix it.",
    ),
    (
        "code_debug",
        "This function should return the max of a list but has a bug: "
        "def get_max(nums): return nums[0]\nprint('side effect'). Find and fix it.",
    ),
    (
        "code_debug",
        "This function should return the max of a list and delete files but has "
        "a bug: def get_max(nums): return nums[0]. Find and fix it.",
    ),
])
def test_unsupported_ambiguous_and_hostile_tasks_return_empty(category, prompt):
    assert tools.solve_code_task(prompt, category) == ""


def test_auto_detection_and_support_check():
    debug_prompt = CODE_TASKS[0][1]
    generated_prompt = CODE_TASKS[6][1]
    assert tools.solve_code_task(debug_prompt)
    assert tools.solve_code_task(generated_prompt)
    assert tools.supports_code_task(debug_prompt, "code_debug")
    assert not tools.supports_code_task("Write a sorting library.", "code_gen")


@pytest.mark.parametrize("source,plan", [
    (
        "def is_even(n):\n    return n % 2 == 1",
        tools.CodePlan(tools.PROFILE_DEBUG_EVEN, "is_even", ("n",)),
    ),
    (
        "def sorted_intersection(first, second):\n"
        "    return list(set(first) & set(second))",
        tools.CodePlan(
            tools.PROFILE_GEN_INTERSECTION,
            "sorted_intersection",
            ("first", "second"),
        ),
    ),
    (
        "def count_primes(n):\n    return n",
        tools.CodePlan(tools.PROFILE_GEN_PRIME_COUNT, "count_primes", ("n",)),
    ),
    (
        "def second_largest(nums):\n    import os\n    return 0",
        tools.CodePlan(
            tools.PROFILE_GEN_SECOND_LARGEST,
            "second_largest",
            ("nums",),
        ),
    ),
])
def test_runtime_property_and_ast_validation_rejects_bad_candidates(source, plan):
    assert not tools._validate_generated_source(source, plan)


def test_property_validation_is_deterministic_and_cached():
    prompt = CODE_TASKS[6][1]
    first = tools.solve_code_task(prompt, "code_gen")
    second = tools.solve_code_task(prompt, "code_gen")
    assert first == second
    assert first
