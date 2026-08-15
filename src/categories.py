"""The closed set of CI failure categories.

Defined here rather than in config.yaml because Literal is a static type
annotation: it is evaluated when the class body is executed, which happens
before any config file is read. Literal[some_variable] is not valid Python,
so the set cannot come from config.

config.yaml declares the same list and is cross-checked against this one at
load time, so the two cannot drift apart silently.

Adding or renaming a category means changing three things: this file,
config.yaml, and prompts/<version>.txt. The first two are checked against each
other automatically; the prompt is prose and is not, so update it deliberately.
"""
from typing import Literal, get_args

Category = Literal['flaky_test', 'genuine_regression', 'infra', 'transient']

CATEGORIES: tuple[str, ...] = get_args(Category)
"""Runtime tuple of the same values, for code that iterates the categories."""