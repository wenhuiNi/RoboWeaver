"""Explicit task files and user-supplied Python verifier entry points."""

import importlib
from pathlib import Path

from roboweaver.contracts import TaskSpec


def load_task(path: Path) -> TaskSpec:
    task = TaskSpec.model_validate_json(path.read_text())
    load_verifier(task.verifier)
    return task


def load_verifier(reference: str):
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError("Verifier must use module:function syntax")
    verifier = getattr(importlib.import_module(module), name)
    if not callable(verifier):
        raise ValueError("Verifier entry point must be callable")
    return verifier
