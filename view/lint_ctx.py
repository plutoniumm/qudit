import ast
import re
from pathlib import Path

_ASSERT_ATTRS = {"stateEqual", "matEqual"}
_DIVIDER_RE = re.compile(r"(.)\1{4,}")


class Ctx:
    __slots__ = ("path", "fixes", "errs", "msgs")

    def __init__(self, path: Path, fixes: bool):
        self.path = path
        self.fixes = fixes
        self.errs: list[str] = []
        self.msgs: list[str] = []

    def E(self, line, msg):
        self.errs.append(f"{self.path}:{line}: {msg}")

    def fix(self, msg):
        self.msgs.append(f"fixed  {self.path} ({msg})")


def is_docstring(node) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def is_assert_call(node) -> bool:
    if isinstance(node, ast.Assert):
        return True

    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return False

    func = node.value.func
    attr = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)

    is_camel_assert = attr.startswith("assert") and (len(attr) <= 6 or attr[6].isupper())

    return attr is not None and (is_camel_assert or attr in _ASSERT_ATTRS)
