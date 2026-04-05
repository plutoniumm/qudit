#!/usr/bin/env python3
"""Code style linter for qudit. Run via ./do test."""
import ast
import re
import sys
import tokenize
import io
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY_FILES = sorted(ROOT.glob("qudit/**/*.py")) + sorted(ROOT.glob("tests/*.py"))

errors: list[str] = []
fixes = "--fix" in sys.argv


def E(path, line, msg):
    errors.append(f"{path}:{line}: {msg}")


# ── Python ────────────────────────────────────────────────────────────────────

def _is_docstring(node) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def check_py(path: Path):
    src = path.read_text()
    lines = src.splitlines()

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        E(path, exc.lineno or 0, f"SyntaxError: {exc.msg}")
        return

    _check_docstrings(path, tree, lines)
    _check_returns(path, tree, lines, src)
    _check_assert_calls(path, tree, lines)
    _check_inline_dicts(path, tree, lines)
    _check_trailing_comma(path, tree, src)
    _check_semicolons(path, src)
    _check_type_ignore(path, src)


def _check_docstrings(path: Path, tree, lines: list[str]):
    escape_violations: list[tuple[int, int]] = []
    inline_violations: list[int] = []  # linenos where text starts on same line as """

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            continue
        if not (node.body and _is_docstring(node.body[0])):
            continue

        ds = node.body[0]
        lineno = ds.lineno
        raw = lines[lineno - 1]

        # Rule 1: text must not start on same line as opening """
        stripped = raw.lstrip()
        for q in ('"""', "'''"):
            if stripped.startswith(q):
                after = stripped[len(q):]
                if after.strip() and not after.strip().startswith(q):
                    E(path, lineno, f"Docstring text on same line as opening {q}")
                    inline_violations.append(lineno)
                break

        # Rule 2: escape sequences \n \r \t \b \f \v \a must be doubled in docstrings
        end = ds.end_lineno
        ds_src = "\n".join(lines[lineno - 1: end])
        for m in re.finditer(r'(?<!\\)\\([nrtbfva])(?!\\)', ds_src):
            offset = ds_src[: m.start()].count("\n")
            actual_line = lineno + offset
            char = m.group(1)
            escape_violations.append((actual_line, m.start() - ds_src.rfind("\n", 0, m.start()) - 1))
            E(path, actual_line, f"Docstring contains \\{char}; use \\\\{char} for literal escape")

    if fixes and inline_violations:
        cur = path.read_text().splitlines()
        for lineno in sorted(set(inline_violations), reverse=True):
            raw = cur[lineno - 1]
            indent = len(raw) - len(raw.lstrip())
            pad = " " * indent
            stripped = raw.lstrip()
            for q in ('"""', "'''"):
                if stripped.startswith(q):
                    inner = stripped[len(q):]
                    close_q = q
                    # strip closing quotes if on same line (single-line docstring)
                    if inner.rstrip().endswith(q):
                        inner = inner.rstrip()[: -len(q)].strip()
                    else:
                        inner = inner.strip()
                    cur[lineno - 1] = f'{pad}{q}'
                    cur.insert(lineno, f'{pad}{close_q}')
                    cur.insert(lineno, f'{pad}{inner}')
                    break
        path.write_text("\n".join(cur) + "\n")
        print(f"fixed  {path} ({len(set(inline_violations))} docstring(s))")

    if fixes and escape_violations:
        text = path.read_text()
        ranges: list[tuple[int, int]] = []
        cur_tree = ast.parse(text)
        for node in ast.walk(cur_tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                continue
            if node.body and _is_docstring(node.body[0]):
                ds = node.body[0]
                ranges.append((ds.lineno, ds.end_lineno))

        new_lines = text.splitlines()
        for start, end in ranges:
            for i in range(start - 1, end):
                new_lines[i] = re.sub(r'(?<!\\)\\([nrtbfva])(?!\\)', r'\\\\\1', new_lines[i])

        new_text = "\n".join(new_lines) + "\n"
        if new_text != text:
            path.write_text(new_text)
            print(f"fixed  {path} (docstring escapes)")


def _check_returns(path: Path, tree, lines: list[str], src: str):
    """
    Flag every return whose immediately preceding non-empty line is not a block
    opener (ends with ':'), a docstring close (ends with triple-quote), or a comment.
    This catches returns inside nested if/for/else blocks too.
    """
    violations: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue

        rln = node.lineno  # 1-indexed
        if rln < 2:
            continue

        prev = lines[rln - 2].strip()

        if not prev:
            continue  # already has blank line
        if prev.endswith(":"):
            continue  # block opener — return is first stmt in block
        if prev.endswith('"""') or prev.endswith("'''"):
            continue  # closing docstring — single-stmt body
        if prev.startswith("#"):
            continue  # comment-only line before return

        violations.append(rln)

    if not violations:
        return

    if fixes:
        current_lines = path.read_text().splitlines()
        for rln in sorted(set(violations), reverse=True):
            current_lines.insert(rln - 1, "")
        path.write_text("\n".join(current_lines) + "\n")
        print(f"fixed  {path} ({len(set(violations))} return(s))")
    else:
        for rln in sorted(set(violations)):
            E(path, rln, "Missing blank line before return")


def _check_inline_dicts(path: Path, tree, lines: list[str]):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        if len(node.keys) < 2:
            continue
        if node.lineno == node.end_lineno:
            E(path, node.lineno, f"Inline dict with {len(node.keys)} keys — expand to one key per line")


def _check_trailing_comma(path: Path, tree, src: str):
    skip = {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        if node.lineno == node.end_lineno or not node.keys:
            continue

        rbrace_idx = next(
            (i for i, t in enumerate(toks)
             if t.start[0] == node.end_lineno and t.type == tokenize.OP and t.string == "}"),
            None,
        )
        if rbrace_idx is None:
            continue

        j = rbrace_idx - 1
        while j >= 0 and toks[j].type in skip:
            j -= 1

        if j >= 0 and not (toks[j].type == tokenize.OP and toks[j].string == ","):
            E(path, toks[j].end[0], "Multi-line dict missing trailing comma after last entry")


def _check_type_ignore(path: Path, src: str):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return

    found = [t for t in toks if t.type == tokenize.COMMENT and "type: ignore" in t.string]
    if not found:
        return

    if fixes:
        new_lines = src.splitlines()
        for tok in found:
            ln = tok.start[0] - 1
            new_lines[ln] = re.sub(r"\s*#\s*type:\s*ignore[^\n]*", "", new_lines[ln]).rstrip()
        path.write_text("\n".join(new_lines) + "\n")
        print(f"fixed  {path} ({len(found)} type: ignore comment(s))")
    else:
        for tok in found:
            E(path, tok.start[0], "type: ignore comment — fix the type error instead")


_ASSERT_ATTRS = {"stateEqual"}


def _is_assert_call(node) -> bool:
    if isinstance(node, ast.Assert):
        return True
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return False
    func = node.value.func
    attr = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)

    return attr is not None and (attr.startswith("assert") or attr in _ASSERT_ATTRS)


def _check_assert_calls(path: Path, tree, lines: list[str]):
    violations: list[int] = []

    for node in ast.walk(tree):
        if not _is_assert_call(node):
            continue

        aln = node.lineno
        if aln < 2:
            continue

        prev = lines[aln - 2].strip()

        if not prev:
            continue
        if prev.endswith(":"):
            continue
        if prev.endswith('"""') or prev.endswith("'''"):
            continue
        if prev.startswith("#"):
            continue

        violations.append(aln)

    if not violations:
        return

    if fixes:
        current_lines = path.read_text().splitlines()
        for aln in sorted(set(violations), reverse=True):
            current_lines.insert(aln - 1, "")
        path.write_text("\n".join(current_lines) + "\n")
        print(f"fixed  {path} ({len(set(violations))} assert(s))")
    else:
        for aln in sorted(set(violations)):
            E(path, aln, "Missing blank line before assert")


def _check_semicolons(path: Path, src: str):
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return

    for tok in tokens:
        if tok.type == tokenize.OP and tok.string == ";":
            E(path, tok.start[0], "Semicolon separating statements on one line — split onto separate lines")


# ── Main ──────────────────────────────────────────────────────────────────────

for p in PY_FILES:
    check_py(p)

if errors:
    for e in sorted(errors):
        print(e)
    sys.exit(1)

print(f"ok  {len(PY_FILES)} py")
