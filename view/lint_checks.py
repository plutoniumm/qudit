import ast
import io
import re
import tokenize

from .lint_ctx import Ctx, _DIVIDER_RE, is_assert_call, is_docstring


def check_py(args: tuple) -> tuple[list[str], list[str]]:
    from pathlib import Path

    path, fixes = args
    ctx = Ctx(path, fixes)
    src = path.read_text()
    lines = src.splitlines()

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        ctx.E(exc.lineno or 0, f"SyntaxError: {exc.msg}")

        return ctx.errs, ctx.msgs

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        toks = []

    file_start(ctx, src)
    docstrings(ctx, tree, lines)
    returns(ctx, tree, lines)
    return_after(ctx, tree, lines)

    assert_calls(ctx, tree, lines)

    assert_messages(ctx, tree)
    inline_dicts(ctx, tree)
    trailing_comma(ctx, tree, toks)
    divider_comments(ctx, toks, src)
    semicolons(ctx, toks)
    type_ignore(ctx, toks, src)

    return ctx.errs, ctx.msgs


def file_start(ctx: Ctx, src: str):
    for i, line in enumerate(src.splitlines()):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#!"):
            continue
        if s.startswith("#") or s.startswith('"""') or s.startswith("'''"):
            ctx.E(i + 1, "File starts with a comment — begin with imports instead")
        break


def docstrings(ctx: Ctx, tree, lines: list[str]):
    escape_violations: list[tuple[int, int]] = []
    inline_violations: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            continue
        if not (node.body and is_docstring(node.body[0])):
            continue

        ds = node.body[0]
        lineno = ds.lineno
        raw = lines[lineno - 1]

        stripped = raw.lstrip()
        for q in ('"""', "'''"):
            if stripped.startswith(q):
                after = stripped[len(q) :]
                if after.strip() and not after.strip().startswith(q):
                    ctx.E(lineno, f"Docstring text on same line as opening {q}")
                    inline_violations.append(lineno)
                break

        end = ds.end_lineno
        ds_src = "\n".join(lines[lineno - 1 : end])
        for m in re.finditer(r"(?<!\\)\\([nrtbfva])(?!\\)", ds_src):
            offset = ds_src[: m.start()].count("\n")
            actual_line = lineno + offset
            char = m.group(1)
            escape_violations.append(
                (actual_line, m.start() - ds_src.rfind("\n", 0, m.start()) - 1)
            )
            ctx.E(
                actual_line,
                f"Docstring contains \\{char}; use \\\\{char} for literal escape",
            )

    if ctx.fixes and inline_violations:
        cur = ctx.path.read_text().splitlines()
        for lineno in sorted(set(inline_violations), reverse=True):
            raw = cur[lineno - 1]
            indent = len(raw) - len(raw.lstrip())
            pad = " " * indent
            stripped = raw.lstrip()
            for q in ('"""', "'''"):
                if stripped.startswith(q):
                    inner = stripped[len(q) :]
                    if inner.rstrip().endswith(q):
                        inner = inner.rstrip()[: -len(q)].strip()
                        cur[lineno - 1] = f"{pad}{q}"
                        cur.insert(lineno, f"{pad}{q}")
                        cur.insert(lineno, f"{pad}{inner}")
                    else:
                        cur[lineno - 1] = f"{pad}{q}"
                        cur.insert(lineno, f"{pad}{inner.strip()}")
                    break
        ctx.path.write_text("\n".join(cur) + "\n")
        ctx.fix(f"{len(set(inline_violations))} docstring(s)")

    if ctx.fixes and escape_violations:
        text = ctx.path.read_text()
        ranges: list[tuple[int, int]] = []
        cur_tree = ast.parse(text)
        for node in ast.walk(cur_tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            ):
                continue
            if node.body and is_docstring(node.body[0]):
                ds = node.body[0]
                ranges.append((ds.lineno, ds.end_lineno))

        new_lines = text.splitlines()
        for start, end in ranges:
            for i in range(start - 1, end):
                new_lines[i] = re.sub(
                    r"(?<!\\)\\([nrtbfva])(?!\\)", r"\\\\\1", new_lines[i]
                )

        new_text = "\n".join(new_lines) + "\n"
        if new_text != text:
            ctx.path.write_text(new_text)
            ctx.fix("docstring escapes")


def returns(ctx: Ctx, tree, lines: list[str]):
    violations: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue

        rln = node.lineno
        if rln < 2:
            continue

        prev = lines[rln - 2].strip()

        if not prev:
            continue
        if prev.endswith(":"):
            continue
        if prev.endswith('"""') or prev.endswith("'''"):
            continue
        if prev.startswith("#"):
            continue

        violations.append(rln)

    if not violations:
        return

    if ctx.fixes:
        current_lines = ctx.path.read_text().splitlines()
        for rln in sorted(set(violations), reverse=True):
            current_lines.insert(rln - 1, "")
        ctx.path.write_text("\n".join(current_lines) + "\n")
        ctx.fix(f"{len(set(violations))} return(s)")
    else:
        for rln in sorted(set(violations)):
            ctx.E(rln, "Missing blank line before return")


def return_after(ctx: Ctx, tree, lines: list[str]):
    violations: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        rets = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
        if len(rets) < 2:
            continue
        for ret in sorted(rets, key=lambda r: r.end_lineno)[:-1]:
            after_idx = ret.end_lineno
            if after_idx >= len(lines):
                continue
            nxt = lines[after_idx].strip()
            if not nxt:
                continue
            if nxt.startswith(("else", "elif", "except", "finally")):
                continue
            violations.append(ret.end_lineno)

    if not violations:
        return

    if ctx.fixes:
        current_lines = ctx.path.read_text().splitlines()
        for ins in sorted(set(violations), reverse=True):
            current_lines.insert(ins, "")
        ctx.path.write_text("\n".join(current_lines) + "\n")
        ctx.fix(f"{len(set(violations))} return-after(s)")
    else:
        for ins in sorted(set(violations)):
            ctx.E(ins, "Missing blank line after return in multi-return function")


def assert_calls(ctx: Ctx, tree, lines: list[str]):
    violations: list[int] = []

    for node in ast.walk(tree):
        if not is_assert_call(node):
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

    if ctx.fixes:
        current_lines = ctx.path.read_text().splitlines()
        for aln in sorted(set(violations), reverse=True):
            current_lines.insert(aln - 1, "")
        ctx.path.write_text("\n".join(current_lines) + "\n")
        ctx.fix(f"{len(set(violations))} assert(s)")
    else:
        for aln in sorted(set(violations)):
            ctx.E(aln, "Missing blank line before assert")


def assert_messages(ctx: Ctx, tree):
    from .lint_ctx import _ASSERT_ATTRS

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            if node.msg is None:
                ctx.E(node.lineno, "assert missing failure message")
            continue

        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            continue
        func = node.value.func
        attr = (
            func.attr
            if isinstance(func, ast.Attribute)
            else (func.id if isinstance(func, ast.Name) else None)
        )
        is_camel_assert = attr.startswith("assert") and (
            len(attr) <= 6 or attr[6].isupper()
        )
        if attr is None or (not is_camel_assert and attr not in _ASSERT_ATTRS):
            continue

        call = node.value
        has_msg = any(kw.arg == "msg" for kw in call.keywords)
        if not has_msg and call.args:
            last = call.args[-1]
            has_msg = isinstance(last, ast.JoinedStr) or (
                isinstance(last, ast.Constant) and isinstance(last.value, str)
            )

        if not has_msg:
            ctx.E(node.lineno, "assert missing msg= argument")


def inline_dicts(ctx: Ctx, tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        if len(node.keys) < 2:
            continue
        if node.lineno == node.end_lineno:
            ctx.E(
                node.lineno,
                f"Inline dict with {len(node.keys)} keys — expand to one key per line",
            )


def trailing_comma(ctx: Ctx, tree, toks: list):
    skip = {
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        if node.lineno == node.end_lineno or not node.keys:
            continue

        rbrace_idx = next(
            (
                i
                for i, t in enumerate(toks)
                if t.start[0] == node.end_lineno
                and t.type == tokenize.OP
                and t.string == "}"
            ),
            None,
        )
        if rbrace_idx is None:
            continue

        j = rbrace_idx - 1
        while j >= 0 and toks[j].type in skip:
            j -= 1

        if j >= 0 and not (toks[j].type == tokenize.OP and toks[j].string == ","):
            ctx.E(
                toks[j].end[0],
                "Multi-line dict missing trailing comma after last entry",
            )


def divider_comments(ctx: Ctx, toks: list, src: str):
    found = [
        t
        for t in toks
        if t.type == tokenize.COMMENT and _DIVIDER_RE.search(t.string[1:])
    ]
    if not found:
        return

    if ctx.fixes:
        new_lines = src.splitlines()
        for tok in sorted(found, reverse=True, key=lambda t: t.start[0]):
            ln = tok.start[0] - 1
            stripped = new_lines[ln][: tok.start[1]].rstrip()
            if stripped:
                new_lines[ln] = stripped
            else:
                del new_lines[ln]
        ctx.path.write_text("\n".join(new_lines) + "\n")
        ctx.fix(f"{len(found)} divider comment(s)")
    else:
        for tok in found:
            ctx.E(tok.start[0], "Divider comment — remove it")


def semicolons(ctx: Ctx, toks: list):
    for tok in toks:
        if tok.type == tokenize.OP and tok.string == ";":
            ctx.E(
                tok.start[0],
                "Semicolon separating statements on one line — split onto separate lines",
            )


def type_ignore(ctx: Ctx, toks: list, src: str):
    found = [
        t for t in toks if t.type == tokenize.COMMENT and "type: ignore" in t.string
    ]
    if not found:
        return

    if ctx.fixes:
        new_lines = src.splitlines()
        for tok in found:
            ln = tok.start[0] - 1
            new_lines[ln] = re.sub(
                r"\s*#\s*type:\s*ignore[^\n]*", "", new_lines[ln]
            ).rstrip()
        ctx.path.write_text("\n".join(new_lines) + "\n")
        ctx.fix(f"{len(found)} type: ignore comment(s)")
    else:
        for tok in found:
            ctx.E(tok.start[0], "type: ignore comment — fix the type error instead")
