from utils import read_ignore, isIgnore, cleanup, make_tree
from pathlib import Path
import ast
import json
import textwrap


def fmt_doc(doc: str | None) -> str | None:
    if doc is None:
        return None

    cleaned = textwrap.dedent(doc).strip()

    return cleaned or None


def isStatic(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "staticmethod":
            return True

        if isinstance(dec, ast.Attribute) and dec.attr == "staticmethod":
            return True

        if ast.unparse(dec) == "staticmethod":
            return True

    return False


def parse(path: Path) -> ast.AST | None:
    text = path.read_text(encoding="utf-8")

    return ast.parse(text, filename=str(path))


def format(node: ast.AST) -> str | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    args: list[str] = []
    for a in node.args.args:
        ann = ast.unparse(a.annotation) if a.annotation else "Any"
        args.append(f"{a.arg}: {ann}")

    returns = ast.unparse(node.returns) if node.returns is not None else "Any"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"

    return f"{prefix} {node.name}({', '.join(args)}) -> {returns}"


def try_un(value: ast.AST | None, default: str = "Any") -> str:
    if value is None:
        return default

    return ast.unparse(value)


def class_attr(node: ast.ClassDef) -> list[tuple[str, str, str]]:
    """
    Return (name, annotation, default) for class-scope attributes.

    Supports:
      - AnnAssign: x: T = v
      - Assign: x = v (type unknown -> "None")
    """
    attrs: list[tuple[str, str, str]] = []

    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            ann = try_un(stmt.annotation, default="Any")
            default = (
                try_un(stmt.value, default="None") if stmt.value is not None else "None"
            )
            attrs.append((name, ann, default))
            continue

        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    name = t.id
                    ann = "None"
                    default = (
                        try_un(stmt.value, default="None")
                        if stmt.value is not None
                        else "None"
                    )
                    attrs.append((name, ann, default))

    attrs.sort(key=lambda x: x[0])
    return attrs


allow_func = lambda name: name == "__init__" or not name.startswith("_")


def extract_symbols(path: Path) -> tuple[
    list[tuple[ast.ClassDef, str | None, str, list[tuple[str, str | None, str]]]],
    list[tuple[str, str | None, str]],
]:
    tree = parse(path)
    if tree is None:
        return ([], [])

    assert isinstance(tree, ast.Module)
    src_text = path.read_text(encoding="utf-8")

    classes: list[
        tuple[ast.ClassDef, str | None, str, list[tuple[str, str | None, str]]]
    ] = []
    global_funcs: list[tuple[str, str | None, str]] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cdoc = fmt_doc(ast.get_docstring(node, clean=False))
            csrc = ast.get_source_segment(src_text, node) or ""

            methods: list[tuple[str, str | None, str]] = []
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not allow_func(stmt.name):
                        continue

                    mh = format(stmt)
                    if mh is None:
                        continue

                    if isStatic(stmt):
                        mh = f"{mh} [static]"

                    mdoc = fmt_doc(ast.get_docstring(stmt, clean=False))
                    msrc = ast.get_source_segment(src_text, stmt) or ""
                    methods.append((mh, mdoc, msrc))

            classes.append((node, cdoc, csrc, methods))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not allow_func(node.name):
                continue

            fh = format(node)

            if fh is None:
                continue

            fdoc = fmt_doc(ast.get_docstring(node, clean=False))
            fsrc = ast.get_source_segment(src_text, node) or ""
            global_funcs.append((fh, fdoc, fsrc))

    return (classes, global_funcs)


def attr_table(attrs: list[tuple[str, str, str]]) -> list[str]:
    lines: list[str] = []
    if not attrs:
        return lines

    lines.append("| name | type | default |")
    lines.append("|---|---|---|")

    for name, ann, default in attrs:
        lines.append(f"| {name} | {ann} | {default} |")

    lines.append("")
    return lines


def fmt_impl(code: str) -> str:
    """Return code with docstrings removed from modules, classes and functions.

    This preserves the rest of the implementation for display in the docs.
    If parsing fails for any reason, returns the original code.
    """

    def _strip_in_body(body: list[ast.stmt]) -> list[ast.stmt]:
        if not body:
            return body

        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body = body[1:]

        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stmt.body = _strip_in_body(stmt.body)

        return body

    tree = ast.parse(code)
    assert isinstance(tree, ast.Module)
    tree.body = _strip_in_body(tree.body)
    ast.fix_missing_locations(tree)

    return ast.unparse(tree).rstrip() + "\n"


def fmt_callable(
    header: str, doc: str | None, code: str, heading_level: int
) -> list[str]:
    lines: list[str] = []

    name = header
    if header.startswith("async def "):
        name = header[len("async def ") :]
    elif header.startswith("def "):
        name = header[len("def ") :]
    name = name.split("(", 1)[0].strip() or header

    lines.append(f"{'#' * heading_level} {name}")
    lines.append("")

    if doc:
        lines.append(doc)
    else:
        lines.append("No Definition provided")
    lines.append("")

    lines.append("```py")
    lines.append(header)
    lines.append("```")
    lines.append("")

    lines.append("<details>")
    lines.append("<summary>Implementation</summary>")
    lines.append("")
    lines.append("<div>")
    lines.append("")

    impl = fmt_impl(code.rstrip())

    lines.append("```python")
    lines.append(impl.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("</div>")
    lines.append("</details>")
    lines.append("")

    return lines


def render(
    file_rel: Path,
    classes: list[
        tuple[ast.ClassDef, str | None, str, list[tuple[str, str | None, str]]]
    ],
    functions: list[tuple[str, str | None, str]],
) -> str:
    lines: list[str] = []
    lines.append(f"# {file_rel.as_posix()}")
    lines.append("")

    if not classes and not functions:
        lines.append("*(No functions or classes found.)*")
        lines.append("")
        return "\n".join(lines)

    for cnode, cdoc, _csrc, methods in classes:
        lines.append(f"## class {cnode.name}")
        lines.append("")

        if cdoc and cdoc.strip():
            lines.append(cdoc.strip())
        else:
            lines.append("No Definition provided")
        lines.append("")

        attrs = class_attr(cnode)
        lines.extend(attr_table(attrs))

        for mh, mdoc, mcode in methods:
            lines.extend(fmt_callable(mh, mdoc, mcode, heading_level=3))

    if functions:
        for header, doc, code in functions:
            lines.extend(fmt_callable(header, doc, code, heading_level=2))

    return "\n".join(lines)


def generate(
    repo_root: str | Path,
    src_dir: str | Path = "qudit",
    out_dir: str | Path = "docs/code",
    ignore: list[str] | None = None,
) -> None:
    root = Path(repo_root)
    src = (root / src_dir).resolve()
    out = (root / out_dir).resolve()

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")

    patterns: list[str] = []

    ignores = Path(__file__).resolve().parent / "ignore.txt"
    patterns.extend(read_ignore(root, ignores.parent, ignore_file="ignore.txt"))

    normalized: list[str] = []
    for pat in patterns:
        p = pat.strip()
        if p.startswith("**/"):
            normalized.append(p[3:])
        normalized.append(p)
    patterns = normalized

    if ignore:
        patterns.extend(ignore)

    cleanup(out)

    files_json = out / "files.json"
    files_json.write_text(
        json.dumps(make_tree(src, patterns, suffix=".py"), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for py in sorted(src.rglob("*.py")):
        rel = py.relative_to(src)
        rel_posix = rel.as_posix()

        if isIgnore(rel_posix, patterns):
            continue

        classes, functions = extract_symbols(py)

        md_path = out / rel.with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render(rel, classes, functions), encoding="utf-8")


if __name__ == "__main__":
    here = Path(__file__).resolve()
    repo_root = here.parents[1]

    generate(repo_root=repo_root, src_dir="qudit", out_dir="docs/code")
