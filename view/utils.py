from __future__ import annotations
from pathlib import Path
import fnmatch
import shutil


def cleanup(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)


def _add_path(tree: dict, parts: list[str], value: str) -> None:
    cur = tree
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def fmt_title(s: str) -> str:
    parts = [p for p in s.replace("_", " ").replace("-", " ").split() if p]

    return " ".join(p.capitalize() for p in parts) if parts else s


def fmt_link(rel_posix: str, *, prefix: str = "code/") -> str:
    p = rel_posix.lstrip("./")

    if p.endswith(".py"):
        p = p[: -len(".py")]

    return f"{prefix}{p}.md"


def toVitepress(node: dict, *, prefix: str = "code/") -> list[dict]:
    items: list[dict] = []

    for name in sorted(node.keys()):
        child = node[name]
        if isinstance(child, dict):
            items.append(
                {
                    "text": fmt_title(name),
                    "collapsible": True,
                    "collapsed": True,
                    "items": toVitepress(child, prefix=prefix),
                }
            )
        else:
            rel_posix = str(child)
            stem = Path(rel_posix).name

            if stem.endswith(".py"):
                stem = stem[: -len(".py")]

            items.append(
                {
                    "text": stem,
                    "link": fmt_link(rel_posix, prefix=prefix),
                }
            )

    return items


def make_tree(src: Path, patterns: list[str], *, suffix: str = ".py") -> list[dict]:
    tree: dict = {}

    for p in sorted(src.rglob(f"*{suffix}")):
        rel = p.relative_to(src)
        rel_posix = rel.as_posix()
        if isIgnore(rel_posix, patterns):
            continue

        parts = rel_posix.split("/")
        _add_path(tree, parts, rel_posix)

    return toVitepress(tree, prefix="code/")


def read_ignore(repo_root: Path, src: Path, ignore_file: str) -> list[str]:
    p = (
        (repo_root / src.relative_to(repo_root) / ignore_file)
        if src.is_absolute()
        else (repo_root / src / ignore_file)
    )

    if not p.exists():
        p = src / ignore_file
    if not p.exists() or not p.is_file():
        return []

    lines: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)

    return lines


def isIgnore(rel_posix: str, patterns: list[str]) -> bool:
    rel_posix = rel_posix.lstrip("./")
    for pat in patterns:
        p = pat.strip()
        if not p:
            continue

        if p.endswith("/"):
            base = p.rstrip("/")
            if rel_posix == base or rel_posix.startswith(base + "/"):
                return True

        if fnmatch.fnmatch(rel_posix, p):
            return True

        name = rel_posix.split("/")[-1]
        if fnmatch.fnmatch(name, p.rstrip("/")):
            return True

    return False
