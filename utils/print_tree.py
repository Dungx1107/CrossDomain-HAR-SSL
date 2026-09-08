from pathlib import Path

ROOT = Path("..")

IGNORE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    "experiments",
    "document",
    "data",
}

IGNORE_FILES = {
    "*.pyc",
    "*.pt",
    "*.pth",
}


def should_ignore(path: Path):
    if path.is_dir() and path.name in IGNORE_DIRS:
        return True

    if path.is_file():
        for pattern in IGNORE_FILES:
            if path.match(pattern):
                return True

    return False


def print_tree(path: Path, prefix=""):
    if should_ignore(path):
        return

    children = sorted(
        [p for p in path.iterdir() if not should_ignore(p)],
        key=lambda p: (not p.is_dir(), p.name.lower())
    )

    for i, child in enumerate(children):
        is_last = i == len(children) - 1

        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{child.name}")

        if child.is_dir():
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(child, next_prefix)


print(ROOT.resolve())
print_tree(ROOT)
