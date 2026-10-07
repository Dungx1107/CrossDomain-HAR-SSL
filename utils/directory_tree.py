from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# directory_tree.py nằm trong:
#
# CrossDomain-HAR-SSL/
# └── utils/
#     └── directory_tree.py
#
# .parent       -> utils/
# .parent.parent -> CrossDomain-HAR-SSL/

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# PRINT TREE
# ============================================================

def print_tree(path: Path, prefix=""):
    """
    In toàn bộ cấu trúc thư mục và file bên trong path.
    Không bỏ qua bất kỳ file/thư mục nào.
    """

    try:
        children = sorted(
            path.iterdir(),
            key=lambda p: (
                not p.is_dir(),   # Folder trước, file sau
                p.name.lower()
            )
        )

    except PermissionError:
        print(f"{prefix}└── [Permission Denied]")
        return

    for index, child in enumerate(children):

        is_last = index == len(children) - 1

        # Ký hiệu tree
        connector = "└── " if is_last else "├── "

        print(f"{prefix}{connector}{child.name}")

        # Nếu là thư mục -> đệ quy
        if child.is_dir():

            next_prefix = prefix + (
                "    " if is_last else "│   "
            )

            print_tree(child, next_prefix)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # PATH BÊN TRONG PROJECT
    # ========================================================

    path_string = "checkpoints/cross_domain_fewshot/crosshar/cnn_transformer"

    # Ghép với PROJECT_ROOT
    root = PROJECT_ROOT / path_string

    # Chuẩn hóa đường dẫn
    root = root.resolve()

    # ========================================================
    # VALIDATE
    # ========================================================

    if not root.exists():
        print("\n[ERROR] Đường dẫn không tồn tại:")
        print(root)
        return

    if not root.is_dir():
        print("\n[ERROR] Đường dẫn không phải là thư mục:")
        print(root)
        return

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print(root)
    print_tree(root)


if __name__ == "__main__":
    main()

