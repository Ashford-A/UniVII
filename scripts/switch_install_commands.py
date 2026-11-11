"""After the first PyPI release, replace the install-from-GitHub commands with the PyPI name.

    python scripts/switch_install_commands.py --dry-run
    python scripts/switch_install_commands.py

Rewrites ``univii @ git+https://github.com/Ashford-A/UniVII.git`` (with or without an
extra such as ``[tutorials]``) to ``univii`` in the README, the documentation and the
notebooks' Colab install cells. Uses only the standard library. Review ``git diff`` afterwards.
"""
from pathlib import Path
import argparse
import re

PATTERN = re.compile(r'univii(\[[a-z,]+\])? @ git\+https://github\.com/[A-Za-z0-9_.-]+/UniVII\.git(@[A-Za-z0-9_.-]+)?')
SUFFIXES = {".md", ".ipynb"}
SKIP_DIRS = {".git", "notebooks", "_build", "generated", "dist", "build"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="list the files that would change")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    changed = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or path.suffix not in SUFFIXES or SKIP_DIRS & set(rel.parts) or rel.name == "RELEASE.md":
            continue
        text = path.read_text(encoding="utf-8")
        new = PATTERN.sub(lambda m: "univii" + (m.group(1) or ""), text)
        if new != text:
            changed.append(rel)
            if not args.dry_run:
                path.write_text(new, encoding="utf-8")
    print(f"{len(changed)} files {'would change' if args.dry_run else 'changed'}:")
    for rel in changed:
        print("  ", rel)
    if changed and not args.dry_run:
        print("Now reword the sentences around those commands (README 'Installation', docs/installation.md), "
              "which still say UniVII is not on PyPI.")


if __name__ == "__main__":
    main()
