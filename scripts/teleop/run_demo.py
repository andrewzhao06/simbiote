"""Convenience entrypoint: `python scripts/teleop/run_demo.py [args]`.

Equivalent to `python -m simbiote.teleop.teleop_session [args]`; see that
module's argparse setup for options (--sink console|pybullet, --camera-index,
--preview, --save, etc).
"""

import sys
from pathlib import Path

# These scripts are run by path (often under Isaac Sim's bundled interpreter,
# where the package isn't installed), so put the repo on sys.path. Located by
# walking up to pyproject.toml rather than by a fixed parent count, which
# silently breaks the moment the file moves between script subdirectories.
REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from simbiote.teleop.teleop_session import main  # noqa: E402

if __name__ == "__main__":
    main()
