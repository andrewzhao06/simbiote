"""Convenience entrypoint: `python scripts/teleop/run_demo.py [args]`.

Equivalent to `python -m simbiote.teleop.teleop_session [args]`; see that
module's argparse setup for options (--sink console|pybullet, --camera-index,
--preview, --save, etc).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simbiote.teleop.teleop_session import main  # noqa: E402

if __name__ == "__main__":
    main()
