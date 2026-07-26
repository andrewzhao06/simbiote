"""Convenience entrypoint: `python run_teleop_demo.py [args]`.

Equivalent to `python -m factoryflow.teleop.teleop_session [args]`; see that
module's argparse setup for options (--sink console|pybullet, --camera-index,
--preview, --save, etc).
"""

from factoryflow.teleop.teleop_session import main

if __name__ == "__main__":
    main()
