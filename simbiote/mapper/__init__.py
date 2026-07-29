"""Module 1: Stray Scanner capture → OpenUSD scene + scene graph.

`cli.py` is the `simbiote-map` entry point; `pipeline.run_pipeline` chains the
five stages in `stages.py` and hands off through `usd.validate_map`.

The package version lives on `simbiote.__version__`; this subpackage does not
carry one of its own (the two used to disagree).
"""

from simbiote.mapper.models import CaptureBundle, SceneGraph

__all__ = ["CaptureBundle", "SceneGraph"]
