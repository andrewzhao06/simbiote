"""Rewrite MANO_RIGHT.pkl without chumpy so smplx can load it on Python 3.12.

The MANO release pickles several of its arrays as `chumpy.ch.Ch` objects.
`smplx.MANOLayer.__init__` does a plain `pickle.load(...)`, which needs those
classes to be importable -- and chumpy is dead upstream: it imports
`numpy.bool`/`numpy.object` (removed in numpy 2) and `inspect.getargspec`
(removed in Python 3.11), so it cannot be installed here at all.

Nothing in WiLoR's inference path actually uses chumpy's autodiff; it only
reads the underlying arrays. So we unpickle with stub classes standing in for
the chumpy types, unwrap each stub to its plain numpy array, and write the
result back out. The rewritten file is a drop-in for the original.

Usage:
    python scripts/gb10/teleop/dechumpify_mano.py \
        --src /home/dell/AI/models/wilor/mano_data/mano/MANO_RIGHT.pkl \
        --out <repo>/simbiote/assets/mano/MANO_RIGHT.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

DEFAULT_SRC = Path("/home/dell/AI/models/wilor/mano_data/mano/MANO_RIGHT.pkl")


class _ChumpyStub:
    """Stands in for any chumpy class encountered during unpickling.

    Chumpy objects restore through `__setstate__`/`__dict__`, so we just
    capture whatever state pickle hands us and let `_unwrap` evaluate it
    back down to a plain array afterwards.
    """

    _chumpy_class = "chumpy"

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


_STUBS: dict[str, type] = {}


class _ChumpyUnpickler(pickle.Unpickler):
    """Maps every `chumpy.*` class to a per-class `_ChumpyStub` subclass.

    Keeping one stub type per chumpy class lets `_unwrap` tell a plain value
    node (`chumpy.ch.Ch`) apart from a lazy op node (`chumpy.reordering.Select`),
    which have to be evaluated differently.
    """

    def find_class(self, module, name):
        if module.startswith("chumpy"):
            qualname = f"{module}.{name}"
            if qualname not in _STUBS:
                _STUBS[qualname] = type(name, (_ChumpyStub,), {"_chumpy_class": qualname})
            return _STUBS[qualname]
        return super().find_class(module, name)


def _unwrap(obj):
    """Recursively replace chumpy stubs with the numpy arrays they evaluate to."""

    if isinstance(obj, _ChumpyStub):
        state = obj.__dict__

        # chumpy.reordering.Select is a lazy gather-and-reshape over another
        # node. In MANO_RIGHT.pkl this is how `shapedirs` slices the stored
        # 20 shape components down to the 10 the model actually uses.
        # chumpy's own compute_r() is: a.r.ravel()[idxs].reshape(preferred_shape)
        if obj._chumpy_class.endswith("reordering.Select"):
            source = np.asarray(_unwrap(state["a"]))
            idxs = np.asarray(state["idxs"]).ravel()
            return source.ravel()[idxs].reshape(state["preferred_shape"])

        # A plain chumpy.ch.Ch value node keeps its array in `x`.
        for key in ("x", "r", "_state"):
            if key in state:
                return _unwrap(state[key])
        raise ValueError(
            f"unhandled chumpy node {obj._chumpy_class} with fields {sorted(state)}"
        )
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_unwrap(v) for v in obj]
        return type(obj)(converted)
    return obj


def dechumpify(src: Path, out: Path) -> dict:
    with src.open("rb") as fh:
        data = _ChumpyUnpickler(fh, encoding="latin1").load()

    clean = _unwrap(data)

    # Sanity-check the fields smplx actually reads for MANO.
    required = ["v_template", "f", "shapedirs", "posedirs", "J_regressor", "weights", "hands_mean"]
    missing = [k for k in required if k not in clean]
    if missing:
        raise ValueError(f"converted MANO pickle is missing {missing}")
    for key in required:
        value = clean[key]
        if hasattr(value, "toarray"):  # scipy sparse J_regressor stays sparse; that's fine
            continue
        clean[key] = np.asarray(value)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(clean, fh, protocol=4)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.src.exists():
        sys.exit(f"source MANO pickle not found: {args.src}")

    clean = dechumpify(args.src, args.out)
    v = np.asarray(clean["v_template"])
    print(f"wrote {args.out} (v_template {v.shape}, {len(clean)} keys, chumpy-free)")


if __name__ == "__main__":
    main()
