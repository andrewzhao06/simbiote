# Gagan — Scan & Map

The mapper implementation is packaged in `src/factoryflow_mapper/`.

This directory holds Gagan-specific tests. The mapper ingests Stray Scanner
captures and produces an OpenUSD scene plus a scene-graph sidecar for Suraj's
simulator and Andrew's agentic scene query layer.

Run only the mapper tests:

```bash
python -m pytest Gagan/tests
```
