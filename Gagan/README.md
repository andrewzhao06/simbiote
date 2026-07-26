# Gagan — Scan & Map

The mapper implementation is packaged in `simbiote/mapper/`.

This directory holds Gagan-specific tests. The mapper ingests Stray Scanner
captures and produces an OpenUSD scene plus a scene-graph sidecar for Suraj's
simulator and Andrew's agentic scene query layer.

## Upload phone scans here

Drop real Stray Scanner exports into the repo root folder:

`../UPLOAD_PHONE_SCANS_HERE/`

That is the obvious place for phone captures. See that folder's README for the
required files and validation commands.

Run only the mapper tests:

```bash
python -m pytest Gagan/tests
```

## SAM 3 checkpoint

Production semantic labeling uses the official Hugging Face
`facebook/sam3` checkpoint, `sam3.pt`. Store it at `D:\AI\models\sam3`
locally and `/mnt/simbiote-ssd/AI/models/sam3` on the GB10. The adapter
applies configured text prompts to the first RGB frame and converts detections
into the mapper's `detections.json` contract.
