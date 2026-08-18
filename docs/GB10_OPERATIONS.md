# GB10 Operations Guide

Operational reference for running the **production** scan-to-map pipeline on the
Dell Pro Max with GB10. The top-level [`README.md`](../README.md) covers the
concept and the laptop-tier quickstart; this document covers the air-gapped
production run: model adapters, setup, and the Step 1 → Step 2 handoff contract.

See also:

- [`SIMBIOTE_MASTER_PLAN.md`](SIMBIOTE_MASTER_PLAN.md) — full platform plan
- [`GB10_DOWNLOADS.md`](GB10_DOWNLOADS.md) — model/download checklist
- [`SSD_LAYOUT.md`](SSD_LAYOUT.md) — canonical external SSD layout

---

## Pipeline modes

The mapper (`simbiote/mapper/`) runs in one of three modes, set by
`pipeline.mode` in the TOML config or the `SIMBIOTE_MODE` environment variable:

| Mode | External tools | Geometry produced | Use for |
| --- | --- | --- | --- |
| `proxy` (default) | none | synthetic floor + one placeholder cube | wiring/integration tests only — fails production validation |
| `preview` | none (CPU) | real, sparse LiDAR point cloud (`lidar_preview.usda`) | laptop capture-geometry sanity check |
| `production` | COLMAP, Depth Anything, 3DGRUT, SAM 3 | full reconstructed OpenUSD scene | the real GB10 demo map |

Proxy output is visibly marked (`simbiote:proxy = true`) and fails production
validation unless `--allow-proxy` is explicitly supplied.

### Local CPU preview (no CUDA required)

A Windows/CPU laptop can run `preview` mode. It fuses the Stray Scanner's
confidence-filtered LiDAR depth with its ARKit poses into a sparse OpenUSD
`Points` layer. This is a real capture-geometry preview, not a 3DGUT
reconstruction or a semantic SAM 3 run.

```powershell
$env:SIMBIOTE_MODE="preview"
$env:SIMBIOTE_WORK_ROOT="$PWD\.local\work"

simbiote-map --config config/mapper.example.toml run `
  --capture ".\UPLOAD_PHONE_SCANS_HERE\<scan-name>" `
  --out ".\.local\preview.usda"
```

The output references `lidar_preview.usda` under its timestamped work folder;
open it in any USD viewer. The floor and graspable-object metadata are
development placeholders in this mode.

---

## SSD layout

The prepared SSD has one canonical `AI/` layout. Do not duplicate models or
captures inside the Git repository (see [`SSD_LAYOUT.md`](SSD_LAYOUT.md)):

```text
<SSD>/AI/
├── captures/
├── models/
│   ├── depth-anything/
│   ├── nemotron/
│   ├── sam3/
│   ├── theia/
│   └── wilor/
├── repos/
│   ├── 3dgrut/
│   ├── colmap/
│   ├── Depth-Anything-3/
│   ├── sam3/
│   ├── IsaacSim/
│   └── IsaacLab/
└── assets/
    └── hospital/
        └── hospital.usd
```

`scripts/setup_gb10_mapper.sh` detects `<SSD>/AI` automatically and generates the
mapper config for this layout. Isaac Sim and Isaac Lab source are staged on the
SSD but must be built on the ARM64 GB10.

---

## GB10 setup

On the GB10:

```bash
chmod +x scripts/setup_gb10_mapper.sh
./scripts/setup_gb10_mapper.sh /absolute/path/to/ssd
```

This creates `/var/simbiote/stage/mapper`, generates `config/mapper.gb10.toml`,
and installs the Python package with `uv`.

Edit `config/mapper.gb10.toml` to match the actual downloaded directories, then:

```bash
uv run simbiote-map --config config/mapper.gb10.toml doctor
```

`doctor` returns nonzero if a requirement for the selected mode is missing. The
architecture check is informational so development on x86 laptops still works.

### First capture test (proxy)

Start in proxy mode in `config/mapper.gb10.toml`:

```bash
uv run simbiote-map --config config/mapper.gb10.toml \
  ingest /path/to/ssd/captures/test-scan

uv run simbiote-map --config config/mapper.gb10.toml \
  run --capture /path/to/ssd/captures/test-scan \
  --out /var/simbiote/stage/test-scan.usda \
  --allow-proxy

uv run simbiote-map validate /var/simbiote/stage/test-scan.usda --allow-proxy
```

### Metadata-only Stray Scanner exports

Some exports (pose/IMU-only downloads) contain only `camera_matrix.csv`,
`odometry.csv`, and `imu.csv`. They verify that pose data reaches the mapper but
cannot reconstruct or label a room — no RGB frames or LiDAR depth. Validate one
explicitly:

```bash
uv run simbiote-map --config config/mapper.gb10.toml \
  ingest /path/to/metadata-only-scan --allow-metadata-only
```

In `proxy` mode, `run` accepts this capture and produces a clearly marked test
USD. Production mode always requires `rgb.mp4`, `depth/`, and `confidence/`.

---

## Production adapters

The downloaded model repositories are independently versioned and do not expose
a stable shared Python API. The mapper therefore shells out to **executable
adapter contracts** rather than importing unpinned repository internals. Set
these variables after sourcing a local `.env` or the generated
`config/mapper.gb10.env`.

### COLMAP

```bash
export SIMBIOTE_COLMAP_COMMAND="/opt/simbiote/bin/run_colmap.sh {capture} {output}"
```

Placeholders: `{capture}`, `{output}`. The adapter receives the Stray Scanner
directory and an empty output directory; it must seed/refine from `odometry.csv`
and write its COLMAP artifacts under the output path.

### Depth Anything

```bash
export SIMBIOTE_DEPTH_COMMAND="/opt/simbiote/bin/run_depth.sh {capture} {colmap} {checkpoint} {output}"
```

Placeholders: `{capture}`, `{colmap}`, `{checkpoint}`, `{output}`. It must
confidence-filter the LiDAR depth, complete/register it at RGB resolution, and
write completed-depth artifacts under `{output}` (which must be nonempty).

### 3DGRUT

```bash
export SIMBIOTE_DGRUT_COMMAND="/opt/simbiote/bin/run_3dgrut.sh {capture} {colmap} {depth} {output}"
```

Placeholders: `{capture}`, `{colmap}`, `{depth}`, `{output}`, `{dgrut}`. It must
optimize the Gaussian scene, generate a collision-capable mesh, convert that to
an OpenUSD layer, and emit at least one `.usd`, `.usda`, or `.usdc` directly in
`{output}`. The pipeline refuses to continue if the USD geometry layer does not
exist — a raw `.ply` or `.obj` alone is **not** a valid Step 2 handoff.

### SAM 3

```bash
export SIMBIOTE_SAM3_COMMAND="/opt/simbiote/bin/run_sam3.sh {capture} {geometry} {checkpoint} {output}"
```

Placeholders: `{capture}`, `{geometry}`, `{checkpoint}`, `{output}`. It must emit
`{output}/detections.json`:

```json
{
  "nodes": [
    {
      "node_id": "floor_0",
      "label": "navigable floor",
      "kind": "floor",
      "confidence": 0.98,
      "bounds": { "center": [0.0, 0.0, 0.0], "size": [6.0, 0.05, 8.0] },
      "source_frame_ids": [0, 1, 2]
    },
    {
      "node_id": "tray_0",
      "label": "tray",
      "kind": "object",
      "confidence": 0.91,
      "bounds": { "center": [1.2, 0.8, -0.4], "size": [0.4, 0.08, 0.3] }
    }
  ]
}
```

Set `mode = "production"` only after all adapters pass `doctor`.

### GB10 adapter runtime

The production adapters live in `scripts/gb10/` and use the native tool
interfaces rather than mock output:

1. `run_colmap.sh` extracts `rgb.mp4` and produces `images/` plus `sparse/0/`.
2. `run_depth.sh` runs Depth Anything 3 and writes `exports/mini_npz/results.npz`.
3. `run_3dgrut.sh` trains the `colmap_3dgut` configuration and requires its
   exported USD/USDZ layer.
4. `run_sam3.sh` uses a local `sam3.pt` (official `facebook/sam3`) to turn fixed
   text prompts into the `detections.json` contract.

Run the setup script, then source its generated environment:

```bash
./scripts/setup_gb10_mapper.sh /mnt/simbiote-ssd
source config/mapper.gb10.env
```

The repositories have separate CUDA Python environments. Before a production
run, point these at the executables created when installing each repository:

```bash
export DGRUT_PYTHON="$DGRUT_ROOT/.venv/bin/python"
export DA3_BIN="/path/to/depth-anything-3/.venv/bin/da3"
export SAM3_PYTHON="/path/to/sam3/.venv/bin/python"
```

The adapters require `ffmpeg`, a built `colmap` executable in `PATH`, and
CUDA-enabled PyTorch in the DA3, SAM 3, and 3DGRUT environments. Run each
repository's documented install on the GB10 ARM64 system — copying a Windows
environment will not work.

Run the strict hardware/asset check before a production map:

```bash
scripts/gb10/preflight_mapper.sh /mnt/simbiote-ssd/AI "$PWD"
```

It blocks on missing ARM64/NVIDIA runtime, checkpoints, repositories, COLMAP,
3DGRUT's Python environment, or non-executable adapters. The hospital asset is a
warning (a live scan can still proceed) but remains required for the reliable
fallback demo.

---

## Production run and handoff

```bash
uv run simbiote-map --config config/mapper.gb10.toml \
  run --capture "$SIMBIOTE_SSD_ROOT/captures/demo" \
  --out /var/simbiote/stage/demo.usda

uv run simbiote-map validate /var/simbiote/stage/demo.usda
```

Successful output includes:

- `demo.usda` — the Step 2 scene.
- `demo.scene_graph.json` — the planner/query contract.
- `/var/simbiote/stage/mapper/<timestamp>/` — stage manifests and evidence.

The validator (`validate_map`) is the coded Step 1 → Step 2 contract. A valid
production map requires meter scale (`metersPerUnit = 1`), Y-up coordinates, a
`PhysicsCollisionAPI`, a navigable floor, and at least one graspable object with
`mass_kg` and `grasp_type`, plus a referenced `ReconstructedGeometry` layer.
Production validation rejects proxy output.
