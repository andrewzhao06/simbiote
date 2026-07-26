# FactoryFlow Mapper (Teammate 1)

This repository contains only the Teammate 1 path:

`Stray Scanner capture -> pose refinement -> reconstruction -> semantic labels -> scene graph -> physics metadata -> OpenUSD`

It deliberately does not contain robot training, teleoperation, or prompting code.

## What works now

- Strictly validates the confirmed Stray Scanner bundle format.
- Matches RGB/depth/confidence by the `frame` column, including non-contiguous frames.
- Preserves ARKit poses, per-frame intrinsics, IMU samples, and warnings.
- Creates per-run artifacts and a machine-readable scene graph.
- Exports an ASCII OpenUSD stage with collision, scale, navigation, grasp, mass, and friction metadata.
- Validates the exact Step 2 handoff contract.
- Provides a GB10 dependency doctor.
- Supports a safe `proxy` path for end-to-end integration tests.
- Exposes production adapter contracts for the SSD's COLMAP, 3DGRUT, and SAM 3 installs.

Proxy output is visibly marked and fails production validation unless `--allow-proxy` is
explicitly supplied. It is for wiring tests, not the demo map.

## Local CPU preview

Your Windows laptop can run `preview` mode even without NVIDIA CUDA. It fuses
the Stray Scanner's confidence-filtered LiDAR depth with its ARKit poses into a
sparse OpenUSD `Points` layer, then composes it into the mapper output. This is
a real capture-geometry preview; it is not a 3DGUT reconstruction or semantic
SAM 3 run.

```powershell
$env:PYTHONPATH="$PWD\src"
$env:FACTORYFLOW_MODE="preview"
$env:FACTORYFLOW_WORK_ROOT="$PWD\.local\work"

python -m factoryflow_mapper.cli --config config/mapper.example.toml run `
  --capture ".\ab567bedef" `
  --out ".\.local\ab567bedef-preview.usda"
```

The output references `lidar_preview.usda` under its timestamped work folder.
Open it in a USD viewer or Isaac Sim tomorrow to inspect the point cloud. The
floor and graspable-object metadata are development placeholders in this mode.

## SSD layout

Mount the prepared SSD, then arrange or symlink the downloads like this:

```text
<ssd>/
├── models/
│   ├── depth-anything/
│   └── sam3/
├── tools/
│   └── 3dgrut/
├── assets/
│   └── hospital/
│       └── hospital.usd
└── captures/
    └── <scan-name>/
        ├── camera_matrix.csv
        ├── odometry.csv
        ├── imu.csv
        ├── rgb.mp4
        ├── depth/
        └── confidence/
```

The mapper does not copy large models or captures into Git.

If the SSD already uses the current `AI/` layout, preserve it rather than
duplicating large downloads:

```text
<SSD>/AI/
├── captures/
├── models/
│   ├── depth-anything/
│   ├── sam3/
│   └── nemotron/
├── repos/
│   ├── 3dgrut/
│   ├── colmap/
│   ├── Depth-Anything-3/
│   └── sam3/
└── assets/
    └── hospital/
        └── hospital.usd
```

`scripts/setup_gb10_mapper.sh` detects `<SSD>/AI` automatically and generates
the GB10 config for this layout.

## GB10 setup

On the GB10:

```bash
chmod +x scripts/setup_gb10_mapper.sh
./scripts/setup_gb10_mapper.sh /absolute/path/to/ssd
```

This creates `/var/factoryflow/stage/mapper`, generates
`config/mapper.gb10.toml`, and installs the Python package with `uv`.

Edit `config/mapper.gb10.toml` to match the actual downloaded directories. Then run:

```bash
uv run factoryflow-map --config config/mapper.gb10.toml doctor
```

`doctor` returns nonzero if a requirement for the selected mode is missing. The
architecture check is informational so development on x86 laptops still works.

## First capture test

Start in proxy mode in `config/mapper.gb10.toml`:

```bash
uv run factoryflow-map --config config/mapper.gb10.toml \
  ingest /path/to/ssd/captures/test-scan

uv run factoryflow-map --config config/mapper.gb10.toml \
  run --capture /path/to/ssd/captures/test-scan \
  --out /var/factoryflow/stage/test-scan.usda \
  --allow-proxy
```

Then validate:

```bash
uv run factoryflow-map validate /var/factoryflow/stage/test-scan.usda --allow-proxy
```

### Metadata-only Stray Scanner export

Some exports, including pose/IMU-only downloads, contain only
`camera_matrix.csv`, `odometry.csv`, and `imu.csv`. They can be used to verify
that pose data reaches the mapper, but they cannot reconstruct or label a room:
they have no RGB frames or LiDAR depth.

Validate one explicitly:

```bash
uv run factoryflow-map --config config/mapper.gb10.toml \
  ingest /path/to/metadata-only-scan --allow-metadata-only
```

In `proxy` mode, `run` accepts this capture and produces a clearly marked test
USD. Production mode always requires `rgb.mp4`, `depth/`, and `confidence/`.

## Production adapters

The downloaded model repositories are independently versioned and do not expose a
stable shared Python API. The mapper therefore uses executable adapter contracts rather
than importing unpinned repository internals.

Set these variables after sourcing a local `.env` or shell script:

### COLMAP

```bash
export FACTORYFLOW_COLMAP_COMMAND="/opt/factoryflow/bin/run_colmap.sh {capture} {output}"
```

Available placeholders: `{capture}`, `{output}`.

The adapter receives the Stray Scanner directory and an empty output directory. It must
seed/refine from `odometry.csv` and write its COLMAP artifacts under the output path.

### Depth Anything

```bash
export FACTORYFLOW_DEPTH_COMMAND="/opt/factoryflow/bin/run_depth.sh {capture} {colmap} {checkpoint} {output}"
```

Available placeholders: `{capture}`, `{colmap}`, `{checkpoint}`, `{output}`.

It must confidence-filter the LiDAR depth, complete/register it at RGB resolution, and
write completed-depth artifacts under `{output}`. The pipeline requires that directory
to be nonempty.

### 3DGRUT

```bash
export FACTORYFLOW_DGRUT_COMMAND="/opt/factoryflow/bin/run_3dgrut.sh {capture} {colmap} {depth} {output}"
```

Available placeholders: `{capture}`, `{colmap}`, `{depth}`, `{output}`, `{dgrut}`.

It must optimize the Gaussian scene, generate a collision-capable mesh, convert that
geometry to an OpenUSD layer, and emit at least one `.usd`, `.usda`, or `.usdc` directly
in `{output}`. The final mapper stage composes that layer into the handoff scene. The
pipeline refuses to continue if the USD geometry layer does not exist; a raw `.ply` or
`.obj` alone is not a valid Step 2 handoff.

### SAM 3

```bash
export FACTORYFLOW_SAM3_COMMAND="/opt/factoryflow/bin/run_sam3.sh {capture} {geometry} {checkpoint} {output}"
```

Available placeholders: `{capture}`, `{geometry}`, `{checkpoint}`, `{output}`.

It must emit `{output}/detections.json`:

```json
{
  "nodes": [
    {
      "node_id": "floor_0",
      "label": "navigable floor",
      "kind": "floor",
      "confidence": 0.98,
      "bounds": {
        "center": [0.0, 0.0, 0.0],
        "size": [6.0, 0.05, 8.0]
      },
      "source_frame_ids": [0, 1, 2]
    },
    {
      "node_id": "tray_0",
      "label": "tray",
      "kind": "object",
      "confidence": 0.91,
      "bounds": {
        "center": [1.2, 0.8, -0.4],
        "size": [0.4, 0.08, 0.3]
      }
    }
  ]
}
```

Set `mode = "production"` only after all adapters pass `doctor`.

### GB10 adapter runtime

The production adapters are in `scripts/gb10/`. They use the native tool
interfaces rather than mock output:

1. `run_colmap.sh` extracts `rgb.mp4` and produces `images/` plus `sparse/0/`.
2. `run_depth.sh` runs DA3 and writes `exports/mini_npz/results.npz`.
3. `run_3dgrut.sh` trains the existing `colmap_3dgut` configuration and requires
   its exported USD/USDZ layer.
4. `run_sam3.sh` uses local `sam3.pt` to turn fixed text prompts into the
   `detections.json` contract.

Run the setup script, then source its generated environment:

```bash
./scripts/setup_gb10_mapper.sh /mnt/factoryflow-ssd
source config/mapper.gb10.env
```

The repositories have separate CUDA Python environments. Before a production
run, set these three variables in `config/mapper.gb10.env` to the executables
created when installing the corresponding repositories:

```bash
export DGRUT_PYTHON="$DGRUT_ROOT/.venv/bin/python"
export DA3_BIN="/path/to/depth-anything-3/.venv/bin/da3"
export SAM3_PYTHON="/path/to/sam3/.venv/bin/python"
```

The adapters require `ffmpeg`, a built `colmap` executable in `PATH`, and
CUDA-enabled PyTorch in the DA3, SAM 3, and 3DGRUT environments. Run each
repository's documented install process on the GB10 ARM64 system; copying this
Windows environment will not work.

## Production run and handoff

```bash
uv run factoryflow-map --config config/mapper.gb10.toml \
  run --capture "$FACTORYFLOW_SSD_ROOT/captures/demo" \
  --out /var/factoryflow/stage/demo.usda

uv run factoryflow-map validate /var/factoryflow/stage/demo.usda
```

Successful output includes:

- `demo.usda` — Step 2 scene.
- `demo.scene_graph.json` — planner/query contract.
- `/var/factoryflow/stage/mapper/<timestamp>/` — stage manifests and evidence.

The validator requires meter scale, Y-up coordinates, collision APIs, a navigable floor,
and at least one graspable object with mass and grasp type. Production validation rejects
proxy output.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```
