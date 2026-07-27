# Scan and map

The mapper implementation is packaged in `src/`.

Mapper tests live in `tests/mapper/`. The mapper ingests Stray Scanner captures
and produces an OpenUSD scene plus a scene-graph sidecar for the simulator and
agentic scene-query layer.

## Upload phone scans here

Drop real Stray Scanner exports into the repo root folder:

`../../UPLOAD_PHONE_SCANS_HERE/`

That is the obvious place for phone captures. See that folder's README for the
required files and validation commands.

Run only the mapper tests:

```bash
source scripts/gb10/env.gb10.sh
$FF_PY -m pytest tests/mapper
```

## Running it on the GB10 (verified end to end)

There is no venv for this. `scripts/gb10/env.gb10.sh` points at Isaac Sim's bundled
interpreter, which is the only Python on the box with numpy, PIL *and* `pxr`
(there is no `usd-core` aarch64 wheel — `pxr` comes from the `omni.usd.libs`
kit extension, which Isaac Sim's own `setup_python_env.sh` leaves off
`PYTHONPATH`).

```bash
source scripts/gb10/env.gb10.sh
export FACTORYFLOW_MODE=preview

# 1. validate the capture parses and every frame has a depth/confidence pair
$FF_PY -m src.cli --config config/mapper.gb10.toml \
    ingest UPLOAD_PHONE_SCANS_HERE/<scan>

# 2. scan -> USD
$FF_PY -m src.cli --config config/mapper.gb10.toml run \
    --capture UPLOAD_PHONE_SCANS_HERE/<scan> \
    --out artifacts/mapper/out/<scan>.usda

# 3. SCAN_MAP.md 4.5 acceptance test: does it open in Isaac Sim?
/home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
    scripts/gb10/check_mapper_usd.py artifacts/mapper/out/<scan>.usda      # add --gui to watch it
```

Step 2 writes two files that travel together: `<scan>.usda` (the scene) and
`<scan>.geometry.usda` (the LiDAR cloud it references relatively). Keep them in
the same directory — copying only the first hands Step 2 a dangling reference.

### Copying a scan off the iPhone

Stray Scanner exposes its captures over AFC, so a plugged-in phone mounts
without `ifuse`:

```bash
cp -r "/run/user/1000/gvfs/afc:host=<UDID>,port=3/dev.keke.StrayScanner/<capture-id>" \
      UPLOAD_PHONE_SCANS_HERE/<scan>
```

### Two things Step 2 needs to know

**Gravity.** The contract says `upAxis = "Y"`, but Isaac Sim's `PhysicsContext`
resets the stage up axis to Z when it initialises and re-derives gravity from
it — overwriting whatever is authored on `/physicsScene`. A Y-up scene silently
gets gravity along **-Z**, sideways through the walls, and the robot topples on
spawn. After `initialize_physics()`:

```python
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
simulation_context.get_physics_context().set_gravity(-9.81)
```

`check_mapper_usd.py` asserts this and then drops a rigid body to prove the
floor collider actually catches it.

**Simulated poses aren't in USD.** PhysX writes transforms to Fabric, so
reading `xformOp:translate` after stepping returns the authored spawn pose, not
the simulated one. Read through `SingleRigidPrim.get_world_pose()`.

## Preview mode vs production

`preview` reconstructs real geometry from the LiDAR depth maps on CPU: it
unprojects confidence-filtered depth into the ARKit world frame, voxel-downsamples
to 2 cm, and measures the floor plane out of the resulting cloud. That is enough
to satisfy the Step 2 contract with genuine scanned geometry.

It is *not* the production path. `production` additionally needs the four
adapter commands (COLMAP, Depth Anything, 3DGRUT, SAM 3) exported — the
checkpoints and repos are already on the box under `/home/dell/AI`, but
`FACTORYFLOW_*_COMMAND` are unset, so `doctor` reports those four as missing.
Until they are wired, the graspable object is a labelled proxy: preview mode has
no semantic segmentation, so `navigable_floor` is real and measured while
`proxy_graspable` is a placeholder cube resting on it.

## SAM 3 checkpoint

Production semantic labeling uses the official Hugging Face
`facebook/sam3` checkpoint, `sam3.pt`. On the GB10 it lives at
`/home/dell/AI/models/sam3`, with the code checkout at
`/home/dell/AI/repos/sam3`.

### Real graspable objects — `scripts/gb10/adapters/sam3_detect.py`

The adapter runs SAM 3 over frames sampled across the whole sweep, projects
each **mask** into 3D with that frame's own pose and LiDAR depth, and merges
detections of the same object seen from several angles into one node.

```bash
source scripts/gb10/env.gb10.sh
export FACTORYFLOW_MODE=preview
export FACTORYFLOW_SAM3_COMMAND="$FF_PY $PWD/scripts/gb10/adapters/sam3_detect.py \
    --capture {capture} --geometry {geometry} \
    --checkpoint {checkpoint} --output {output} \
    --config $PWD/config/mapper.gb10.toml --frames 16"

$FF_PY -m src.cli --config config/mapper.gb10.toml run \
    --capture UPLOAD_PHONE_SCANS_HERE/<scan> --out artifacts/mapper/out/<scan>.usda
```

Semantic labelling only needs posed RGB + depth, so it runs in **preview** mode
too — you do not need COLMAP and 3DGRUT wired to get real objects. Setting
`FACTORYFLOW_SAM3_COMMAND` is what switches it on; leave it unset for the
placeholder cube.

Tuning knobs: `--object-prompts "chair;bottle;tray"` (semicolon-separated, these
are open-vocabulary phrases, not a fixed class list), `--min-score`,
`--merge-radius` (metres; how far apart two detections must be to count as
separate objects), `--max-object-size` (rejects over-segmented blobs).

The floor is handled specially: SAM 3 supplies the *height* (semantics it is
reliable at), and the extent comes from every reconstructed point lying in that
plane — SAM 3 alone only sees the floor inside the frames sampled and
under-covers the room by roughly half.

`scripts/gb10/sam3_labels.py` is the earlier contract stub. It reads only the
first frame and places every detection at one point (first pose + scene-median
depth), so object positions are not real. Prefer the adapter above.

### Capture gotcha this adapter guards against

`rgb.mp4` can hold fewer frames than `odometry.csv` has rows — the gb10
walkthrough is short exactly one, because recording stopped after the last pose
was written. Pairing frame *i* with row *i* blindly would attach the wrong pose
to everything after a drop and silently misplace every object. The adapter
compares video presentation timestamps against odometry timestamps and only
uses the verified prefix, refusing to run if they desynchronise mid-stream.
