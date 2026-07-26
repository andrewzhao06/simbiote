# Shared schema proposals — from Step 4 (agentic control)

Three interfaces cross role boundaries, and the master doc says all three are
agreements, not unilateral decisions (§6.2 / §6b.2, §6b.7). Here is a concrete
starting point for each so the conversation happens over something real.

**None of this is settled. Push back on any of it.** Step 4 is already built
against these, but every one of them is isolated behind a single adapter
function, so a change costs one edit, not a rewrite:

| Schema | Adapter seam |
| :---- | :---- |
| `RobotAction` | `RobotAction.to_dict` / `from_dict` in `factoryflow/robot_iface/actions.py` |
| Trajectory | `TrajectoryStep.to_dict` / `from_dict` in `factoryflow/demo_logger.py` |
| Scene graph | `_parse()` in `factoryflow/agentic/scene_query.py` |

---

## 1. `RobotAction` — Teammate 3 + Teammate 4 produce, Teammate 2 consumes

`factoryflow/robot_iface/actions.py`

```python
RobotAction(
    base_velocity: tuple[float, float, float],   # (vx, vy, omega), m/s and rad/s
    arm_target_pose: Pose | None,                # None on a nav-only step
    gripper_state: GripperState,                 # "open" | "closed"
)

Pose(x, y, z, qx, qy, qz, qw)   # metres + XYZW quaternion
```

Serialized form:

```json
{
  "base_velocity": [0.6, 0.0, 0.0],
  "arm_target_pose": null,
  "gripper_state": "open"
}
```

**Choices worth arguing about:**

- `arm_target_pose` is **optional**. A pure navigation step has no arm target,
  and filling the field with a dummy pose would put fabricated arm data into the
  trajectories Step 2 fine-tunes on. Teammate 3's teleop always has a hand pose,
  so this costs teleop nothing.
- Pose is `x, y, z, qx, qy, qz, qw` to match Stray Scanner's `odometry.csv`
  (§4.2) — same shape from phone capture through to logged trajectory.
- Stdlib-only frozen dataclasses, no pydantic/numpy, so adopting this drags in
  no dependency and needs no aarch64 wheel.

**Open questions:**

1. **Gripper: binary or width?** Currently `open`/`closed`. The Franka hand can
   be commanded to a width — if hand-tracking wants to pass through a continuous
   pinch distance, that is a schema change, so decide now rather than later.
2. **Arm target: pose or joint angles?** Cartesian EE pose assumes an IK layer
   (cuMotion/cuRobo) between action and robot. If Teammate 3's retargeting
   emits joint angles directly, we need either a second field or a convention.
3. **Frame convention.** Is `arm_target_pose` in world frame or base frame?
   Proposal: **world frame**, since the scene graph is. Needs Teammate 2 to
   confirm it matches what the policies expect.
4. **Units.** Metres, m/s, rad/s throughout. Confirm against Isaac Lab's task
   definitions.

---

## 2. Trajectory log — Teammate 3 + Teammate 4 produce, Teammate 2 ingests

`factoryflow/demo_logger.py`. Per §6b.7 this one is a **group** decision — it is
the single interface all four roles touch.

One JSON object per line, at `$FACTORYFLOW_STAGE/sessions/<session_id>.jsonl`:

```json
{"t": 1785000123.412, "source": "agentic", "skill": "navigate_to", "action": {...}, "ok": true}
{"t": 1785000123.478, "source": "agentic", "skill": "pick_up",     "action": {...}, "ok": true}
```

| Field | Meaning |
| :---- | :---- |
| `t` | Unix timestamp, seconds, float |
| `source` | `"teleop"` or `"agentic"` — §6.5's required discriminator |
| `skill` | Step 4's addition: the atomic skill that produced this action. Teleop writes `null`. |
| `action` | A serialized `RobotAction` |
| `ok` | Whether this action was *issued* without error |

**Choices worth arguing about:**

- **JSONL, not one JSON array.** A session killed mid-run still leaves a
  readable partial trajectory. With ten rehearsal loops planned under time
  pressure, that is worth more than a tidier file.
- **`skill` is additive.** Teleop leaving it `null` is valid, so both producers
  write the same format and Step 2 needs one reader, not two. It lets Step 2
  slice a fine-tune by skill.
- **`ok` is per-action, not per-skill.** Whether a *skill* ultimately succeeded
  is not knowable at the moment an action is emitted, so it lives in a sidecar
  `<session_id>.report.json` instead — plan, per-step status, timings, failure
  reason.
- **Failed runs still log.** Partial demonstrations are useful training data,
  and an empty file after a failure hides what went wrong.

**Open questions:**

1. **What exactly does `ingest_demo()` want?** If Step 2 wants observations
   alongside actions, that is a bigger change and needs to happen tonight, not
   tomorrow afternoon.
2. **Absolute timestamps or session-relative?** Currently absolute Unix time.
   Session-relative is friendlier for replay.
3. **Does Step 2 want the sidecar report at all,** or should success/failure be
   folded into each record?

---

## 3. Scene graph — Teammate 1 produces, Teammate 4 consumes

`factoryflow/agentic/scene_query.py`, with a working fixture at
`factoryflow/fixtures/hospital_scene_graph.json`.

This is the one I am **least** confident in, because it is reverse-engineered
from §4.3's description of `build_graph.py` rather than from real output.
Teammate 1 should treat it as a strawman.

```json
{
  "schema_version": "0.1",
  "scene_id": "hospital_01",
  "units": "meters",
  "up_axis": "Z",
  "nodes": [
    {
      "id": "supply_room", "kind": "location", "label": "supply room",
      "aliases": ["supply", "storage room"],
      "pose": {"x": -4.0, "y": 6.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
      "bbox": {"min": [-7.0, 3.5, 0.0], "max": [-1.5, 8.5, 2.8]},
      "navigable": true
    },
    {
      "id": "tray_01", "kind": "object", "label": "tray",
      "aliases": ["food tray"],
      "pose": {"x": -4.2, "y": 6.4, "z": 0.82, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
      "bbox": {"min": [-4.45, 6.2, 0.78], "max": [-3.95, 6.6, 0.86]},
      "is_graspable": true, "grasp_type": "top", "mass_kg": 0.8
    }
  ],
  "edges": [
    {"from": "tray_01", "to": "supply_room", "relation": "in"}
  ]
}
```

**What Step 4 actually needs from it:**

- **Stable string ids.** They go straight into the LLM prompt and every emitted
  tool call is validated against them, which is what stops a hallucinated
  `room_7` from becoming a navigation goal.
- **`is_graspable` / `grasp_type` / `mass_kg` per object** — §5.5 says Step 2's
  grasp task reads these, and Step 4 rejects a `pick_up` on anything tagged
  `is_graspable: false`.
- **A containment edge** (`{"from": <object>, "to": <location>, "relation": "in"}`)
  so "the tray in the supply room" can be turned into
  `navigate_to(supply_room)` → `pick_up(tray_01)`. Without it Step 4 cannot know
  where to drive before grasping.
- **`navigable` on locations**, so a non-navigable region is rejected at
  validation rather than handed to the nav policy.

**Nice to have, not required:**

- `aliases` — improves phrase resolution ("trolley" → `cart_01`) but Step 4
  works without it.
- `handle_pose` on objects with a dedicated grasp point (the wheelchair). If
  it is absent, Step 4 falls back to the object's centre pose.

**Open questions:**

1. **Does `build_graph.py` emit stable ids across runs,** or are they
   regenerated per reconstruction? Tool calls reference them by name.
2. **Are locations first-class nodes,** or are rooms inferred from geometry? If
   the latter, Step 4 needs some other way to name "Room 2".
3. **Same schema from the library hospital scene and from a live phone scan?**
   Tier 1 uses the library scene; Tier 3 uses a live scan. If they differ,
   `_parse()` needs two branches.

---

## What I need, and by when

Tonight, ideally:

- **Teammate 3** — sign off on `RobotAction`, or tell me what teleop needs
  differently (questions 1–2 above).
- **Teammate 1** — a real sample of `build_graph.py`'s output, even a partial or
  hand-edited one. I will rewrite `_parse()` against it; nothing else changes.
- **Teammate 2** — the exact shape `ingest_demo()` expects, plus the inference
  call signatures for `navigate_to` / `pick_up` (and the wheelchair four if the
  stretch task is on). `CheckpointBackend` in `factoryflow/agentic/robot_tools.py`
  is a stub waiting on precisely that.

Everything on the Step 4 side runs today against the fixture and a stub backend,
so none of these are blocking me — but each one that lands tonight is one less
thing to discover at 2 PM tomorrow.
