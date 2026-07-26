# GB10 SSD layout

Mount the drive at a path such as `/mnt/simbiote-ssd`; all portable assets
live below its `AI/` directory.

```text
AI/
├── models/
│   ├── depth-anything/  # Gagan: depth completion
│   ├── sam3/            # Gagan: official facebook/sam3 checkpoint
│   ├── nemotron/        # Andrew: agentic planning model
│   ├── theia/           # Suraj: optional vision backbone
│   └── wilor/           # Sky: hand-pose model
├── repos/
│   ├── 3dgrut/          # Gagan: reconstruction source
│   ├── colmap/          # Gagan: build CUDA/ARM binary on GB10
│   ├── Depth-Anything-3/
│   ├── sam3/
│   ├── curobo/
│   ├── IsaacSim/        # build on GB10 ARM64
│   └── IsaacLab/        # install against the GB10 Isaac Sim build
├── assets/
│   └── hospital/        # populated from Isaac Sim/NGC on GB10
└── captures/            # Stray Scanner exports; never commit to Git
```

Run `scripts/setup_gb10_mapper.sh <SSD mount>` on the GB10 to create the
mapper configuration. Then build Isaac Sim and install Isaac Lab on that
machine; their Linux ARM64 build artifacts are not portable from Windows.
