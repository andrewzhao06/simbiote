# Drop Stray Scanner exports here

This is the upload folder for real phone scans.

## What to do

1. Export a capture from the **Stray Scanner** app on the iPhone.
2. Copy the whole export folder into this directory.

Example:

```text
UPLOAD_PHONE_SCANS_HERE/
└── hospital-walkthrough/
    ├── camera_matrix.csv
    ├── odometry.csv
    ├── imu.csv
    ├── rgb.mp4
    ├── depth/
    │   └── 000000.png ...
    └── confidence/
        └── 000000.png ...
```

## Required files

| File / folder | Required |
| --- | --- |
| `camera_matrix.csv` | yes |
| `odometry.csv` | yes |
| `imu.csv` | yes |
| `rgb.mp4` | yes |
| `depth/` | yes |
| `confidence/` | yes |

## Validate after upload

```powershell
python -m simbiote.mapper.cli ingest ".\UPLOAD_PHONE_SCANS_HERE\hospital-walkthrough"
```

## Preview locally

```powershell
$env:FACTORYFLOW_MODE="preview"
$env:FACTORYFLOW_WORK_ROOT="$PWD\.local\work"

python -m simbiote.mapper.cli --config config\mapper.example.toml run `
  --capture ".\UPLOAD_PHONE_SCANS_HERE\hospital-walkthrough" `
  --out ".\.local\preview.usda"
```

## Optional: copy to the SSD

If you also want the scan on the external drive used by GB10:

```powershell
python scripts\mapper\import_stray_capture.py ".\UPLOAD_PHONE_SCANS_HERE\hospital-walkthrough" --name hospital-walkthrough
```

That copies it to `D:\AI\captures\hospital-walkthrough`.

Scan media is gitignored. Only this README stays in the repository.
