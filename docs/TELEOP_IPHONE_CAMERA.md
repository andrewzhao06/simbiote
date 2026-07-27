# Teleop camera: getting the iPhone into the GB10

Step 3's teleop chain starts with a camera frame. The spec's plan was Iriun
Webcam — the iPhone publishes into a virtual webcam on the host, and OpenCV
just opens that device. That still holds on a laptop. **It does not work on
the GB10**, and this doc explains why and what to do instead.

## Why Iriun can't be the GB10 path

Iriun ships one Linux package, `iriunwebcam-2.9.1.deb`. Its control file says
`Architecture: all`, which is misleading — the payload is a single compiled
binary:

```
$ dpkg-deb -x iriunwebcam-2.9.1.deb x && file x/usr/local/bin/iriunwebcam
ELF 64-bit LSB pie executable, x86-64, ..., stripped
```

x86-64 only. The GB10 is `aarch64`, so the client cannot execute here at all,
and Iriun publishes no ARM build. This is the same class of problem as
MediaPipe having no `linux-aarch64` wheel — it isn't a configuration issue,
the artifact doesn't exist.

Two further constraints on this box:

- `Depends: v4l2loopback-dc`. Any virtual-webcam approach needs the
  `v4l2loopback` kernel module, which means DKMS and **root**. `sudo` on this
  machine requires a password.
- DroidCam, the usual Iriun substitute, has an iOS app and works the same way
  — but its Linux client also ships "standard 64-bit binaries" only, with
  ARM left as compile-from-source.

## Best path: stream over the USB cable

DroidCam serves its MJPEG stream on a port *on the phone*, and `usbmuxd`
(already running here — it's what lets Nautilus browse the iPhone over AFC)
can forward a local TCP port to a tethered device. So the stream can come in
over the cable with no Wi-Fi involved:

```bash
# terminal 1 -- hold the tunnel open
./.venv/bin/python scripts/gb10/usbmux_tunnel.py --device-port 4747

# then the camera URL is simply localhost
./.venv/bin/python scripts/teleop/run_demo.py --sink udp \
    --camera-url http://127.0.0.1:4747/video/640x480
```

`scripts/gb10/usbmux_tunnel.py --list` shows tethered devices. The phone must
be plugged in, unlocked, and trusted; DroidCam must be running on it, but it
does **not** need to be on any Wi-Fi network.

Measured at 640x480: **30.0 FPS over USB**, versus 18.2 on clean Wi-Fi and 4.6
during a congested spell. USB also removes the failure mode that actually bit
during bring-up — the phone silently dropping off the network mid-session.

The usual tool for this is `iproxy` from `libimobiledevice-utils`, which isn't
installed and needs a password to apt-install. `usbmux_tunnel.py` speaks the
usbmuxd protocol directly instead (plist over a world-writable Unix socket,
stdlib only), so it needs no root.

## Fallback: stream over the network

Phone webcam apps also serve the camera as **MJPEG over HTTP**, and OpenCV
opens an MJPEG URL exactly like a device. That path needs no kernel module,
no host client, and **no root** — nothing gets installed on the GB10 at all.
This is the supported teleop camera path here.

### 1. Install a streaming app on the iPhone

[DroidCam](https://apps.apple.com/us/app/droidcam-webcam-obs-camera/id1510258102)
is the one to match the spec's intent. Any app that serves MJPEG works
(IP Camera Lite, Larix Broadcaster for RTSP); only the URL shape changes.

### 2. Put both devices on the same network

The iPhone and the GB10 must be on the same LAN/Wi-Fi. Open the app — it
shows its IP and port. DroidCam defaults to port `4747`.

### 3. Find the stream URL

DroidCam serves:

```
http://<iphone-ip>:4747/video                 # default resolution
http://<iphone-ip>:4747/video/1280x720        # explicit size
```

Check it from the GB10 before wiring teleop to it. Use a ranged **GET**, not
`curl -I` — the app doesn't answer HEAD requests and returns a misleading 404:

```bash
curl -s -o /dev/null -w '%{http_code}\n' --max-time 4 -r 0-1000 \
    http://<iphone-ip>:4747/video
```

`200` means the stream is live.

**Use `640x480`.** Measured on this box over Wi-Fi: 640×480 delivers ~18 FPS
while 1280×720 manages ~7, and the hand model crops to 256×192 internally, so
the larger frame costs throughput without buying any accuracy. Higher
resolutions are the single biggest teleop performance mistake available here.

### 4. Run teleop against it

```bash
cd ~/simbiote
./.venv/bin/python scripts/teleop/run_demo.py \
    --camera-url http://<iphone-ip>:4747/video/640x480 \
    --backend wilor --sink pybullet
```

Or set it once for the shell:

```bash
export SIMBIOTE_CAMERA_URL=http://<iphone-ip>:4747/video/640x480
./.venv/bin/python scripts/teleop/run_demo.py --sink pybullet
```

Hold or mount the phone in **landscape**. DroidCam follows the device's
orientation, so a portrait phone streams the scene rotated 90°, and
`ik_bridge` maps image-y to the arm's height axis — an uncorrected rotation
silently swaps your up/down and left/right. If the mount forces portrait, pass
`--rotate 90` (or `270`) to square it back up.

Wi-Fi adds latency a USB tether wouldn't, and it is the dominant variable
here: on a congested link this setup dropped to ~6 FPS with 33% packet loss,
versus ~14 FPS end-to-end on a clean one. Keep the phone on 5 GHz and both
devices on the same access point rather than routing across subnets.

## Optional: the /dev/video path, if you can get root

If someone with the password can install packages, the virtual-webcam route
becomes available and teleop takes `--camera-index` instead. You still need a
client that runs on aarch64, so DroidCam has to be built from source:

```bash
sudo apt install v4l2loopback-dkms v4l-utils linux-headers-$(uname -r) build-essential
git clone https://github.com/dev47apps/droidcam && cd droidcam
make && sudo ./install-client
sudo modprobe v4l2loopback exclusive_caps=1 card_label="Phone Cam"
```

Then find the index and use it:

```bash
./.venv/bin/python -c \
  "from simbiote.teleop.camera_source import list_camera_indices, list_video_devices; \
   print(list_video_devices(), list_camera_indices())"

./.venv/bin/python scripts/teleop/run_demo.py --camera-index 0 --sink pybullet
```

This buys lower latency and lets any other app use the phone as a webcam. It
is not required for teleop — the URL path above is equivalent for our purposes.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `no /dev/video* devices exist on this machine at all` | Nothing has published a virtual webcam. Use `--camera-url`. |
| `could not open camera stream` | Wrong IP, phone asleep, app backgrounded, or devices on different subnets. Re-check with the ranged GET above. |
| `404` when probing the URL | You used `curl -I`. The app rejects HEAD; use a ranged GET. |
| Up/down hand motion moves the arm sideways | Feed is rotated — phone is in portrait. Use `--rotate 90` or `270`. |
| Window opens, feed is frozen | The app stopped streaming; iOS suspends it when backgrounded. Keep DroidCam foregrounded and the phone unlocked. |
| Feed is live, no skeleton drawn | Hand out of frame or too small. Fill roughly a third of the frame height, keep it lit. |
| Motion lags visibly behind your hand | Wi-Fi congestion, or 1080p. Drop to `640x480`. |
| Left/right feels inverted | The feed is mirrored by default so it reads like a mirror. Pass `--no-mirror` to disable. |
