"""Forward a local TCP port to a port on a USB-attached iPhone, via usbmuxd.

Why this exists
---------------
The teleop camera normally arrives over Wi-Fi: DroidCam serves MJPEG on the
phone and OpenCV opens `http://<phone-ip>:4747/video`. That works, but it puts
the whole control loop behind the venue's Wi-Fi -- and Wi-Fi is exactly what
failed during bring-up (100% packet loss mid-session, and an earlier run at
33% loss dropped teleop from ~14 FPS to ~6).

USB removes that variable entirely. `usbmuxd` (already running on this box --
it's what lets Nautilus browse the iPhone over AFC) multiplexes TCP
connections to a tethered iOS device. Point this at the phone's DroidCam port
and the stream becomes `http://127.0.0.1:4747/video`, over the cable.

The usual tool for this is `iproxy` from libimobiledevice-utils, which isn't
installed and needs a password to apt-install. usbmuxd's protocol is small and
its socket is world-writable, so this speaks it directly instead: plist
messages over a Unix socket, stdlib only.

Protocol: a 16-byte little-endian header (length, version=1, message=8/plist,
tag) followed by an XML plist body. `Connect` turns the control connection
itself into a raw pipe to the device port -- so after a successful Connect, the
same socket is just bytes in both directions.

Usage:
    python scripts/gb10/usbmux_tunnel.py --device-port 4747
    # then: curl -sI http://127.0.0.1:4747/video   (or point teleop at it)
"""

from __future__ import annotations

import argparse
import plistlib
import socket
import socketserver
import struct
import sys
import threading

USBMUXD_SOCKET = "/var/run/usbmuxd"
_MESSAGE_PLIST = 8
_VERSION = 1

_tag_lock = threading.Lock()
_tag_counter = 0


def _next_tag() -> int:
    global _tag_counter
    with _tag_lock:
        _tag_counter += 1
        return _tag_counter


def _connect_usbmuxd(timeout: float = 10.0) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(USBMUXD_SOCKET)
    return sock


def _send(sock: socket.socket, message: dict) -> int:
    payload = plistlib.dumps(
        {"ClientVersionString": "simbiote-teleop", "ProgName": "simbiote", **message}
    )
    tag = _next_tag()
    sock.sendall(struct.pack("<IIII", 16 + len(payload), _VERSION, _MESSAGE_PLIST, tag) + payload)
    return tag


def _recv(sock: socket.socket) -> dict:
    header = b""
    while len(header) < 16:
        chunk = sock.recv(16 - len(header))
        if not chunk:
            raise ConnectionError("usbmuxd closed the connection")
        header += chunk
    length = struct.unpack("<I", header[:4])[0]
    body = b""
    while len(body) < length - 16:
        chunk = sock.recv(length - 16 - len(body))
        if not chunk:
            raise ConnectionError("usbmuxd closed mid-message")
        body += chunk
    return plistlib.loads(body)


def list_devices() -> list[dict]:
    sock = _connect_usbmuxd()
    try:
        _send(sock, {"MessageType": "ListDevices"})
        return _recv(sock).get("DeviceList", [])
    finally:
        sock.close()


def pick_device(udid: str | None = None) -> tuple[int, str]:
    """Return (device_id, udid) for the tethered device."""

    devices = list_devices()
    if not devices:
        raise RuntimeError(
            "no iOS device visible to usbmuxd. Is the phone plugged in, unlocked, "
            "and trusted ('Trust This Computer')?"
        )
    for device in devices:
        properties = device.get("Properties", {})
        if udid is None or properties.get("SerialNumber") == udid:
            return int(device["DeviceID"]), properties.get("SerialNumber", "?")
    raise RuntimeError(f"no device matching udid {udid!r}; found {[d['Properties'].get('SerialNumber') for d in devices]}")


def open_device_port(device_id: int, port: int) -> socket.socket:
    """Open a raw byte pipe to `port` on the device.

    On success the control socket *becomes* the tunnel, so it's returned
    rather than closed.
    """

    sock = _connect_usbmuxd()
    # usbmuxd wants the port in network byte order inside a host-order field.
    _send(sock, {"MessageType": "Connect", "DeviceID": device_id,
                 "PortNumber": struct.unpack(">H", struct.pack("<H", port))[0]})
    reply = _recv(sock)
    result = reply.get("Number", -1)
    if result != 0:
        sock.close()
        hint = {
            2: f"device is not listening on port {port} -- is DroidCam actually running "
               "and foregrounded on the phone?",
            3: "usbmuxd refused the connection (port not allowed)",
            5: "device is locked -- unlock the iPhone",
        }.get(result, "")
        raise ConnectionError(f"usbmuxd Connect to port {port} failed (Number={result}). {hint}")
    sock.settimeout(None)
    return sock


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class _Handler(socketserver.BaseRequestHandler):
    device_id: int = 0
    device_port: int = 0

    def handle(self) -> None:
        try:
            device = open_device_port(self.device_id, self.device_port)
        except Exception as exc:  # noqa: BLE001 - report and drop this connection only
            print(f"[usbmux] connection refused: {exc}", file=sys.stderr)
            return
        # MJPEG is a long-lived stream in one direction plus a short request in
        # the other, so both directions get their own pump.
        up = threading.Thread(target=_pump, args=(self.request, device), daemon=True)
        up.start()
        _pump(device, self.request)
        up.join(timeout=1.0)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-port", type=int, default=4747, help="Port on the iPhone (DroidCam: 4747).")
    parser.add_argument("--local-port", type=int, default=0, help="Local port (default: same as --device-port).")
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--udid", default=None, help="Pick a specific device by UDID.")
    parser.add_argument("--list", action="store_true", help="List tethered devices and exit.")
    args = parser.parse_args()

    if args.list:
        for device in list_devices():
            properties = device.get("Properties", {})
            print(f"DeviceID={device['DeviceID']} udid={properties.get('SerialNumber')} "
                  f"connection={properties.get('ConnectionType')}")
        return 0

    device_id, udid = pick_device(args.udid)
    local_port = args.local_port or args.device_port

    _Handler.device_id = device_id
    _Handler.device_port = args.device_port

    server = _Server((args.local_host, local_port), _Handler)
    print(f"[usbmux] device {udid} (DeviceID={device_id}) over USB")
    print(f"[usbmux] http://{args.local_host}:{local_port}  ->  iPhone port {args.device_port}")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[usbmux] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
