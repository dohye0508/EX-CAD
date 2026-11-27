"""EX-CAD AI Direct Bridge client (standalone C++ app, TCP JSON lines on 127.0.0.1:9877).

Usage:
    python excad_client.py '{"cmd":"ping"}'
    python excad_client.py --demo
"""
import json
import socket
import sys


def send(commands, host="127.0.0.1", port=9877, timeout=120.0):
    replies = []
    with socket.create_connection((host, port), timeout=timeout) as s:
        f = s.makefile("rwb")
        for cmd in commands:
            f.write((json.dumps(cmd) + "\n").encode("utf-8"))
            f.flush()
            replies.append(json.loads(f.readline().decode("utf-8")))
    return replies


DEMO = [
    {"cmd": "ping"},
    {"cmd": "clear"},
    {"cmd": "sdf_humanoid", "height": 1.8, "muscularity": 0.8, "resolution": 110},
    {"cmd": "sketch_new", "plane": "XY"},
    {"cmd": "sketch_add_rect", "u0": 0.7, "v0": -0.3, "width": 0.6, "height": 0.6},
    {"cmd": "sketch_solve"},
    {"cmd": "sketch_extrude", "depth": 0.4, "name": "Block"},
    {"cmd": "list"},
    {"cmd": "frame_all"},
]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--demo":
        cmds = [json.loads(a) for a in sys.argv[1:]]
    else:
        cmds = DEMO
    for c, r in zip(cmds, send(cmds)):
        print(c.get("cmd"), "->", json.dumps(r))
