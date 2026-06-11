#!/usr/bin/env python3
"""Tiny bridge for the local web board: serves board/index.html and relays
moves to a single `azt uci` engine process.

Usage: board_server.py [port] [--sims N]   (run with DYLD_LIBRARY_PATH set)
"""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8643
SIMS = sys.argv[sys.argv.index("--sims") + 1] if "--sims" in sys.argv else "2000"

LIB = None
for root, dirs, files in os.walk(os.path.join(HERE, "target", "release", "build")):
    if "libtorch_cpu.dylib" in files:
        LIB = root
        break

env = dict(os.environ)
if LIB:
    env["DYLD_LIBRARY_PATH"] = LIB
engine = subprocess.Popen(
    [os.path.join(HERE, "target", "release", "azt"), "uci", "--sims", SIMS],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    env=env,
    cwd=HERE,
)
lock = threading.Lock()


def engine_cmd(lines, until):
    with lock:
        for line in lines:
            engine.stdin.write(line + "\n")
        engine.stdin.flush()
        out = []
        while True:
            line = engine.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            out.append(line.strip())
            if line.startswith(until):
                return out


engine_cmd(["uci", "isready"], "readyok")
print(f"engine ready ({SIMS} sims/move); board at http://localhost:{PORT}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        try:
            with open(os.path.join(HERE, "board", "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/engine":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        moves = req.get("moves", "").strip()
        position = "position startpos" + (f" moves {moves}" if moves else "")
        out = engine_cmd([position, "go"], "bestmove")
        q = None
        for line in out:
            if "string q" in line:
                q = float(line.split("string q")[1].strip())
        best = out[-1].split()[1]
        body = json.dumps({"move": best, "q": q}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
