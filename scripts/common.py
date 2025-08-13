from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import IO, Any

try:  # prefer loguru when available
    from loguru import logger  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - lightweight fallback
    import logging

    logger = logging.getLogger("twentyone.common")


def find_or_build_bridge() -> str:
    """Return the path to the compiled `twentyone_bridge` binary.

    Respects the `TWENTYONE_BRIDGE_BIN` env var. If the binary is not found
    under the project `target/debug` directory, triggers a `cargo build` for
    the bridge binary.
    """
    override = os.environ.get("TWENTYONE_BRIDGE_BIN")
    if override:
        logger.debug("Using TWENTYONE_BRIDGE_BIN override: {}", override)
        return override

    root = Path(__file__).resolve().parents[1]
    bin_path = root / "target" / "debug" / "twentyone_bridge"
    if not bin_path.exists():
        logger.info("Building missing bridge binary at {}", bin_path)
        subprocess.run(["cargo", "build", "--bin", "twentyone_bridge"], cwd=root, check=True)
    return str(bin_path)


class Bridge:
    """Thin JSON line protocol wrapper around the Rust bridge binary.

    Usage:
        with Bridge() as br:
            br.send({"cmd": "new", "seed": 42})
            ...
    """

    def __init__(self, path: str | None = None) -> None:
        if path is None:
            path = find_or_build_bridge()
        self._proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Failed to open bridge stdio pipes")
        self._in: IO[str] = self._proc.stdin
        self._out: IO[str] = self._proc.stdout

    def send(self, obj: dict[str, Any]) -> Any:
        """Send a single JSON command and return the `data` of the response.

        Raises RuntimeError on protocol errors or if the bridge reports an error.
        """
        line = json.dumps(obj)
        self._in.write(line + "\n")
        self._in.flush()
        out = self._out.readline()
        if not out:
            raise RuntimeError("bridge closed")
        try:
            resp = json.loads(out)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON from bridge: {}", out.strip())
            raise RuntimeError(f"invalid bridge response: {e}") from e
        if resp.get("status") == "err":
            raise RuntimeError(str(resp.get("error")))
        return resp["data"]

    def close(self) -> None:
        try:
            self.send({"cmd": "quit"})
        except Exception as e:  # noqa: BLE001 - best effort shutdown
            logger.debug("Bridge close encountered: {}", e)
        finally:
            self._proc.terminate()

    def __enter__(self) -> "Bridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()
