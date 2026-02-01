import os
import signal
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socket import socket


def _can_bind(port: int) -> bool:
    try:
        with socket() as s:
            s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def _pick_port(preferred: int) -> int:
    if _can_bind(preferred):
        return preferred
    for p in range(preferred + 1, preferred + 50):
        if _can_bind(p):
            return p
    raise RuntimeError("No free port found")


def main() -> int:
    root = Path(__file__).resolve().parent

    api_port = 8081
    if not _can_bind(api_port):
        print("Port 8081 is busy. Stop the existing API and retry.", file=sys.stderr)
        return 1

    web_port = _pick_port(5173)

    env = os.environ.copy()
    env["PORT"] = str(api_port)
    env["PYTHONUNBUFFERED"] = "1"

    api = subprocess.Popen([sys.executable, str(root / "api.py")], cwd=str(root), env=env)

    def _shutdown(*_args):
        if api.poll() is None:
            api.send_signal(signal.SIGINT)
            try:
                api.wait(timeout=3)
            except Exception:
                api.kill()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=str(root), **kwargs)
    httpd = ThreadingHTTPServer(("127.0.0.1", web_port), handler)

    print(f"Frontend: http://127.0.0.1:{web_port}/", file=sys.stderr)
    print(f"Backend:  http://127.0.0.1:{api_port}/api/health", file=sys.stderr)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        _shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
