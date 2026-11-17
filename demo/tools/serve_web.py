"""Serve web/ locally with correct MIME types (Windows can map .js to text/plain, which breaks modules).

    python tools/serve_web.py [port]
"""
import functools
import http.server
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".html": "text/html",
        ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream",
        ".bin": "application/octet-stream", ".jpg": "image/jpeg", ".png": "image/png",
    }


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=str(WEB)))
    print(f"Serving {WEB} at http://localhost:{port}")
    server.serve_forever()
