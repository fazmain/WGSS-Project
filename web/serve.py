#!/usr/bin/env python3
"""
Local server for the web version.

Serves the project root (so web/ can reach ../models/) with caching disabled,
so a plain reload always picks up code changes.

    python3 web/serve.py          →  http://localhost:8765/web/
"""

import functools
import http.server
from pathlib import Path

PORT = 8765
ROOT = Path(__file__).resolve().parent.parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    handler = functools.partial(NoCacheHandler, directory=str(ROOT))
    print(f"Serving {ROOT} at http://localhost:{PORT}/web/")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler).serve_forever()
