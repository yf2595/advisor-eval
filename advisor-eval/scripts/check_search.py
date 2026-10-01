#!/usr/bin/env python3
"""Check whether the hosts behind the GAIA tools are reachable (DuckDuckGo, Wikipedia, arXiv)."""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gaia_runner import _DEFAULT_HEADERS  # noqa: E402

PROBES = {
    "duckduckgo (web_search)": "https://html.duckduckgo.com/html/?q=Book+of+Esther",
    "wikipedia (wiki_search)": "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=Esther&format=json",
    "arxiv (arxiv_search)": "http://export.arxiv.org/api/query?search_query=all:electron&max_results=1",
}


def main() -> None:
    ok_all = True
    for name, url in PROBES.items():
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers=_DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            note = ""
            if "duckduckgo" in url and "result__a" not in body:
                note = " (page returned, but no search results in it)"
                ok_all = False
            print(f"OK    {name:26s} {resp.status} {time.time() - t0:.1f}s{note}")
        except Exception as exc:  # noqa: BLE001
            ok_all = False
            print(f"FAIL  {name:26s} {exc} ({time.time() - t0:.1f}s)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
