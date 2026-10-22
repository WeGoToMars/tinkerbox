"""Download a catalog model's bundle for this Mac, printing its progress as JSON lines."""

from __future__ import annotations

import argparse
import json

from tinkerbox_ane.models import CATALOG, download


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=[item["id"] for item in CATALOG])
    args = parser.parse_args(argv)
    download(args.model, lambda done, total: print(
        json.dumps({"progress": min(done / total, 1.0) if total else None}), flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
