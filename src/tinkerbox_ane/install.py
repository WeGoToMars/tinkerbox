"""Install a locally compiled bundle as a catalog model, so the app and load() use it without the Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinkerbox_ane.models import CATALOG, install


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=[item["id"] for item in CATALOG])
    parser.add_argument("bundle", type=Path, help="compiled bundle directory containing model.json")
    args = parser.parse_args(argv)
    print(install(args.model, args.bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
