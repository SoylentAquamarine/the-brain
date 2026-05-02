#!/usr/bin/env python3
"""
push_stats.py — Update README with current usage stats and push to GitHub.

Run this at the end of any coding session to keep the README stats current.
Completely decoupled from health checks — just reads stats/usage.json and publishes.

    python push_stats.py          # update README and push
    python push_stats.py --dry-run  # print what would be written, no changes
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Print output without writing or pushing")
    args = p.parse_args()

    script = Path(__file__).parent / "update_readme_stats.py"
    cmd = [sys.executable, str(script)]
    if args.dry_run:
        cmd.append("--dry-run")
    else:
        cmd.append("--push")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
