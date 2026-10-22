"""Freeze what a run is built from: git commit, configs, caches, checkpoints.

Run before launching any experiment and keep the output beside its results, so
a number can always be traced to the exact code and data that produced it
(a review on 2026-10-22 found code edited while step 6 was running). Refuses a
dirty worktree unless --allow-dirty; small files are hashed in full, multi-GB
caches by size plus the hash of their first and last 64 MB.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

CHUNK = 64 * 2 ** 20


def digest(path):
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size <= 2 * CHUNK:
            h.update(f.read())
        else:
            h.update(f.read(CHUNK))
            f.seek(size - CHUNK)
            h.update(f.read(CHUNK))
    return {"size": size, "sha256": h.hexdigest()[:16], "full": size <= 2 * CHUNK}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True, help="files or directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not a.allow_dirty:
        raise SystemExit(f"worktree has uncommitted changes:\n{dirty}")
    files = {}
    for p in map(Path, a.paths):
        for f in ([p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())):
            files[str(f)] = digest(f)
    Path(a.out).write_text(json.dumps({
        "written": datetime.now().isoformat(timespec="seconds"),
        "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty), "files": files}, indent=2))
    print(f"wrote {a.out}: {len(files)} files at {git('rev-parse', '--short', 'HEAD')}")


if __name__ == "__main__":
    main()
