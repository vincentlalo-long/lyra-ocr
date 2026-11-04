"""Recompute and freeze artifact hashes for a golden dataset manifest.

Run this after intentionally regenerating golden artifacts. It keeps the
manifest in sync so evaluation can reject stale/edited tracks.json files.

Usage:
    python scripts/freeze_golden.py [golden_dir]
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from lyra_ocr.labels import sha256_prefix


def resolve_artifact_dir(root: Path, golden_dir: Path, artifact_dir: str) -> Path:
    p = Path(artifact_dir)
    if p.is_absolute():
        return p
    for candidate in (root / p, Path.cwd() / p, golden_dir / p):
        if candidate.exists():
            return candidate
    return root / p


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    golden_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "golden" / "v1"
    manifest_path = golden_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"Golden manifest not found: {manifest_path}")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    changed = 0
    for video in manifest.get("videos", []):
        art_dir = resolve_artifact_dir(root, golden_dir, video.get("artifact_dir", ""))
        tracks_path = art_dir / "tracks.json"
        if not tracks_path.exists():
            print(f"ERROR: missing {tracks_path}")
            return 1

        new_hash = sha256_prefix(tracks_path)
        old_hash = video.get("artifact_hash")
        if new_hash != old_hash:
            print(f"{video.get('video_id')}: {old_hash} -> {new_hash}")
            video["artifact_hash"] = new_hash
            changed += 1

        art_manifest_path = art_dir / "manifest.json"
        if art_manifest_path.exists():
            with open(art_manifest_path, "r", encoding="utf-8") as f:
                art_manifest = json.load(f)
            config_hash = art_manifest.get("config_hash")
            if config_hash:
                video["config_hash"] = config_hash

    manifest["frozen_at"] = date.today().isoformat()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Froze {len(manifest.get('videos', []))} videos ({changed} hash changes) in {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
