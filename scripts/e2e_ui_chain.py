#!/usr/bin/env python
"""Acceptance harness for the UI chain: dataset -> ingest -> run -> label -> promote -> eval.

Drives the real HTTP API of the web server against a synthetic karaoke video, so
the whole labelling loop can be proven without a YouTube download. Each text line
in the fixture has a known span, which is what makes the turnover split and the
lyric/non_lyric rules measurable.

Usage:
    .venv/bin/python scripts/e2e_ui_chain.py [--port 8765] [--workspace /tmp/lyra_e2e_ws]
Requires ffmpeg with drawtext and a TrueType font on disk.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

FONTS = [
    "/usr/share/fonts/google-droid-sans-fonts/DroidSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
]

LYRIC_LINES = [
    "We can be alone in the castle in the sky",
    "The moon is a lantern hanging low",
]
TITLE_CARD = "CASTLE IN THE SKY - OFFICIAL LYRIC VIDEO"
WATERMARK = "@lyraofficial"

# (text, start_sec, end_sec, expected label) as constructed by the fixture.
FIXTURE = [
    (WATERMARK, 0, 12, "non_lyric"),
    (LYRIC_LINES[0], 0, 4, "lyric"),
    (LYRIC_LINES[1], 4, 8, "lyric"),
    (TITLE_CARD, 8, 12, "non_lyric"),
]


class Harness:
    def __init__(self, base: str) -> None:
        self.base = base
        self.failures: list[str] = []

    def request(self, method: str, path: str, body: dict | None = None, timeout: float = 60.0):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
        )
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode()
            try:
                return exc.code, json.loads(raw)
            except json.JSONDecodeError:
                return exc.code, {"raw": raw}

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  :: {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def wait_jobs(self, matches, label: str, timeout: float = 900.0) -> list[dict]:
        deadline = time.time() + timeout
        previous = None
        while time.time() < deadline:
            _, data = self.request("GET", "/api/jobs")
            jobs = [j for j in data.get("jobs", []) if matches(j)]
            if jobs and all(j["state"] in ("done", "failed", "cancelled") for j in jobs):
                return jobs
            snapshot = [(j["state"], j.get("stage"), round(j.get("percent", 0))) for j in jobs]
            if snapshot != previous:
                print(f"   [{label}] {snapshot}")
                previous = snapshot
            time.sleep(1.0)
        raise TimeoutError(f"jobs for {label} did not finish within {timeout}s")


def synth_video(target: Path, font: str) -> None:
    """One 12s clip: a persistent corner handle, two lyric lines, one title card."""
    overlays = [
        f"drawtext=fontfile={font}:text='{WATERMARK}':fontcolor=0x888888:fontsize=24:"
        f"x=20:y=20:enable='between(t,0,12)'"
    ]
    for text, start, end, _ in FIXTURE[1:]:
        overlays.append(
            f"drawtext=fontfile={font}:text='{text}':fontcolor=white:fontsize=28:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,{start},{end})'"
        )
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x101418:s=960x540:d=12:r=25",
         "-vf", ",".join(overlays), "-pix_fmt", "yuv420p", "-c:v", "libx264",
         "-preset", "ultrafast", str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-1200:], file=sys.stderr)
        raise SystemExit("fixture synthesis failed")


def run_chain(h: Harness, workspace: Path) -> None:
    dataset = "e2e_ds"
    h.check(
        "create dataset",
        h.request("POST", "/api/datasets", {"name": dataset, "description": "synthetic"})[0] == 201,
    )

    status, body = h.request(
        "POST", f"/api/datasets/{dataset}/download",
        {"videos": [{"kind": "local", "path": "inbox/synth_lyric.mp4"}]},
    )
    h.check("queue ingest", status == 202 and bool(body.get("enqueued")), str(body.get("rejected")))
    jobs = h.wait_jobs(lambda j: j["dataset"] == dataset and j["kind"] == "ingest", "ingest")
    h.check("ingest done", all(j["state"] == "done" for j in jobs), jobs[-1].get("error", ""))

    _, ds = h.request("GET", f"/api/datasets/{dataset}")
    videos = ds.get("videos", [])
    h.check("manifest registers the video", len(videos) == 1 and videos[0]["downloaded"], str(videos))
    video_id = videos[0]["video_id"]

    status, body = h.request("POST", f"/api/datasets/{dataset}/runs", {"tracker": "greedy", "fps": 1.0})
    h.check("create run", status == 201, str(body))
    run_id = body["run"]["run_id"]

    h.request("POST", f"/api/datasets/{dataset}/runs/{run_id}/items", {"video_ids": "all"})
    jobs = h.wait_jobs(lambda j: j.get("run_id") == run_id, "run")
    h.check("run job done", all(j["state"] == "done" for j in jobs), jobs[-1].get("error", ""))

    artifact = workspace / "datasets" / dataset / "runs" / run_id / "videos" / video_id
    tracks = json.loads((artifact / "tracks.json").read_text()).get("tracks", [])
    for track in tracks:
        print(
            f"      track {track['track_id']:>2} {track['start_ms']:>5}-{track['end_ms']:>5} "
            f"{track['label']:<9} {track['decision']:<7} {track['confidence']:.2f} "
            f"| {track['features']['dominant_text'][:44]}"
        )
    h.check("one track per text identity", len(tracks) == len(FIXTURE), f"{len(tracks)} tracks")

    for text, start_sec, _, expected in FIXTURE:
        match = next(
            (t for t in tracks if t["features"]["dominant_text"].startswith(text[:18])), None
        )
        verdict = (
            {"label": match["label"], "decision": match["decision"],
             "span": [match["start_ms"], match["end_ms"]]}
            if match else None
        )
        ok = (
            match is not None
            and match["start_ms"] == start_sec * 1000
            and (match["label"] == expected or match["decision"] == "review")
        )
        h.check(f"{text[:18]!r} is {expected}", ok, str(verdict))

    def expected_for(track: dict) -> str:
        text = track["features"]["dominant_text"]
        return next((exp for txt, _, _, exp in FIXTURE if text.startswith(txt[:18])), "non_lyric")

    for track in tracks:
        h.request("POST", "/api/label", {
            "artifact_dir": f"datasets/{dataset}/runs/{run_id}/videos/{video_id}",
            "video_id": video_id,
            "track_id": track["track_id"],
            "human_label": expected_for(track),
            "human_decision": "accept",
            "annotator": "e2e_harness",
        })
    h.check("labels persisted", (artifact / "labels.jsonl").is_file())

    status, body = h.request(
        "POST", f"/api/datasets/{dataset}/ground_truth/promote", {"run_id": run_id}
    )
    h.check("promote ground truth", status == 200 and body.get("total", 0) > 0, str(body))

    status, body = h.request("GET", f"/api/eval?dataset={dataset}&run={run_id}")
    report = body.get("report", {})
    h.check(
        "eval scores the run",
        status == 200 and report.get("coverage_rate") == 1.0,
        json.dumps({k: report.get(k) for k in ("coverage_rate", "binary_lyric", "gate_passed")}),
    )

    status, body = h.request("POST", f"/api/datasets/{dataset}/runs", {"tracker": "bogus", "fps": 1.0})
    h.check("unknown tracker rejected at creation", status == 400, str(body))

    # Second tracker on the same ground truth: the comparison the UI offers.
    status, body = h.request("POST", f"/api/datasets/{dataset}/runs", {"tracker": "bytetrack", "fps": 1.0})
    other_run = body["run"]["run_id"]
    h.request("POST", f"/api/datasets/{dataset}/runs/{other_run}/items", {"video_ids": "all"})
    jobs = h.wait_jobs(lambda j: j.get("run_id") == other_run, "bytetrack run")
    h.check("bytetrack run done", all(j["state"] == "done" for j in jobs), jobs[-1].get("error", ""))
    status, body = h.request("GET", f"/api/eval?dataset={dataset}&run=all")
    h.check(
        "both trackers scored side by side",
        status == 200 and len(body.get("reports", [])) == 2,
        json.dumps([
            (r["run_id"].split("-")[0], r["report"].get("gate_passed"),
             r["report"].get("binary_lyric", {}).get("f1"))
            for r in body.get("reports", [])
        ]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/tmp/lyra_e2e_ws")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    font = next((f for f in FONTS if Path(f).is_file()), None)
    if font is None:
        raise SystemExit("no TrueType font available for the fixture")

    workspace = Path(args.workspace)
    inbox = workspace / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    video = inbox / "synth_lyric.mp4"
    if not video.is_file():
        synth_video(video, font)

    from lyra_ocr.web.server import create_server

    if (workspace / "datasets" / "e2e_ds").exists():
        shutil.rmtree(workspace / "datasets" / "e2e_ds")

    httpd = create_server("127.0.0.1", args.port, workspace_root=workspace)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    harness = Harness(f"http://127.0.0.1:{args.port}")
    print(f"harness against {harness.base}, workspace {workspace}")
    try:
        run_chain(harness, workspace)
    finally:
        httpd.shutdown()
        httpd.job_queue.shutdown()
        httpd.server_close()

    print("\n" + ("ALL PASS" if not harness.failures
                  else f"FAILURES ({len(harness.failures)}): " + ", ".join(harness.failures)))
    return 1 if harness.failures else 0


if __name__ == "__main__":
    sys.exit(main())
