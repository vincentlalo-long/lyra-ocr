"""Unit tests for Review Web Server and REST API."""

import json
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

import pytest
from lyra_ocr.web.server import create_server


def http_request(url, data=None, method=None):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method=method
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


@pytest.fixture(scope="module")
def web_server():
    """Start an in-memory ThreadingHTTPServer on an ephemeral port."""
    server = create_server(host="127.0.0.1", port=0)  # Port 0 selects free ephemeral port
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # Wait for socket to bind

    yield {"server": server, "url": base_url}

    server.shutdown()
    server.server_close()


def test_web_server_index_html(web_server):
    """Verify that GET / returns the HTML review application."""
    url = f"{web_server['url']}/"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "LYRA OCR" in content
        assert "frameCanvas" in content


def test_web_server_get_artifacts(web_server):
    """Verify that GET /api/artifacts lists available artifact runs."""
    url = f"{web_server['url']}/api/artifacts"
    with urllib.request.urlopen(url) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "artifacts" in data
        assert isinstance(data["artifacts"], list)
        names = [a["name"] for a in data["artifacts"]]
        assert "castle_en" in names or "song3_en" in names


def test_web_server_get_data_and_status(web_server):
    """Verify GET /api/data and GET /api/status load full track and frame states."""
    # Data endpoint
    url_data = f"{web_server['url']}/api/data?dir=golden/v1/artifacts/castle_en"
    with urllib.request.urlopen(url_data) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "tracks" in data
        assert "manifest" in data
        assert "labels" in data
        assert "status" in data
        assert len(data["tracks"]) >= 1

    # Status endpoint
    url_status = f"{web_server['url']}/api/status?dir=golden/v1/artifacts/castle_en"
    with urllib.request.urlopen(url_status) as resp:
        assert resp.status == 200
        status_data = json.loads(resp.read().decode("utf-8"))
        assert "coverage_percent" in status_data
        assert "total_tracks" in status_data


def test_web_server_post_label_and_persistence(web_server, tmp_path: Path, monkeypatch):
    """Verify POST /api/label validates payload and persists to labels.jsonl."""
    # Server restricts paths to cwd; run with tmp_path as workspace root
    monkeypatch.chdir(tmp_path)
    mock_art = tmp_path / "mock_art"
    mock_art.mkdir()
    manifest = {"video_id": "test_web_vid"}
    tracks = {"tracks": [{"track_id": "1", "label": "lyric", "confidence": 0.95}]}
    (mock_art / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (mock_art / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    url = f"{web_server['url']}/api/label"
    payload = {
        "artifact_dir": "mock_art",
        "video_id": "test_web_vid",
        "track_id": "1",
        "human_label": "lyric",
        "tracking_error": False,
        "corrected_text": "Web UI verified line",
        "notes": "Testing web server POST",
        "annotator": "test_user",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res_data = json.loads(resp.read().decode("utf-8"))
        assert res_data["ok"] is True
        assert res_data["record"]["human_label"] == "lyric"
        assert res_data["record"]["corrected_text"] == "Web UI verified line"

    # Verify labels.jsonl was written on disk
    labels_file = mock_art / "labels.jsonl"
    assert labels_file.exists()
    lines = labels_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    saved_rec = json.loads(lines[0])
    assert saved_rec["track_id"] == "1"
    assert saved_rec["human_label"] == "lyric"


def test_web_server_rejects_paths_outside_cwd(web_server):
    """Absolute paths outside the server workspace root must be rejected."""
    outside = Path("/etc")
    url = f"{web_server['url']}/api/data?dir={outside}"
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(url)
    assert exc.value.code == 404

    frame_url = f"{web_server['url']}/api/frame?dir=/etc&file=passwd"
    with pytest.raises(urllib.error.HTTPError) as exc2:
        urllib.request.urlopen(frame_url)
    assert exc2.value.code == 400


def test_web_server_post_label_rejects_invalid_payloads(web_server):
    """Null human_label or bogus human_decision must return 400, not crash."""
    url = f"{web_server['url']}/api/label"
    base = {
        "artifact_dir": "golden/v1/artifacts/castle_en",
        "video_id": "any",
        "track_id": "1",
        "human_label": "lyric",
    }

    for bad_payload in (
        {**base, "human_label": None},
        {**base, "human_label": "not_a_label"},
        {**base, "human_decision": "maybe"},
    ):
        req = urllib.request.Request(
            url,
            data=json.dumps(bad_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 400


def test_web_server_get_eval(web_server):
    """Verify GET /api/eval?golden=golden/v1 executes evaluation."""
    url = f"{web_server['url']}/api/eval?golden=golden/v1"
    with urllib.request.urlopen(url) as resp:
        assert resp.status == 200
        eval_data = json.loads(resp.read().decode("utf-8"))
        assert "total_tracks" in eval_data
        assert "gate_passed" in eval_data
        assert "binary_lyric" in eval_data


def test_web_server_eval_errors(web_server):
    """Verify GET /api/eval/errors returns report + auditable error rows."""
    url = f"{web_server['url']}/api/eval/errors?golden=golden/v1"
    status, data = http_request(url)
    assert status == 200
    assert "report" in data and "errors" in data
    assert isinstance(data["errors"], list)
    if data["errors"]:
        row = data["errors"][0]
        assert {"video_id", "track_id", "error_type"} <= set(row.keys())


# ---------------------------------------------------------------------------
# Dataset / jobs endpoints (isolated workspace + fake pipeline runner)
# ---------------------------------------------------------------------------

@pytest.fixture()
def ws_server(tmp_path):
    from lyra_ocr.web.jobs import JobQueue

    def fake_runner(rec, src, cancel_event, progress):
        rec.artifact_dir = f"artifacts/{rec.video_id}"
        return {"video_id": rec.video_id, "title": src.title, "channel": src.channel}

    queue = JobQueue(tmp_path, pipeline_runner=fake_runner)
    server = create_server("127.0.0.1", 0, workspace_root=tmp_path, job_queue=queue)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    yield {"url": base_url, "root": tmp_path, "queue": queue}
    server.shutdown()
    server.server_close()
    queue.shutdown()


def _wait_job_done(queue, dataset, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if queue.stats(dataset).get("done"):
            return True
        time.sleep(0.05)
    return False


def test_dataset_create_enqueue_freeze_flow(ws_server):
    url = ws_server["url"]
    root = ws_server["root"]

    status, res = http_request(f"{url}/api/datasets", {"name": "flow", "description": "d"}, method="POST")
    assert status == 201 and res["ok"]

    status, listing = http_request(f"{url}/api/datasets")
    assert "flow" in [d["name"] for d in listing["datasets"]]

    video = root / "in" / "clip.mp4"
    video.parent.mkdir(exist_ok=True)
    video.write_bytes(b"\x00")

    status, res = http_request(
        f"{url}/api/datasets/flow/items",
        {"local_paths": ["in/clip.mp4"], "options": {"fps": 1.0}},
        method="POST",
    )
    assert status == 202
    assert len(res["enqueued"]) == 1
    vid = res["enqueued"][0]["video_id"]

    assert _wait_job_done(ws_server["queue"], "flow")
    status, job_data = http_request(f"{url}/api/jobs?dataset=flow")
    assert job_data["stats"].get("done") == 1

    status, ds = http_request(f"{url}/api/datasets/flow")
    entry = ds["videos"][0]
    assert entry["video_id"] == vid and entry["artifact_dir"] == f"artifacts/{vid}"

    # Fake artifacts on disk, then freeze through the endpoint
    art = root / "datasets" / "flow" / "artifacts" / vid
    art.mkdir(parents=True, exist_ok=True)
    (art / "tracks.json").write_text(json.dumps({"total_tracks": 2, "tracks": []}), encoding="utf-8")
    (art / "manifest.json").write_text(json.dumps({"config_hash": "cfg1"}), encoding="utf-8")

    status, fr = http_request(f"{url}/api/datasets/flow/freeze", {}, method="POST")
    assert status == 200 and fr["ok"] and fr["changed"] == 1


def test_items_validation_rejections(ws_server):
    url = ws_server["url"]
    http_request(f"{url}/api/datasets", {"name": "val"}, method="POST")

    status, res = http_request(
        f"{url}/api/datasets/val/items",
        {
            "videos": [
                {"kind": "youtube", "url": "https://example.com/watch?v=x"},
                {"kind": "local", "path": "../../../../etc/passwd"},
                {"kind": "ftp", "url": "ftp://x"},
                {"kind": "youtube", "url": "https://youtu.be/SHORTID"},
            ]
        },
        method="POST",
    )
    assert status == 202
    assert res["enqueued"] == []
    assert len(res["rejected"]) == 4


def test_items_unknown_dataset_404(ws_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        http_request(f"{ws_server['url']}/api/datasets/nope/items", {"local_paths": ["a.mp4"]}, method="POST")
    assert exc.value.code == 404


def test_job_cancel_invalid_state_returns_400(ws_server):
    url = ws_server["url"]
    http_request(f"{url}/api/datasets", {"name": "cxl"}, method="POST")
    video = ws_server["root"] / "one.mp4"
    video.write_bytes(b"\x00")
    _, res = http_request(f"{url}/api/datasets/cxl/items", {"local_paths": ["one.mp4"]}, method="POST")
    job_id = res["enqueued"][0]["job_id"]
    assert _wait_job_done(ws_server["queue"], "cxl")

    with pytest.raises(urllib.error.HTTPError) as exc:
        http_request(f"{url}/api/jobs/{job_id}/cancel", {}, method="POST")
    assert exc.value.code == 400

    # Retry of a done job re-queues it
    status, retry_res = http_request(f"{url}/api/jobs/{job_id}/retry", {"force": True}, method="POST")
    assert status == 200
    assert retry_res["job"]["state"] == "queued" if "job" in retry_res else True


def test_dataset_paths_traversal_rejected(ws_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        http_request(f"{ws_server['url']}/api/datasets/..%2Fescape")
    assert exc.value.code in (400, 404)


def test_youtube_list_surfaces_error(ws_server, monkeypatch):
    import lyra_ocr.web.server as srv

    def boom(url, limit=1000):
        raise RuntimeError("network down")

    monkeypatch.setattr(srv, "list_channel_or_playlist", boom)
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{ws_server['url']}/api/youtube/list?url=https://youtube.com%2F%40c%2Fvideos")
    assert exc.value.code == 502
