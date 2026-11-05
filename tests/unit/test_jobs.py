"""Unit tests for the background JobQueue (fake pipeline runner)."""

import json
import time

import pytest

from lyra_ocr.dataset import registry
from lyra_ocr.dataset.models import VideoSource
from lyra_ocr.pipeline import PipelineCancelled
from lyra_ocr.web.jobs import JobQueue
from lyra_ocr.web.runners import _downloaded_videos


def make_queue(tmp_path, runner):
    q = JobQueue(tmp_path, pipeline_runner=runner)
    q.start()
    return q


def wait_for_states(q, dataset, expected, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        stats = q.stats(dataset)
        if all(stats.get(state, 0) >= count for state, count in expected.items()):
            return True
        time.sleep(0.05)
    return False


def ok_runner(rec, src, cancel_event, progress):
    rec.artifact_dir = f"artifacts/{rec.video_id}"
    progress("ocr", 55.0, "mid")
    return {"video_id": rec.video_id, "title": src.title, "channel": src.channel}


@pytest.fixture()
def dataset(tmp_path):
    registry.create_dataset(tmp_path, "qd")
    return tmp_path


def test_enqueue_run_done_and_manifest_sync(dataset):
    q = make_queue(dataset, ok_runner)
    try:
        created = q.enqueue("qd", [VideoSource(kind="local", path="/x/a.mp4", yt_id="")])
        assert len(created) == 1
        assert wait_for_states(q, "qd", {"done": 1})
        rec = q.get(created[0].job_id)
        assert rec.state == "done"
        assert rec.percent == 100.0
        data = registry.get_dataset(dataset, "qd")
        assert [v["video_id"] for v in data["videos"]] == [rec.video_id]
    finally:
        q.shutdown()


def test_dedup_and_failed_retry_reenqueue(dataset):
    def fail_once(rec, src, cancel_event, progress):
        if rec.attempts == 0:
            raise RuntimeError("boom stage failure")
        return ok_runner(rec, src, cancel_event, progress)

    q = make_queue(dataset, fail_once)
    try:
        src = VideoSource(kind="youtube", url="https://youtu.be/AAAAAAAAAAA", yt_id="AAAAAAAAAAA")
        first = q.enqueue("qd", [src])
        assert wait_for_states(q, "qd", {"failed": 1})
        rec = q.get(first[0].job_id)
        assert "boom" in rec.error
        assert rec.failed_stage

        # Same video_id re-enqueue is deduplicated while job exists (failed → allowed)
        second = q.enqueue("qd", [src])
        assert len(second) == 1  # failed jobs do not block re-enqueue

        q.retry(first[0].job_id)
        assert wait_for_states(q, "qd", {"done": 1}, timeout=8.0)
    finally:
        q.shutdown()


def test_retry_from_done_requires_no_error_state_gate(dataset):
    q = make_queue(dataset, ok_runner)
    try:
        created = q.enqueue("qd", [VideoSource(kind="local", path="/x/b.mp4")])
        assert wait_for_states(q, "qd", {"done": 1})
        rec = q.retry(created[0].job_id, force=True)
        assert rec.state == "queued"
        assert rec.options.get("force") is True
        with pytest.raises(ValueError):
            q.retry(rec.job_id)  # already queued, not retryable
    finally:
        q.shutdown()


def test_cancel_queued_and_persistence_reload(dataset):
    q = JobQueue(dataset, pipeline_runner=ok_runner)  # worker NOT started: jobs stay queued
    srcs = [VideoSource(kind="local", path=f"/x/c{i}.mp4") for i in (0, 1)]
    created = q.enqueue("qd", srcs)
    assert len(created) == 2

    cancelled = q.cancel(created[0].job_id)
    assert cancelled.state == "cancelled"

    jobs_file = dataset / "datasets" / "qd" / "jobs.json"
    assert jobs_file.is_file()
    rows = json.loads(jobs_file.read_text(encoding="utf-8"))
    assert len(rows) == 2
    states = {r["job_id"]: r["state"] for r in rows}
    assert states[created[0].job_id] == "cancelled"
    assert states[created[1].job_id] == "queued"

    # Simulate a crash mid-run: mark the second job 'running' on disk
    rows = json.loads(jobs_file.read_text(encoding="utf-8"))
    rows[1]["state"] = "running"
    jobs_file.write_text(json.dumps(rows), encoding="utf-8")

    # Restart: interrupted 'running' resumes as queued and completes; cancelled stays terminal
    q2 = JobQueue(dataset, pipeline_runner=ok_runner)
    q2.start()
    try:
        assert wait_for_states(q2, "qd", {"done": 1, "cancelled": 1}, timeout=8.0)
    finally:
        q2.shutdown()


def test_cancelled_pipeline_maps_to_cancelled_state(dataset):
    def cancelling_runner(rec, src, cancel_event, progress):
        raise PipelineCancelled("cancelled at ocr")

    q = make_queue(dataset, cancelling_runner)
    try:
        q.enqueue("qd", [VideoSource(kind="local", path="/x/d.mp4")])
        assert wait_for_states(q, "qd", {"cancelled": 1})
    finally:
        q.shutdown()


def test_downloaded_videos_matches_only_the_completed_merge(tmp_path):
    """Already-downloaded detection must use the literal "[<id>]" suffix.

    An id interpolated into a glob pattern becomes a character class: it matches
    no yt-dlp output name while still matching stray ".f140.mp4" fragments.
    """
    videos = tmp_path / "videos"
    videos.mkdir()
    other = videos / "other_song [AAAAAAAAAAA].mp4"
    hit = videos / "song_one [dQw4w9WgXcQ].mp4"
    fragment = videos / "song_one [dQw4w9WgXcQ].f140.mp4"
    for path in (other, hit, fragment):
        path.write_bytes(b"x")

    found = _downloaded_videos(videos, "dQw4w9WgXcQ")
    assert found == [hit]
    assert _downloaded_videos(videos, "") == []
    assert _downloaded_videos(tmp_path / "missing", "dQw4w9WgXcQ") == []
