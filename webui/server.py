#!/usr/bin/env python3
"""
Local web UI for clip.py: paste a YouTube URL, watch progress, download the
generated clips. Runs clip.py as a subprocess per job and parses its stdout
(including yt-dlp's and Whisper's own progress indicators, which use \\r to
overwrite a line rather than \\n) to drive a real progress bar.

Usage:
    ./venv/bin/python webui/server.py
    -> open http://127.0.0.1:5050
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, abort

PROJECT_DIR = Path(__file__).parent.parent
VENV_PYTHON = PROJECT_DIR / "venv" / "bin" / "python"
CLIP_PY = PROJECT_DIR / "clip.py"
OUTPUT_DIR = PROJECT_DIR / "output"
DOWNLOAD_DIR = PROJECT_DIR / "downloads"
CAMPAIGNS_DIR = PROJECT_DIR / "campaigns"  # uploaded campaign-brief PDFs, one per job

sys.path.insert(0, str(PROJECT_DIR))
import clip  # noqa: E402 - reuses clip.py's Ollama helpers for the caption chatbox

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4GB - covers a campaign PDF or an uploaded video file

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

jobs: dict = {}
jobs_lock = threading.Lock()
active_job_id: str | None = None  # only one job runs at a time - see api_generate

NUM_CLIPS = 5
# Rough phase weights for the overall progress bar. Transcription dominates
# real runtime, so it gets the biggest share; the rest is a coarse estimate
# since we don't have a reliable way to predict total time up front.
DOWNLOAD_RANGE = (0, 15)
TRANSCRIBE_RANGE = (15, 65)
HIGHLIGHT_RANGE = (65, 70)
CLIPS_RANGE = (70, 100)

DOWNLOAD_PCT_RE = re.compile(r"\[download\]\s+([\d.]+)%\s+of")
WHISPER_PCT_RE = re.compile(r"^\s*(\d+)%\|")
CLIP_DONE_RE = re.compile(r"===\s*Clip\s+(\d+)/(\d+)\s+rendered\s*===")
RANKING_LINE_RE = re.compile(r"^\s*#\d+\s+Clip\s+(\d+):\s+(\d+)%")
DONE_RE = re.compile(r"^Done! (\d+) clip")
FILE_LINE_RE = re.compile(r"^\s{2}(.+\.mp4)\s*$")
CAPTION_LINE_RE = re.compile(r"^\s{4}caption:\s*(.*)$")


def _iter_progress_chunks(stream):
    """Yields each chunk of output split on \\r OR \\n, so progress bars
    that overwrite a line in place (yt-dlp, Whisper's tqdm) still produce
    a stream of updates instead of one giant buffered line."""
    buf = ""
    while True:
        char = stream.read(1)
        if char == "":
            if buf:
                yield buf
            break
        if char in ("\r", "\n"):
            if buf:
                yield buf
            buf = ""
        else:
            buf += char


def _scale(pct: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * max(0.0, min(100.0, pct)) / 100.0


def _run_job(job_id: str, url: str | None, campaign_pdf: Path | None, local_video: Path | None):
    job = jobs[job_id]
    cmd = [str(VENV_PYTHON), "-u", str(CLIP_PY)]
    cmd += [url] if url else ["--local-video", str(local_video)]
    cmd += ["--whisper-model", "base", "--vertical", "--num-clips", str(NUM_CLIPS)]
    if campaign_pdf:
        cmd += ["--campaign-pdf", str(campaign_pdf)]
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    stage = "starting"
    completed_clips = set()  # clips render concurrently now, so track by which
                              # finished (via their "=== Clip N/M rendered ===" marker), not by strict order
    collecting_files = False
    files = []
    captions = []  # aligned with `files` - the line right after each file path, if any
    rankings = {}  # clip number (1-based, matches `files` order) -> estimated hook-strength %

    def set_progress(pct, stage_text):
        with jobs_lock:
            job["progress"] = round(pct, 1)
            job["stage"] = stage_text

    set_progress(1, "Starting...")

    for chunk in _iter_progress_chunks(proc.stdout):
        with jobs_lock:
            job["log_tail"].append(chunk)
            job["log_tail"] = job["log_tail"][-50:]

        if "Downloading video" in chunk:
            stage = "download"
            set_progress(DOWNLOAD_RANGE[0], "Downloading video...")
        elif "Using local video file" in chunk:
            stage = "download"
            set_progress(DOWNLOAD_RANGE[1], "Using your uploaded video...")
        elif m := DOWNLOAD_PCT_RE.search(chunk):
            if stage == "download":
                set_progress(_scale(float(m.group(1)), *DOWNLOAD_RANGE), "Downloading video...")
        elif "Transcribing" in chunk:
            stage = "transcribe"
            set_progress(TRANSCRIBE_RANGE[0], "Transcribing audio...")
        elif m := WHISPER_PCT_RE.match(chunk):
            if stage == "transcribe":
                set_progress(_scale(float(m.group(1)), *TRANSCRIBE_RANGE), "Transcribing audio...")
        elif "Finding top" in chunk:
            stage = "highlights"
            set_progress(HIGHLIGHT_RANGE[0], "Finding the best highlights...")
        elif m := RANKING_LINE_RE.match(chunk):
            rankings[int(m.group(1))] = int(m.group(2))
        elif "Rendering" in chunk and "clip(s)" in chunk:
            stage = "clips"
            set_progress(CLIPS_RANGE[0], f"Rendering clips (0/{NUM_CLIPS})...")
        elif m := CLIP_DONE_RE.search(chunk):
            stage = "clips"
            completed_clips.add(int(m.group(1)))
            total = int(m.group(2))
            pct = _scale((len(completed_clips) / total) * 100, *CLIPS_RANGE)
            set_progress(pct, f"Rendering clips ({len(completed_clips)}/{total})...")
        elif DONE_RE.match(chunk):
            collecting_files = True
        elif collecting_files and (m := FILE_LINE_RE.match(chunk)):
            files.append(Path(m.group(1)).name)
            captions.append("")
        elif collecting_files and (m := CAPTION_LINE_RE.match(chunk)) and captions:
            captions[-1] = m.group(1)

    proc.wait()

    campaign = None
    for f in OUTPUT_DIR.glob("*_campaign.json"):
        try:
            campaign = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
        break

    with jobs_lock:
        if proc.returncode == 0 and files:
            job["status"] = "done"
            job["progress"] = 100
            job["stage"] = "Done"
            job["files"] = files
            job["campaign"] = campaign
            # rankings/captions are keyed by clip order, matching `files`
            job["rankings"] = [rankings.get(i + 1) for i in range(len(files))]
            job["captions"] = captions if len(captions) == len(files) else [""] * len(files)
            job["chat"] = {name: [] for name in files}  # per-clip chatbox history
        else:
            job["status"] = "error"
            job["error"] = "".join(job["log_tail"][-15:])


def _clear_dir(d: Path):
    if not d.exists():
        return
    for f in d.iterdir():
        if f.is_file():
            f.unlink(missing_ok=True)
        elif f.is_dir():
            for sub in f.rglob("*"):
                if sub.is_file():
                    sub.unlink(missing_ok=True)
            f.rmdir()


@app.post("/api/generate")
def api_generate():
    global active_job_id

    if request.content_type and "multipart/form-data" in request.content_type:
        url = (request.form.get("url") or "").strip()
        pdf_file = request.files.get("campaign_pdf")
        video_file = request.files.get("video_file")
    else:
        data = request.get_json(force=True) or {}
        url = (data.get("url") or "").strip()
        pdf_file = None
        video_file = None

    has_video_file = bool(video_file and video_file.filename)
    if not url and not has_video_file:
        return jsonify({"error": "Provide a YouTube url or upload a video file"}), 400
    if url and has_video_file:
        return jsonify({"error": "Provide only one of a YouTube url or an uploaded video file"}), 400
    if pdf_file and pdf_file.filename and not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Campaign brief must be a PDF"}), 400
    if has_video_file and Path(video_file.filename).suffix.lower() not in _VIDEO_EXTS:
        return jsonify({"error": f"Video file must be one of: {', '.join(sorted(_VIDEO_EXTS))}"}), 400

    with jobs_lock:
        if active_job_id and jobs.get(active_job_id, {}).get("status") == "running":
            return jsonify({"error": "A job is already running - wait for it to finish first"}), 409

        # One video at a time: clear out the previous video's downloads,
        # generated clips, and uploaded campaign PDF so disk space doesn't
        # pile up across videos.
        _clear_dir(DOWNLOAD_DIR)
        _clear_dir(OUTPUT_DIR)
        _clear_dir(CAMPAIGNS_DIR)

        job_id = uuid.uuid4().hex[:12]
        active_job_id = job_id

        campaign_pdf_path = None
        if pdf_file and pdf_file.filename:
            CAMPAIGNS_DIR.mkdir(exist_ok=True)
            campaign_pdf_path = CAMPAIGNS_DIR / f"{job_id}.pdf"
            pdf_file.save(str(campaign_pdf_path))

        local_video_path = None
        if has_video_file:
            DOWNLOAD_DIR.mkdir(exist_ok=True)
            ext = Path(video_file.filename).suffix.lower()
            local_video_path = DOWNLOAD_DIR / f"{job_id}{ext}"
            video_file.save(str(local_video_path))

        jobs[job_id] = {
            "status": "running", "progress": 0, "stage": "Queued",
            "files": [], "rankings": [], "captions": [], "campaign": None, "chat": {},
            "error": None, "log_tail": [], "started_at": time.time(),
        }
    threading.Thread(
        target=_run_job, args=(job_id, url or None, campaign_pdf_path, local_video_path), daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.get("/api/status/<job_id>")
def api_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        elapsed = time.time() - job["started_at"]
        return jsonify({
            "status": job["status"], "progress": job["progress"], "stage": job["stage"],
            "files": job["files"], "rankings": job.get("rankings", []),
            "captions": job.get("captions", []), "campaign": job.get("campaign"),
            "error": job["error"], "elapsed": round(elapsed, 1),
        })


def _revise_caption(campaign: dict | None, current_caption: str, history: list, message: str) -> str:
    """Reuses clip.py's own Ollama helpers so the chatbox's tone/hashtag
    rules stay identical to the caption clip.py drafted in the first
    place. Returns the caption UNCHANGED (never raises) if Ollama isn't
    reachable, so a chat request just becomes a no-op rather than an error."""
    if not clip._ensure_ollama_ready():
        return current_caption

    if campaign:
        rules_text = (
            f"Brand/campaign: {campaign.get('brand', '')}\n"
            "Required hashtag(s) - MUST appear verbatim in the caption: "
            f"{', '.join(campaign.get('required_hashtags') or []) or '(none)'}\n"
            f"Tone: {campaign.get('tone') or 'engaging, casual, like a meme/entertainment page'}\n"
            f"Avoid: {', '.join(campaign.get('avoid') or []) or '(nothing specific)'}\n"
        )
    else:
        rules_text = "Tone: short, punchy, engaging TikTok caption.\n"

    convo = "\n".join(f"{h['role']}: {h['text']}" for h in history[-6:])
    prompt = (
        f"{rules_text}\n"
        f'Current caption: "{current_caption}"\n\n'
        f"{'Conversation so far:' + chr(10) + convo + chr(10) + chr(10) if convo else ''}"
        f'The user now says: "{message}"\n\n'
        "Revise the caption to satisfy their request. Keep it under 150 characters. If a required "
        "hashtag is listed above, it MUST still appear verbatim. Reply with ONLY the new caption text, "
        "nothing else - no quotes, no explanation."
    )
    reply = clip._ollama_chat(prompt, timeout=60.0)
    if not reply:
        return current_caption
    caption = reply.strip().strip('"').strip()
    if campaign:
        for tag in campaign.get("required_hashtags") or []:
            if tag and tag.lower() not in caption.lower():
                caption = f"{caption} {tag}"
    return caption or current_caption


@app.get("/api/caption/<job_id>/<filename>")
def api_get_caption(job_id, filename):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or filename not in job.get("files", []):
            abort(404)
        idx = job["files"].index(filename)
        captions = job.get("captions", [])
        caption = captions[idx] if idx < len(captions) else ""
        history = job.get("chat", {}).get(filename, [])
        return jsonify({"caption": caption, "history": history})


@app.post("/api/caption/<job_id>/<filename>")
def api_revise_caption_route(job_id, filename):
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Missing message"}), 400

    with jobs_lock:
        job = jobs.get(job_id)
        if not job or filename not in job.get("files", []):
            abort(404)
        idx = job["files"].index(filename)
        captions = job.get("captions", [])
        current_caption = captions[idx] if idx < len(captions) else ""
        campaign = job.get("campaign")
        history = job.setdefault("chat", {}).setdefault(filename, [])
        history_snapshot = list(history)

    try:
        new_caption = _revise_caption(campaign, current_caption, history_snapshot, message)
    except Exception as e:
        return jsonify({"error": f"Caption revision failed: {e}"}), 500

    with jobs_lock:
        if idx < len(job.get("captions", [])):
            job["captions"][idx] = new_caption
        history.append({"role": "user", "text": message})
        history.append({"role": "assistant", "text": new_caption})
        history_out = list(history)

    return jsonify({"caption": new_caption, "history": history_out})


@app.get("/api/download/<job_id>/<filename>")
def api_download(job_id, filename):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or filename not in job.get("files", []):
            abort(404)
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.get("/api/media/<job_id>/<filename>")
def api_media(job_id, filename):
    """Same file as /api/download but served inline (not as an attachment),
    so it can be used directly as a <video> element's src for in-page preview."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or filename not in job.get("files", []):
            abort(404)
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False, conditional=True)


@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
