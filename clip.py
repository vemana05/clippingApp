#!/usr/bin/env python3
"""
Download a YouTube video, transcribe it, auto-detect short (30-45s) highlight
clips, and burn in plain TikTok-style word-by-word captions.

Usage:
    ./venv/bin/python clip.py "https://youtube.com/watch?v=..." \
        --min-duration 30 --max-duration 45 --whisper-model small --vertical

Pipeline:
    1. Download video with yt-dlp (optionally its top comments too)
    2. Transcribe with Whisper (word-level timestamps)
    3. Score transcript windows for "highlight-worthiness" (audio energy,
       transcript keywords, an optional local-LLM funniness read, and
       optional comment-timestamp mentions) and pick the best, most
       concentrated one(s), each sized to fit its own engaging moment
    4. Cut that window out of the video with ffmpeg (optional face-tracking
       vertical crop) with a short attention-grabbing "hook" teaser first
    5. Render plain rolling word-by-word subtitles as PNG frames with
       Pillow and composite them onto the clip
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# ffmpeg setup: use the binary bundled by imageio-ffmpeg so this works with
# no system-wide install (no Homebrew/sudo required). imageio-ffmpeg's binary
# isn't literally named "ffmpeg" (e.g. "ffmpeg-macos-aarch64-v7.1"), but
# Whisper's internal audio loader hardcodes the command name "ffmpeg" - so we
# symlink it into a local bin dir and put that on PATH.
# ---------------------------------------------------------------------------
WORK_DIR = Path(__file__).parent
DOWNLOAD_DIR = WORK_DIR / "downloads"
OUTPUT_DIR = WORK_DIR / "output"

_bin_dir = WORK_DIR / ".bin"
_bin_dir.mkdir(exist_ok=True)
FFMPEG_BIN = str(_bin_dir / "ffmpeg")
if not os.path.exists(FFMPEG_BIN):
    os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), FFMPEG_BIN)
os.environ["PATH"] = str(_bin_dir) + os.pathsep + os.environ.get("PATH", "")

# Local, free LLM for scoring transcript content (funniest/weirdest moments -
# see llm_funniness_scores below): a small model served by Ollama's CLI
# binary (bundled in .ollama/, no system-wide install / no account needed).
OLLAMA_BIN = str(WORK_DIR / ".ollama" / "ollama")
OLLAMA_MODELS_DIR = str(WORK_DIR / ".ollama" / "models")
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"

EXCITEMENT_WORDS = {
    "crazy", "insane", "amazing", "incredible", "unbelievable", "wow",
    "wild", "huge", "massive", "best", "worst", "never", "literally",
    "secret", "shocking", "seriously", "no way", "what the", "oh my god",
    "omg", "actually", "honestly", "ridiculous", "genius", "brutal",
    "terrifying", "hilarious", "insanely", "perfect", "epic",
}

# Caption rendering: plain, standard-size text - no emoji, no per-word
# color/caps/punctuation flourishes.
TEXT_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
_FONT_CACHE: dict = {}


def _get_text_font(size: int):
    key = ("text", size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(TEXT_FONT_PATH, size)
    return _FONT_CACHE[key]


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
# YouTube's web client currently blocks yt-dlp's default extraction
# ("The page needs to be reloaded"), and blocks web/android from getting
# anything above 360p without a PO token. The android_vr client isn't
# subject to either restriction, so it's what actually gets us real
# resolution - but it's an unofficial workaround, and YouTube's anti-bot
# system can and does start blocking it mid-session, so it's tried first
# and NOT relied on: RELIABLE_ARGS (360p, has held up all session) is the
# guaranteed fallback if the high-res attempt fails.
#
# The high-res path is the real, current fix (not a client-spoofing trick):
# a PO Token minted through an actual Chrome browser session (the
# yt-dlp-getpot-wpc plugin - it briefly opens a visible Chrome window, do
# not close it) for the "mweb" client, plus YouTube's separate "n" signature
# challenge solved via a small official yt-dlp script run through the
# bundled Deno runtime (.bin/deno). Verified end-to-end: real 1080p60 file,
# full duration, clean merge. Requires yt-dlp-getpot-wpc to be installed and
# Chrome to be present - both true in this project's venv.
_HIGH_RES_ARGS = {"youtube": {"player_client": ["mweb"]}}
_RELIABLE_ARGS = {"youtube": {"player_client": ["android", "web"]}}


def _try_download(url: str, extractor_args: dict, max_height: int, remote_components=None,
                   get_comments: bool = False):
    import yt_dlp

    outtmpl = str(DOWNLOAD_DIR / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "ffmpeg_location": os.path.dirname(FFMPEG_BIN),
        "quiet": False,
        "noplaylist": True,
        "extractor_args": extractor_args,
    }
    if remote_components:
        ydl_opts["remote_components"] = remote_components
    if get_comments:
        # Top-level comments only, sorted by likes, capped so this doesn't
        # turn into an unbounded fetch on a video with tens of thousands of
        # comments - a few hundred of the most-liked ones is plenty of
        # signal for "which moments did viewers actually react to".
        ydl_opts["getcomments"] = True
        merged_args = dict(extractor_args)
        merged_args["youtube"] = {
            **merged_args.get("youtube", {}),
            "max_comments": ["200"],   # yt-dlp wants each value as its own list item, not one comma string
            "comment_sort": ["top"],
        }
        ydl_opts["extractor_args"] = merged_args
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
        if not path.exists():
            path = path.with_suffix(".mp4")
        return path, info


def download_video(url: str, max_height: int = 1080, get_comments: bool = False):
    """Returns (video_path, info_dict). info_dict includes duration, and,
    when YouTube has generated one for this video, the public 'most replayed'
    audience-engagement heatmap and/or (if get_comments) the video's top
    comments. Tries for max_height first (see _HIGH_RES_ARGS); if that's
    unavailable for any reason, falls back to the guaranteed-working 360p
    route rather than failing outright."""
    from yt_dlp.utils import DownloadError

    DOWNLOAD_DIR.mkdir(exist_ok=True)

    if max_height > 360:
        try:
            print("Getting a real PO Token via Chrome - a browser window will briefly open, leave it be...")
            return _try_download(url, _HIGH_RES_ARGS, max_height, remote_components=["ejs:github"],
                                  get_comments=get_comments)
        except DownloadError as e:
            print(f"High-res download failed ({e}) - falling back to 360p")

    return _try_download(url, _RELIABLE_ARGS, 360, get_comments=get_comments)


def probe_duration(video_path: Path) -> float:
    """Reads a video's duration straight from ffmpeg's own probe output
    (the header it prints before erroring out with no output specified) -
    fast, no decoding, works on any local file regardless of source."""
    proc = subprocess.run(
        [FFMPEG_BIN, "-i", str(video_path)],
        stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not m:
        return 0.0
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def load_local_video(video_path: Path) -> tuple:
    """Returns (video_path, info_dict) for an already-local video file, in
    the same shape download_video returns - so the rest of the pipeline
    (transcribe, highlight-finding, etc.) doesn't need to know or care
    whether the source came from YouTube or a file the user uploaded. No
    heatmap and no comments are possible here (there's no YouTube page to
    pull them from), so those two signals just sit out for this run - the
    other signals (audio, transcript, LLM funniness) are unaffected."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    info = {"duration": probe_duration(video_path), "title": video_path.stem, "heatmap": None}
    return video_path, info


# ---------------------------------------------------------------------------
# 2. Transcribe
# ---------------------------------------------------------------------------
def transcribe(video_path: Path, model_size: str = "small"):
    import whisper

    print(f"Loading whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    print("Transcribing (this can take a while)...")
    result = model.transcribe(str(video_path), word_timestamps=True, verbose=False)

    words = []
    for segment in result["segments"]:  # type: ignore[index]
        for w in segment.get("words", []):  # type: ignore[union-attr]
            words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })
    if not words:
        raise RuntimeError("Whisper returned no word-level timestamps.")
    return words


# ---------------------------------------------------------------------------
# 3. Highlight detection
# ---------------------------------------------------------------------------
def score_word(word: str) -> float:
    w = word.lower().strip(".,!?\"'")
    score = 0.0
    if w in EXCITEMENT_WORDS:
        score += 3.0
    if word.endswith("!"):
        score += 1.5
    if word.isupper() and len(word) > 2:
        score += 1.0
    return score


def heatmap_scores(heatmap: list, n_seconds: int) -> np.ndarray:
    """YouTube's public 'most replayed' audience-engagement heatmap (the graph
    under the seek bar), when yt-dlp is able to extract one for this video.
    This is the real public signal for 'where users are most engaged' -
    the official YouTube Analytics API only works for videos on channels
    you own, so this is what's available for arbitrary videos."""
    scores = np.zeros(n_seconds)
    for bucket in heatmap:
        start = max(0, int(bucket["start_time"]))
        end = min(n_seconds, int(bucket["end_time"]))
        scores[start:end] = bucket["value"]
    return scores


def audio_energy_scores(video_path: Path, n_seconds: int, sr: int = 4000) -> np.ndarray:
    """Per-second loudness envelope (RMS) of the audio track - the 'loudest /
    most energetic' signal (reactions, shouting, music stings, laughter)."""
    cmd = [FFMPEG_BIN, "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)

    total_samples = n_seconds * sr
    if len(audio) < total_samples:
        audio = np.pad(audio, (0, total_samples - len(audio)))
    else:
        audio = audio[:total_samples]

    rms = np.sqrt(np.mean(audio.reshape(n_seconds, sr) ** 2, axis=1))
    peak = np.percentile(rms, 99)
    return np.clip(rms / peak, 0, 1) if peak > 0 else rms


def transcript_scores(words: list, n_seconds: int):
    """Per-second excitement-keyword score and speech density (words/sec)."""
    keyword = np.zeros(n_seconds)
    density = np.zeros(n_seconds)
    for w in words:
        idx = int(w["start"])
        if 0 <= idx < n_seconds:
            keyword[idx] += score_word(w["word"])
            density[idx] += 1

    def normalize(arr):
        peak = arr.max()
        return arr / peak if peak > 0 else arr

    return normalize(keyword), normalize(density)


_TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{1,2}):([0-5]\d)(?::([0-5]\d))?(?!\d)")


def comment_timestamp_scores(info: dict, n_seconds: int) -> np.ndarray:
    """Real audience signal: parses the video's own YouTube comments (see
    download_video's get_comments) for timestamp references people actually
    left - "2:15 lol", "omg at 1:30", "1:02:40 is unreal" - and turns
    like-weighted mention density into a per-second score. This is what
    viewers themselves flagged as worth going back to, not a heuristic
    guessing at it from the transcript."""
    scores = np.zeros(n_seconds)
    comments = info.get("comments") or []
    if not comments:
        return scores

    found = 0
    for c in comments:
        text = c.get("text") or ""
        likes = max(0, c.get("like_count") or 0)
        weight = 1.0 + math.log1p(likes)
        for m in _TIMESTAMP_RE.finditer(text):
            a, b, c_ = m.group(1), m.group(2), m.group(3)
            t = (int(a) * 3600 + int(b) * 60 + int(c_)) if c_ is not None else (int(a) * 60 + int(b))
            if 0 <= t < n_seconds:
                lo, hi = max(0, t - 3), min(n_seconds, t + 4)
                scores[lo:hi] += weight
                found += 1

    if found:
        print(f"Found {found} timestamp mention(s) across {len(comments)} comment(s)")
        peak = scores.max()
        if peak > 0:
            scores = scores / peak
    return scores


def _ollama_available() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=2)
        return True
    except Exception:
        return False


def _ensure_ollama_ready() -> bool:
    """Starts the local Ollama server (bundled in .ollama/, no system-wide
    install) if it isn't already running, and pulls its small model if it
    isn't cached yet. Returns False if Ollama isn't set up in this project
    at all, so the caller can skip the LLM-based signal gracefully."""
    if not os.path.exists(OLLAMA_BIN):
        return False

    if not _ollama_available():
        print("Starting local Ollama server...")
        env = {**os.environ, "OLLAMA_MODELS": OLLAMA_MODELS_DIR}
        subprocess.Popen(
            [OLLAMA_BIN, "serve"], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(30):
            if _ollama_available():
                break
            time.sleep(1)
        else:
            print("Local Ollama server didn't come up in time - skipping funniness scoring")
            return False

    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read())
        have_model = any(m.get("model") == OLLAMA_MODEL for m in tags.get("models", []))
    except Exception:
        have_model = False

    if not have_model:
        print(f"Pulling {OLLAMA_MODEL} model (~2GB, one-time download)...")
        env = {**os.environ, "OLLAMA_MODELS": OLLAMA_MODELS_DIR}
        subprocess.run([OLLAMA_BIN, "pull", OLLAMA_MODEL], env=env, check=True)

    return True


def _ollama_chat(prompt: str, timeout: float = 120.0, max_tokens: int = 700,
                  temperature: float | None = None) -> str | None:
    # num_predict must be set explicitly - Ollama's own default caps output
    # around ~128 tokens, which silently truncates anything longer (a JSON
    # rules object, a caption plus reasoning) with no error, just a cut-off
    # reply that then fails to parse. temperature defaults to the model's
    # own (creative) default - callers that need reliably well-formed
    # structured output (JSON) should pass something low like 0.2.
    options: dict = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat", data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data.get("message", {}).get("content")


def _build_timestamped_transcript(words: list, max_chars: int = 60000) -> str:
    """Collapses word-level timestamps into ~6-second-chunk lines like
    "[12s] some words here" - compact enough for an LLM prompt while
    keeping enough timing resolution to locate a moment within a few
    seconds. Shared by every LLM signal that reads the full transcript."""
    lines = []
    chunk_words, chunk_start = [], words[0]["start"]
    for w in words:
        if w["start"] - chunk_start > 6 and chunk_words:
            lines.append(f"[{int(chunk_start)}s] {' '.join(chunk_words)}")
            chunk_words, chunk_start = [], w["start"]
        chunk_words.append(w["word"])
    if chunk_words:
        lines.append(f"[{int(chunk_start)}s] {' '.join(chunk_words)}")
    return "\n".join(lines)[:max_chars]  # defensive cap even though the model's context is huge


_TIME_TOKEN_RE = r"\[?\s*(?:(\d+)\s*m)?\s*(\d+(?:\.\d+)?)\s*s?\s*\]?"


def _parse_time_token(minutes_str, seconds_str) -> float:
    minutes = float(minutes_str) if minutes_str else 0.0
    return minutes * 60 + float(seconds_str)


def llm_funniness_scores(words: list, n_seconds: int, video_title: str = "") -> np.ndarray:
    """Asks the local LLM to read the actual transcript and flag the
    funniest/weirdest/most entertaining moments - a genuine content-
    understanding signal that catches things pure audio-loudness misses
    (a deadpan line, an absurd claim, a twist), not just "loud". Returns an
    all-zero array (safe no-op) if the local LLM isn't available."""
    scores = np.zeros(n_seconds)
    if not words or not _ensure_ollama_ready():
        if not words:
            return scores
        print("Local LLM not available - skipping funniness scoring signal")
        return scores

    transcript = _build_timestamped_transcript(words)

    prompt = (
        f'Video title: "{video_title}"\n\n'
        f"Timestamped transcript:\n{transcript}\n\n"
        "Find the funniest, weirdest, or most surprising/entertaining moments "
        "in this transcript - the kind of moment that would make a great "
        "TikTok clip hook. For each one, reply on its own line as:\n"
        "SECONDS: SCORE\n"
        "where SECONDS is the timestamp (just the number) and SCORE is 0-10 "
        "for how funny/entertaining it is. List up to 15 moments, most "
        "entertaining first. Reply with ONLY those lines, nothing else."
    )

    print("Asking the local LLM to find the funniest/weirdest moments...")
    try:
        reply = _ollama_chat(prompt)
    except Exception as e:
        print(f"LLM funniness scoring failed ({e}) - skipping that signal")
        return scores
    if not reply:
        return scores

    found = 0
    for line in reply.splitlines():
        # Tolerant of however the model actually formats it: "12: 8",
        # "[12s] 8", "12s - 8", "[1m15s] 10", etc. - it doesn't reliably
        # follow the exact "SECONDS: SCORE" format asked for in the prompt,
        # and switches to minutes+seconds for anything past ~a minute.
        m = re.match(r"\s*\[?\s*(?:(\d+)\s*m)?\s*(\d+(?:\.\d+)?)\s*s?\s*\]?\s*[:,\-]?\s*(\d+(?:\.\d+)?)", line)
        if not m:
            continue
        minutes = float(m.group(1)) if m.group(1) else 0.0
        t = minutes * 60 + float(m.group(2))
        score = min(10.0, float(m.group(3)))
        idx = int(t)
        if 0 <= idx < n_seconds:
            lo, hi = max(0, idx - 3), min(n_seconds, idx + 4)
            scores[lo:hi] = np.maximum(scores[lo:hi], score / 10.0)
            found += 1
    print(f"LLM flagged {found} candidate funny/weird moment(s)")
    return scores


def find_curiosity_gaps(words: list, n_seconds: int, video_title: str = "") -> list:
    """Asks the local LLM to find "curiosity gap" pairs in the transcript:
    a moment that poses a question, teases a claim, or builds real tension
    (a HOOK - something that makes a viewer need to know what happens
    next), and the LATER moment where that actually gets answered,
    resolved, or revealed (the PAYOFF). This is the classic hook/story/
    payoff structure - open on the question, hold it open through the
    middle, land the answer near the end as the reward for watching the
    whole thing. Returns a list of (hook_seconds, payoff_seconds) tuples,
    strongest gap first - empty (safe no-op) if the local LLM isn't
    available or finds nothing usable."""
    pairs: list = []
    if not words or not _ensure_ollama_ready():
        return pairs

    transcript = _build_timestamped_transcript(words)
    prompt = (
        f'Video title: "{video_title}"\n\n'
        f"Timestamped transcript:\n{transcript}\n\n"
        'Find "curiosity gap" moments: a point where a QUESTION is asked, a claim or mystery is teased, '
        "or real tension/suspense builds - something that makes a viewer need to know what happens next "
        "(the HOOK) - and the LATER point where that actually gets answered, resolved, or revealed (the "
        "PAYOFF). Only include a pair if the payoff comes at least 5 seconds after its hook. For each "
        "pair, reply on its own line as:\n"
        "HOOK: <seconds> PAYOFF: <seconds>\n"
        "List up to 10 pairs, strongest curiosity gap first. Reply with ONLY those lines, nothing else."
    )

    print("Asking the local LLM to find hook/payoff (curiosity gap) moments...")
    try:
        reply = _ollama_chat(prompt)
    except Exception as e:
        print(f"Curiosity-gap detection failed ({e}) - skipping that signal")
        return pairs
    if not reply:
        return pairs

    pair_re = re.compile(
        r"HOOK\s*:?\s*" + _TIME_TOKEN_RE + r".*?PAYOFF\s*:?\s*" + _TIME_TOKEN_RE,
        re.IGNORECASE,
    )
    for line in reply.splitlines():
        m = pair_re.search(line)
        if not m:
            continue
        hook_t = _parse_time_token(m.group(1), m.group(2))
        payoff_t = _parse_time_token(m.group(3), m.group(4))
        if 0 <= hook_t < n_seconds and 0 <= payoff_t < n_seconds and payoff_t > hook_t + 5:
            pairs.append((hook_t, payoff_t))

    print(f"Found {len(pairs)} hook/payoff pair(s)")
    return pairs


def _normalize_words_for_match(text: str) -> list:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).split()


# Hedge/discourse filler - words that pad out rambling speech without
# carrying content. A line that's mostly THESE isn't a question or a
# statement with substance, no matter how verbatim the quote match is -
# measured directly: "Honestly, like, I feel my most, you know, expressive"
# matched its source perfectly and still wasn't a usable hook.
_FILLER_WORDS = {
    "honestly", "like", "know", "feel", "kind", "sort", "mean", "um", "uh",
    "well", "basically", "literally", "actually", "just", "really", "very",
    "gonna", "wanna", "okay", "ok", "yeah", "yea", "right", "so", "uhh", "umm",
}


def _is_low_substance(quote: str) -> bool:
    """True if `quote` is too short, too repetitive, or too filler-heavy
    to work as a real hook, even if it's a perfectly verbatim match - a
    scored STATEMENT label from the LLM doesn't guarantee the line
    actually says anything. This is a coarse net, not a real coherence
    judge (catching every vague-but-grammatical line, like "we're going
    to end it, OK", needs actual semantic judgment no word list can do -
    see the score-threshold filter applied alongside this one)."""
    tokens = _normalize_words_for_match(quote)
    if len(tokens) < 6:
        return True
    filler = sum(1 for t in tokens if t in _FILLER_WORDS)
    if (filler / len(tokens)) > 0.3:
        return True
    unique_ratio = len(set(tokens)) / len(tokens)
    return unique_ratio < 0.65  # heavy repetition ("no, no, no, no, that's terrible")


def find_quote_span(words: list, quote: str, approx_time: float, search_radius: float = 25.0):
    """Finds the (start_time, end_time) span in `words` that best matches
    an LLM-supplied quotation, searching near `approx_time` (the model's
    timestamp is often a little off, and its quoting is rarely word-for-
    word). This is what lets a hook be anchored to a COMPLETE, real
    sentence instead of an arbitrary fixed-length time slice that can
    truncate a question or statement mid-thought.

    Uses ORDER-SENSITIVE sequence similarity (difflib), not bag-of-words
    overlap - a set-overlap score gives a rambling stretch of filler
    ("honestly, like, you know, I feel...") a deceptively high score
    against almost any quote just because "honestly"/"like"/"you"/"know"
    show up everywhere, which was measured to actually happen and produce
    a real bad hook. Sequence similarity only scores well when the WORDS
    APPEAR IN THE SAME ORDER as the quote, which filler can't fake.

    Returns None - callers must fall back to something else - if no
    sufficiently strong match is found nearby, rather than trust a weak
    guess."""
    target = _normalize_words_for_match(quote)
    if not target:
        return None

    lo_t, hi_t = approx_time - search_radius, approx_time + search_radius
    candidates = [i for i, w in enumerate(words) if lo_t <= w["start"] <= hi_t]
    if not candidates:
        return None
    lo_i, hi_i = candidates[0], candidates[-1]

    # A matched span must never cross a real pause/turn boundary - without
    # this, the best-overlap search can glue two separate, unrelated
    # reactions from different moments/speakers into one "sentence" that
    # reads as incoherent even though neither half looks bad on its own
    # (measured directly: "That's terrible. No." + "And now... JD's on a
    # list" got stitched into a single matched hook this way).
    MAX_GAP = 1.2

    n = len(target)
    best = None  # (ratio, start_i, end_i)
    for start_i in range(lo_i, hi_i + 1):
        max_end_i = start_i + 1
        while max_end_i <= hi_i and words[max_end_i]["start"] - words[max_end_i - 1]["end"] <= MAX_GAP:
            max_end_i += 1
        for length in range(max(1, n - 3), n + 6):
            end_i = start_i + length
            if end_i > hi_i + 1 or end_i > max_end_i:
                break
            span = [w for word in words[start_i:end_i] for w in _normalize_words_for_match(word["word"])]
            if not span:
                continue
            ratio = difflib.SequenceMatcher(a=target, b=span, autojunk=False).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, start_i, end_i)

    if best is None or best[0] < 0.55:  # require the words to actually appear IN ORDER, not just be nearby
        return None
    _, s_i, e_i = best
    e_i = min(e_i, len(words))
    return words[s_i]["start"], words[e_i - 1]["end"]


def find_hook_lines(words: list, n_seconds: int, video_title: str = "", min_candidates: int = 8) -> list:
    """Asks the local LLM for the actual best OPENING HOOK lines in this
    video: a QUESTION that creates curiosity, or a bold/controversial/
    opinionated STATEMENT that would make someone want to keep watching or
    argue in the comments. Unlike find_curiosity_gaps, this does NOT
    require a later resolution - a strong hot take is a complete hook on
    its own. Each candidate is matched back to its exact words in the
    transcript (see find_quote_span) so the hook can be the complete
    sentence, not a truncated fixed-length fragment. Keeps calling the LLM
    (accumulating candidates across calls) until at least min_candidates
    are found or it's made a generous number of attempts - one call's
    sample can be poor even when the video has plenty of good lines, and
    there need to be enough of them that every highlight in the batch can
    get a real one instead of the batch running out and falling back to a
    lesser signal. Returns a list of dicts {start, end, type, score, text},
    strongest first - empty (safe no-op) if the local LLM isn't available
    or nothing matches well after every attempt."""
    lines_out: list = []
    if not words or not _ensure_ollama_ready():
        return lines_out

    transcript = _build_timestamped_transcript(words)
    prompt = (
        f'Video title: "{video_title}"\n\n'
        f"Timestamped transcript:\n{transcript}\n\n"
        "Find lines that would work as a scroll-stopping OPENING HOOK for a short-form video: either a "
        "QUESTION that creates real curiosity, or a bold/controversial/opinionated STATEMENT that would "
        "make someone want to keep watching or argue about it in the comments. It does NOT need to be "
        "resolved later in the video - a strong opinion or question is a complete hook on its own.\n\n"
        "REJECT a candidate line if it is vague, generic, or filler, even if it sounds dramatic in the "
        'moment. For example, none of these are hooks, and a SCORE of 8+ should never go to something '
        'like them: "Honestly, like, I feel my most, you know, expressive when I\'m..." (rambling filler, '
        'no actual content), "No, no, no, no, that\'s terrible" (just a reaction, says nothing), "We\'re '
        'going to end it, OK" (empty of any real claim or question), "Yes." (not even a sentence). A real '
        "hook makes a SPECIFIC claim or asks a SPECIFIC question a viewer could actually react to or "
        "argue about - it has to say something, not just react to something.\n\n"
        "Prefer a full sentence or clause over a sentence fragment - the hook needs to stand on its own. "
        "COPY the words for TEXT directly from the transcript above, character for character - do not "
        "paraphrase, reword, or clean them up even slightly, or the quote won't be findable in the "
        "transcript. For each one, reply on its own line as:\n"
        "TIME: <seconds> | TYPE: QUESTION or STATEMENT | SCORE: <0-10> | TEXT: <words copied verbatim>\n"
        "List up to 15 candidates, strongest hook first. Reply with ONLY those lines, nothing else."
    )

    def parse_hook_line(line: str):
        # The model reliably gives 4 pipe-separated fields IN ORDER, but
        # unreliably includes the "TIME:"/"TYPE:"/"SCORE:"/"TEXT:" labels
        # themselves - confirmed directly: the exact same prompt, later in
        # the same run, replied with bare "2199s | STATEMENT | 9 | Yes."
        # instead of "TIME: 2199s | TYPE: STATEMENT | ...". A regex that
        # requires the label text matches neither case reliably, so parse
        # positionally by "|" and treat each label as optional.
        line = re.sub(r"^\s*\d+[.)]\s*", "", line)  # strip a numbered-list prefix ("1. ") if present -
                                                      # otherwise its digit gets mistaken for the timestamp
        parts = line.split("|")
        if len(parts) < 4:
            return None
        time_part, type_part, score_part = parts[0], parts[1], parts[2]
        text_part = "|".join(parts[3:])

        time_m = re.search(_TIME_TOKEN_RE, re.sub(r"(?i)^\s*TIME\s*:?", "", time_part))
        type_m = re.search(r"QUESTION|STATEMENT", type_part, re.IGNORECASE)
        score_m = re.search(r"\d+(?:\.\d+)?", re.sub(r"(?i)^\s*SCORE\s*:?", "", score_part))
        if not (time_m and type_m and score_m):
            return None

        t = _parse_time_token(time_m.group(1), time_m.group(2))
        kind = type_m.group(0).upper()
        score = min(10.0, float(score_m.group(0)))
        quote = re.sub(r"(?i)^\s*TEXT\s*:?", "", text_part).strip().strip('"').strip()
        return t, kind, score, quote

    # The LLM's OWN sample quality swings hard between calls on identical
    # input - measured going from 7 candidates, to 0, to 10 well-matched
    # ones on the exact same transcript across three back-to-back calls.
    # That's sampling variance, not a bug in the matching itself - so keep
    # calling and accumulating until there's a real supply, not just one.
    print("Asking the local LLM to find question/controversial-statement hook lines...")
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            reply = _ollama_chat(prompt)
        except Exception as e:
            print(f"Hook-line detection failed ({e}) - skipping that signal")
            break
        if not reply:
            continue

        found_this_attempt = 0
        for line in reply.splitlines():
            parsed = parse_hook_line(line)
            if parsed is None:
                continue
            t, kind, score, quote = parsed
            if not (0 <= t < n_seconds) or not quote:
                continue
            if score < 7.0:
                continue  # only the model's own confident picks - low scores skew toward filler/reactions
            if _is_low_substance(quote):
                continue  # too short/repetitive/filler-heavy to be an actual question or claim
            span = find_quote_span(words, quote, t)
            if span is None:
                continue
            start, end = span
            if end - start < 1.0:
                continue  # a single word/fragment isn't a usable standalone hook
            if any(abs(start - hl["start"]) < 1.5 for hl in lines_out):
                continue  # already have a candidate essentially at this same moment
            lines_out.append({"start": start, "end": end, "type": kind, "score": score, "text": quote})
            found_this_attempt += 1

        if len(lines_out) >= min_candidates:
            break
        print(f"  (attempt {attempt + 1}/{max_attempts} matched {found_this_attempt}, "
              f"{len(lines_out)} total so far - "
              f"{'retrying for more' if attempt < max_attempts - 1 else 'stopping'})")

    lines_out.sort(key=lambda h: -h["score"])
    print(f"Found {len(lines_out)} hook line candidate(s) (question/controversial statement)")
    return lines_out


# ---------------------------------------------------------------------------
# Campaign briefs (monetized "clip farm" style deals): read a brand's rules
# PDF and turn it into constraints/guidance this pipeline can actually act
# on - a required hashtag, a required on-screen segment every clip needs,
# a caption tone, and things to avoid.
# ---------------------------------------------------------------------------
def _close_unclosed_json(text: str) -> str:
    """Appends whatever closing quote/bracket/brace characters are needed
    to make `text` structurally balanced, by walking it and tracking what's
    still open. Doesn't validate VALUES, just syntax - a cheap fix for a
    reply that was cut off before finishing, not a general JSON repair."""
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    closer = {"{": "}", "[": "]"}
    suffix = ('"' if in_string else "") + "".join(closer[c] for c in reversed(stack))
    return text + suffix


def _parse_json_object_lenient(reply: str) -> dict | None:
    """Finds and parses the JSON object in an LLM reply, tolerating the
    common small-model failure of stopping generation just before the
    final closing brace(s). Tries, in order: the text trimmed to its last
    literal '}' (handles trailing commentary after otherwise-valid JSON),
    the raw text as-is, and the raw text with missing closers appended."""
    start = reply.find("{")
    if start == -1:
        return None
    raw = reply[start:]

    candidates = []
    end = raw.rfind("}")
    if end != -1:
        candidates.append(raw[:end + 1])
    candidates.append(raw)
    candidates.append(_close_unclosed_json(raw))

    for text in candidates:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_campaign_rules(pdf_text: str) -> dict | None:
    """Asks the local LLM to read a campaign brief (brand-deal / ClipFarm-
    style PDF, already extracted to plain text) and pull out the parts
    this pipeline can actually act on. Returns None - never raises - if
    Ollama isn't available or the reply can't be parsed as JSON, so a bad
    PDF or a flaky LLM reply just means "run unconstrained", not a crash."""
    if not pdf_text.strip() or not _ensure_ollama_ready():
        return None

    prompt = (
        "Below is a brand/creator campaign brief (a PDF's extracted text). "
        "Read it and extract ONLY the following as JSON, nothing else - no "
        "markdown fences, no commentary, just a single JSON object:\n\n"
        "{\n"
        '  "brand": "<brand or campaign name, short>",\n'
        '  "required_hashtags": ["<hashtag(s) that MUST appear in the caption, include the # symbol>"],\n'
        '  "required_keywords": ["<word(s)/short phrase(s) that must be SAID OR CLEARLY VISIBLE in every '
        'clip - the product name, brand name, or the described visual moment, e.g. \\"Goli\\", \\"gummies\\". '
        'Keep each one short (1-3 words) since these get matched against a spoken transcript.>"],\n'
        '  "tone": "<one sentence describing how the caption should be written/sound>",\n'
        '  "avoid": ["<short phrases describing things that get a clip rejected or are forbidden>"],\n'
        '  "summary": "<2-3 sentence plain-English summary of the whole campaign>"\n'
        "}\n\n"
        'If the brief does not specify one of these fields, use "" or [] for it. Reply with ONLY the JSON '
        "object and nothing else - no text before it, no text after it, and make sure every brace, "
        "bracket, and quote is properly closed.\n\n"
        f"CAMPAIGN BRIEF:\n{pdf_text[:20000]}"
    )

    print("Reading campaign brief with the local LLM...")
    # A 3B model consistently stops right after writing the "summary" field
    # on this kind of long structured-JSON reply, before emitting the final
    # closing brace - a real attention/tracking limitation, reproducible
    # regardless of temperature, not sampling noise. A strict "must contain
    # a literal '}'" regex match fails outright against that, so this tries
    # progressively more forgiving parses - trimmed to the last real '}' in
    # the reply, then as-is, then with the missing closing brackets/braces
    # appended - before giving up and retrying the whole call.
    rules = None
    for attempt in range(5):
        try:
            reply = _ollama_chat(prompt)
        except Exception as e:
            print(f"Campaign rule extraction failed ({e}) - continuing without campaign rules")
            return None
        candidate = _parse_json_object_lenient(reply) if reply else None
        if isinstance(candidate, dict):
            rules = candidate
            break
        print(f"  (attempt {attempt + 1}/5 didn't parse cleanly, retrying...)")

    if rules is None:
        print("Couldn't parse campaign rules from the LLM's reply after retrying - continuing without "
              "campaign rules")
        return None

    rules.setdefault("brand", "")
    rules.setdefault("required_hashtags", [])
    rules.setdefault("required_keywords", [])
    rules.setdefault("tone", "")
    rules.setdefault("avoid", [])
    rules.setdefault("summary", "")

    print(f"Campaign: {rules.get('brand') or '(unnamed)'}")
    if rules["required_hashtags"]:
        print(f"  Required hashtag(s): {', '.join(rules['required_hashtags'])}")
    if rules["required_keywords"]:
        print(f"  Required segment/keyword(s): {', '.join(rules['required_keywords'])}")
    if rules["tone"]:
        print(f"  Tone: {rules['tone']}")

    return rules


def generate_caption(words: list, campaign: dict | None, video_title: str = "") -> str:
    """Drafts a caption for one clip with the local LLM: matches the
    campaign's required hashtag(s) and tone when campaign rules are given,
    otherwise just writes a plain engaging caption from the clip's own
    content. Falls back to a plain (never-empty) caption built straight
    from the transcript if the LLM isn't available - never crashes the
    pipeline over a caption."""
    text = " ".join(w["word"] for w in words).strip()
    fallback = (text[:80] + "...") if len(text) > 80 else (text or video_title or "check this out")
    if campaign:
        tags = " ".join(campaign.get("required_hashtags") or [])
        fallback = f"{fallback} {tags}".strip()

    if not _ensure_ollama_ready():
        return fallback

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

    prompt = (
        f"{rules_text}\n"
        f'This clip\'s spoken content: "{text[:1500]}"\n\n'
        "Write ONE short TikTok/Reels caption for this clip (under 150 characters). Follow the tone "
        "exactly. If a required hashtag is listed above, it MUST appear verbatim in your caption. Reply "
        "with ONLY the caption text, nothing else - no quotes, no explanation."
    )
    try:
        reply = _ollama_chat(prompt, timeout=60.0)
    except Exception:
        return fallback
    if not reply:
        return fallback

    caption = reply.strip().strip('"').strip()
    if campaign:
        # Safety net: if the model dropped a required hashtag, append it
        # rather than silently shipping a non-compliant caption - a missing
        # hashtag is an automatic rejection in these campaigns.
        for tag in campaign.get("required_hashtags") or []:
            if tag and tag.lower() not in caption.lower():
                caption = f"{caption} {tag}"
    return caption or fallback


def _snap_to_words(words: list, best_start: float, best_end: float):
    """Snap a raw [start, end) second-range to the surrounding words so the
    cut doesn't land mid-word."""
    candidates = [w for w in words if w["start"] < best_end and w["end"] > best_start]
    if candidates:
        start_time = max(0.0, candidates[0]["start"] - 0.3)
        end_time = candidates[-1]["end"] + 0.3
    else:
        start_time, end_time = float(best_start), float(best_end)
    highlight_words = [w for w in words if w["start"] >= start_time and w["end"] <= end_time]
    return start_time, end_time, highlight_words


_CURIOSITY_BONUS = 1.25  # a real detected hook->payoff arc beats a generically loud/funny window of similar quality


def find_top_highlights(video_path: Path, words: list, info: dict,
                         min_duration: float = 15.0, max_duration: float = 45.0, num_highlights: int = 1,
                         funniness: np.ndarray | None = None, comment_scores: np.ndarray | None = None,
                         required_keywords: list | None = None, curiosity_pairs: list | None = None):
    """Combine YouTube's audience-engagement heatmap (if available), audio
    loudness, transcript excitement/speech-density, an optional LLM-based
    funniness signal, and an optional comment-timestamp signal (real viewer
    reactions - see comment_timestamp_scores) into one per-second score,
    then greedily pick the `num_highlights` best-scoring, non-overlapping
    windows across the whole video. Each pick's own LENGTH is chosen too
    (anywhere from min_duration to max_duration), not just its position -
    for each candidate length, find that length's best-scoring position,
    then take whichever (length, position) combo has the highest average
    score density. A short, tightly-concentrated peak beats a longer window
    that's diluted by weaker padding, so this naturally favors cutting
    right around the actual engaging moment instead of always maxing out
    the duration. Each optional signal claims its own share of the weight
    only when it's actually available, so this degrades gracefully with
    whatever subset of signals this particular video has.

    required_keywords (campaign-brief mode): when given, ONLY windows whose
    transcript mentions at least one of these words/phrases are eligible at
    all - this is a hard requirement (e.g. "every clip must include a Goli
    segment"), not just a scoring nudge, since missing it means a clip is
    unpaid. It's a transcript proxy for an on-screen visual requirement, so
    it only catches segments people actually talk about - if the transcript
    never mentions any of them, this prints a warning and falls back to
    unconstrained selection rather than returning nothing.

    curiosity_pairs (hook/story/payoff mode - see find_curiosity_gaps): each
    (hook_seconds, payoff_seconds) pair proposes a window starting right at
    the hook and ending just after the payoff, so the clip IS the arc -
    open on the question, land the answer near the end. These compete
    directly against the density-based candidates with a scoring bonus,
    so a genuine hook->payoff arc wins over a merely-loud window of similar
    quality, but doesn't override a much stronger one.

    Returns (highlights, composite): a list of (start, end, highlight_words)
    tuples ranked best-first, and the whole-video per-second composite score
    array (reused by choose_hooks so it doesn't need to be rebuilt)."""
    duration = info.get("duration") or words[-1]["end"]
    n_seconds = int(duration) + 1

    print("Analyzing audio loudness...")
    audio = audio_energy_scores(video_path, n_seconds)
    keyword, density = transcript_scores(words, n_seconds)

    heatmap = info.get("heatmap")
    heat = heatmap_scores(heatmap, n_seconds) if heatmap else None
    fun = funniness if funniness is not None and funniness.any() else None
    comments = comment_scores if comment_scores is not None and comment_scores.any() else None

    # Comments are weighted well above everything else, on purpose: real
    # viewers pointing at a specific timestamp ("2:15 lol") is the single
    # most direct signal for "this is the entertaining part" this pipeline
    # has access to - more direct than a heuristic guessing from audio or
    # transcript keywords, and more direct than even the local LLM's read.
    weighted = [(audio, 0.15, "audio"), (keyword, 0.05, "keywords"), (density, 0.03, "speech density")]
    if heat is not None:
        weighted.append((heat, 0.20, "YouTube 'most replayed' heatmap"))
    if fun is not None:
        weighted.append((fun, 0.12, "LLM funniness"))
    if comments is not None:
        weighted.append((comments, 0.45, "comment timestamps (prioritized)"))
    total_w = sum(w for _, w, _ in weighted)
    composite = np.zeros(n_seconds)
    for arr, w, _ in weighted:
        composite += (w / total_w) * arr
    print("Highlight signals: " + ", ".join(name for _, _, name in weighted))

    max_window = max(1, min(int(max_duration), n_seconds))
    min_window = max(1, min(int(min_duration), max_window))
    step = 5
    lengths = sorted(set(range(min_window, max_window, step)) | {max_window})

    keyword_mask = None
    if required_keywords:
        terms = [k.strip() for k in required_keywords if k and k.strip()]
        if terms:
            pattern = re.compile("|".join(re.escape(k) for k in terms), re.IGNORECASE)
            keyword_mask = np.zeros(n_seconds, dtype=bool)
            for w in words:
                if pattern.search(w["word"]):
                    idx = int(w["start"])
                    if 0 <= idx < n_seconds:
                        keyword_mask[max(0, idx - 2):min(n_seconds, idx + 3)] = True
            if keyword_mask.any():
                print(f"Campaign requires: {', '.join(terms)} - only considering clips whose transcript "
                      "mentions this")
            else:
                print(f"WARNING: transcript never mentions the required keyword(s) ({', '.join(terms)}) - "
                      "can't guarantee any clip contains the required segment this way. Falling back to "
                      "unconstrained selection; you'll need to manually verify the required segment is "
                      "visible before posting.")
                keyword_mask = None

    working = composite.copy()
    highlights = []
    raw_scores = []
    pad = max(1, max_window // 4)  # keep picks from landing right next to each other

    for _ in range(num_highlights):
        best = None  # (score, start, length) - score favors density but leans toward longer on close calls
        for length in lengths:
            if length > n_seconds:
                continue
            sums = np.convolve(working, np.ones(length), mode="valid")
            if keyword_mask is not None:
                kw_sums = np.convolve(keyword_mask.astype(np.float64), np.ones(length), mode="valid")
                sums = np.where(kw_sums > 0, sums, -1e9)
            idx = int(np.argmax(sums))
            if sums[idx] <= -1e8:
                continue  # this length has no un-suppressed (and, if required, keyword-containing) room left
            density = sums[idx] / length
            length_bonus = 1.0 + 0.08 * (length - min_window) / max(1, max_window - min_window)
            score = density * length_bonus
            if best is None or score > best[0]:
                best = (score, idx, length)

        for hook_t, payoff_t in (curiosity_pairs or []):
            span = payoff_t - hook_t
            if span > max_window:
                continue  # can't fit the whole hook->payoff arc within the max clip length - skip it
            c_length = max(min_window, min(max_window, int(span) + 3))  # a little pad after the payoff
            c_start = int(hook_t)
            c_end = min(n_seconds, c_start + c_length)
            if c_end - c_start < 1:
                continue
            c_window = working[c_start:c_end]
            if c_window.max() <= -1e8:
                continue  # overlaps an already-picked/suppressed region
            if keyword_mask is not None and not keyword_mask[c_start:c_end].any():
                continue  # campaign's required-segment rule still applies to curiosity-gap picks
            c_density = float(c_window.mean())
            c_length_bonus = 1.0 + 0.08 * (c_length - min_window) / max(1, max_window - min_window)
            c_score = c_density * c_length_bonus * _CURIOSITY_BONUS
            if best is None or c_score > best[0]:
                best = (c_score, c_start, c_length)

        if best is None:
            break  # video's run out of un-suppressed room for another pick
        _, best_start, length = best
        best_end = min(best_start + length, n_seconds)

        h_start, h_end, h_words = _snap_to_words(words, best_start, best_end)
        # _snap_to_words aligns to word boundaries, which can shrink OR grow
        # the picked length - re-enforce [min_window, max_window] afterward
        # so a duration bound the caller asked for doesn't quietly slip.
        snapped_len = h_end - h_start
        if snapped_len < min_window:
            h_end = min(n_seconds, h_start + min_window)
            h_words = [w for w in words if w["start"] >= h_start and w["end"] <= h_end]
        elif snapped_len > max_window:
            h_end = h_start + max_window
            h_words = [w for w in words if w["start"] >= h_start and w["end"] <= h_end]
        highlights.append((h_start, h_end, h_words))
        raw_scores.append(float(composite[best_start:best_end].mean()))

        lo, hi = max(0, best_start - pad), min(n_seconds, best_end + pad)
        working[lo:hi] = -1e9

    if len(highlights) > 1:
        _print_hook_strength_ranking(raw_scores)

    return highlights, composite


def _print_hook_strength_ranking(raw_scores: list):
    """Prints each clip's estimated hook strength RELATIVE TO YOUR OTHER
    PICKS HERE - derived from the same audio/keyword/funniness/comment
    signals used to pick them, scaled so the strongest pick reads 100%.
    This is a heuristic ranking of this video's own candidates against each
    other, not a prediction of real-world TikTok performance - there's no
    legitimate way to know that without access to TikTok's own algorithm
    and other creators' real engagement data, which nobody outside TikTok
    has."""
    peak = max(raw_scores) or 1.0
    pcts = [round(100 * s / peak) for s in raw_scores]
    order = sorted(range(len(raw_scores)), key=lambda i: -raw_scores[i])
    print("\nEstimated hook strength (relative to your OTHER clips from this video only - "
          "not a real virality prediction, just a ranking by audio/keyword/funniness/comment signal strength):")
    for rank, i in enumerate(order, 1):
        print(f"  #{rank}  Clip {i + 1}: {pcts[i]}%")


def choose_hooks(composite: np.ndarray, words: list, highlights: list, hook_duration: float = 2.0,
                  curiosity_pairs: list | None = None, hook_lines: list | None = None) -> list:
    """Picks each highlight's opening 'hook' teaser - the punchline moment
    shown first, TikTok-style, before the clip plays through from the top.

    Priority order (substance over loudness, in every case):

    1. An unused hook_line (see find_hook_lines) - an actual QUESTION or
       bold/controversial STATEMENT, matched to its COMPLETE real sentence
       in the transcript, not a fixed-length time slice that can truncate
       it mid-thought. A line local to this highlight is preferred; if
       none is, the single strongest unused line ANYWHERE in the video is
       pulled in instead. Its full natural length is used, uncapped (aside
       from a generous sanity ceiling) - a real statement is trusted at
       whatever length it actually is.
    2. If every hook_line has already been used once elsewhere in this
       batch: REUSE the single strongest one rather than falling back to
       a generic loud/funny fragment. A repeated real hook has more
       substance than a fresh one that isn't actually a question or a
       statement - there's no reason to prefer novelty over that.
    3. curiosity_pairs - only reached when there are NO hook_lines for
       this video at all. When a highlight was built around one (see
       find_curiosity_gaps and find_top_highlights), i.e. it starts right
       at a detected hook moment, use that instant directly.
    4. Last resort, only when neither of the above exists anywhere in the
       video: the plain loud/funny/commented short-window search this
       pipeline used before hook detection existed.

    Returns a list of (start, end, hook_words) aligned with `highlights`."""
    n_seconds = len(composite)
    window = max(1, min(int(round(hook_duration)), n_seconds))
    used = np.zeros(n_seconds, dtype=bool)
    GLOBAL_MARGIN = 1.15  # global has to clearly beat local, not just edge it out on noise

    def best_in(lo: int, hi: int):
        lo, hi = max(0, lo), min(n_seconds, hi)
        if hi - lo < window:
            return None
        segment = np.where(used[lo:hi], -1e9, composite[lo:hi])
        sums = np.convolve(segment, np.ones(window), mode="valid")
        if sums.size == 0 or sums.max() <= -1e8:
            return None
        idx = int(np.argmax(sums))
        return float(sums[idx] / window), lo + idx

    def line_available(hl):
        lo, hi = max(0, int(hl["start"])), min(n_seconds, int(hl["end"]) + 1)
        return not used[lo:hi].any()

    all_lines = list(hook_lines or [])  # sorted strongest-first; never mutated, so it's there to reuse from
    remaining_lines = list(all_lines)   # consumed (once each) as highlights claim a line

    hooks = []
    for start, end, _ in highlights:
        natural_span, reused = None, False

        local_line = next(
            (hl for hl in remaining_lines if line_available(hl) and hl["start"] <= end and hl["end"] >= start - 3),
            None)
        chosen_line = local_line
        if chosen_line is None:
            chosen_line = next((hl for hl in remaining_lines if line_available(hl)), None)
        if chosen_line is not None:
            natural_span = (chosen_line["start"], chosen_line["end"])
            remaining_lines.remove(chosen_line)
        elif all_lines:
            chosen_line = all_lines[0]  # strongest line in the video, reused rather than falling back
            natural_span = (chosen_line["start"], chosen_line["end"])
            reused = True

        if natural_span is not None:
            h_start, h_end = natural_span
            if not reused:
                mark_lo, mark_hi = int(h_start), int(h_end) + 1
                used[max(0, mark_lo - window):min(n_seconds, mark_hi + window)] = True

            f_start, f_end, f_words = _snap_to_words(words, h_start, h_end)
            # This span already came from find_quote_span matching a real,
            # complete sentence to actual word boundaries - trust its
            # natural length. Only a pathological over-match needs
            # reining in, not a normal-length statement.
            if f_end - f_start > 15.0:
                f_end = f_start + 15.0
                f_words = [w for w in words if w["start"] >= f_start and w["end"] <= f_end]
            hooks.append((f_start, f_end, f_words))
            continue

        # No hook_lines exist for this video at all - fall back to
        # curiosity_pairs, then (last resort) the old loud/funny search.
        curiosity_hook_t = next(
            (hook_t for hook_t, _ in (curiosity_pairs or []) if abs(hook_t - start) <= 2.0), None)
        if curiosity_hook_t is not None:
            hs = int(curiosity_hook_t)
        else:
            local = best_in(int(start), int(end))
            glob = best_in(0, n_seconds)
            if glob is not None and (local is None or glob[0] > local[0] * GLOBAL_MARGIN):
                hs = glob[1]
            elif local is not None:
                hs = local[1]
            else:
                hs = int(start)
        he = min(hs + window, n_seconds)
        used[max(0, hs - window):min(n_seconds, he + window)] = True

        f_start, f_end, f_words = _snap_to_words(words, float(hs), float(he))
        max_hook_len = hook_duration + 2.0
        if f_end - f_start > max_hook_len:
            f_end = f_start + max_hook_len
            f_words = [w for w in words if w["start"] >= f_start and w["end"] <= f_end]
        hooks.append((f_start, f_end, f_words))

    return hooks


def concat_clips(clip_paths: list, out_path: Path):
    """Concatenates already-rendered clips (e.g. a hook teaser + the full
    highlight) into one final video. Uses ffmpeg's concat FILTER (re-encodes)
    rather than the concat demuxer, since it's robust to the small encoding
    differences between separately-rendered segments."""
    cmd = [FFMPEG_BIN, "-y"]
    for p in clip_paths:
        cmd += ["-i", str(p)]
    n = len(clip_paths)
    parts = "".join(f"[{i}:v][{i}:a]" for i in range(n))
    filter_complex = f"{parts}concat=n={n}:v=1:a=1[outv][outa]"
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# 4. Clip
# ---------------------------------------------------------------------------
def get_resolution(video_path: Path):
    proc = subprocess.run(
        [FFMPEG_BIN, "-i", str(video_path)],
        stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    match = re.search(r"(\d{2,5})x(\d{2,5})", proc.stderr)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1920, 1080


_FACE_CASCADES = None


def _get_face_cascades():
    """Frontal + profile cascades. Frontal alone misses a lot of real
    footage - interview/podcast framing has people angled toward each other
    rather than straight at the camera, which only the profile cascade
    catches (checked in both left/right directions via a horizontal flip)."""
    global _FACE_CASCADES
    if _FACE_CASCADES is None:
        base = cv2.data.haarcascades  # type: ignore[attr-defined]
        _FACE_CASCADES = (
            cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(base + "haarcascade_profileface.xml"),
        )
    return _FACE_CASCADES


def _detect_faces(gray: np.ndarray, min_size: int):
    frontal, profile = _get_face_cascades()
    candidates = list(frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_size, min_size)))
    candidates += list(profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_size, min_size)))
    flipped = cv2.flip(gray, 1)
    w = gray.shape[1]
    for fx, fy, fw, fh in profile.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=4, minSize=(min_size, min_size)):
        candidates.append((w - fx - fw, fy, fw, fh))  # mirror x back to the unflipped frame
    return candidates


_MIN_FACE_FRAC = 0.08  # a detection narrower than this fraction of frame width is a
                        # facecam-overlay-sized face (e.g. gameplay commentary), not the main subject


def track_face_x(video_path: Path, start: float, end: float, sample_fps: float = 3.0):
    """Samples the highlight window and returns [(t_relative, center_x_frac)].
    When multiple faces are detected, prefers whichever is closest to the
    previous pick (identity continuity) rather than always the largest -
    otherwise a two-person shot can flicker between faces frame to frame,
    which is what made the old tracking look choppy. A tiny (facecam-sized)
    detection is ignored entirely and resolves to frame-center instead -
    the "standard POV" look for gameplay commentary, where the only face
    on screen is a small webcam overlay, not the actual subject."""
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    track = []
    prev_x = None
    # Seek once, then decode forward sequentially (see asd_model/infer.py
    # for why: re-seeking before every read is ~25x slower on AV1 source).
    native_idx = 0
    t = start
    step = 1.0 / sample_fps
    while t < end:
        target_native = round((t - start) * src_fps)
        ok = True
        while native_idx < target_native:
            ok = cap.grab()
            if not ok:
                break
            native_idx += 1
        if not ok:
            break
        ok, frame = cap.read()
        if not ok:
            break
        native_idx += 1
        h, w = frame.shape[:2]
        work_w = min(w, 480)  # don't downscale low-res sources further than they already are
        small = cv2.resize(frame, (work_w, int(work_w * h / w))) if work_w < w else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sw = small.shape[1]
        faces = [f for f in _detect_faces(gray, max(20, work_w // 20)) if f[2] >= _MIN_FACE_FRAC * sw]
        if faces:
            if prev_x is None:
                fx, _, fw, _ = max(faces, key=lambda f: f[2] * f[3])
            else:
                anchor = prev_x
                fx, _, fw, _ = min(faces, key=lambda f: abs((f[0] + f[2] / 2) / sw - anchor))
            x_frac = (fx + fw / 2) / sw
            track.append((t - start, x_frac))
            prev_x = x_frac
        else:
            track.append((t - start, 0.5))
            prev_x = None
        t += step
    cap.release()
    return track


def _track_smart_crop_x(video_path: Path, start: float, end: float):
    """Finds who's actually SPEAKING (not just whichever face is biggest or
    closest to the last pick) using a real trained active-speaker-detection
    model (audio/lip-sync correlation - see asd_model/infer.py), which is
    what actually fixes tracking the wrong person in multi-person footage.
    Falls back to the plain face-position tracker if the model errors out
    or finds nothing (e.g. no faces at all)."""
    try:
        from asd_model.infer import track_active_speaker_x
        track = track_active_speaker_x(video_path, start, end)
        if track:
            return track
    except Exception as e:
        print(f"Active-speaker detection failed ({e}) - falling back to plain face tracking")
    return track_face_x(video_path, start, end)


def smooth_face_track(track: list, duration: float, median_window: int = 5):
    """Fills any leading/trailing gaps, median-filters out single-sample
    outliers (a stray false detection), then applies a gentle low-pass so
    the crop glides smoothly between positions instead of stepping - the
    "slides from one person to another" look, rather than jittery jumps."""
    pts = list(track)
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < duration:
        pts.append((duration, pts[-1][1]))

    times = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    n = len(xs)
    half = min(median_window, n) // 2
    filtered = [sorted(xs[max(0, i - half):min(n, i + half + 1)])[len(xs[max(0, i - half):min(n, i + half + 1)]) // 2]
                for i in range(n)]

    smoothed = [(times[0], filtered[0])]
    for t, x in zip(times[1:], filtered[1:]):
        _, prev_x = smoothed[-1]
        alpha = 0.18  # lower = smoother/slower glide, higher = snappier
        smoothed.append((t, prev_x + alpha * (x - prev_x)))
    return smoothed


def build_sendcmd_file(keyframes: list, crop_w: int, iw: int, path: Path):
    lines = []
    for t, x_frac in keyframes:
        x_px = int(max(0, min(iw - crop_w, round(x_frac * iw - crop_w / 2))))
        lines.append(f"{t:.3f} crop x {x_px};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clip_video(video_path: Path, start: float, end: float, out_path: Path,
                vertical: bool = False, vertical_mode: str = "smart"):
    duration = end - start
    vf = None
    if vertical and vertical_mode in ("crop", "smart"):
        iw, ih = get_resolution(video_path)
        if iw * 16 <= ih * 9:
            # source is taller than 9:16 - constrain by width instead (rare)
            crop_w = (iw // 32) * 32
            crop_h = crop_w * 16 // 9
        else:
            # normal case: landscape source, crop the sides, keep full height.
            # Rounding to a multiple of 32 (not just "even") keeps crop_w:crop_h
            # EXACTLY 9:16, not just approximately - approximate ratios force
            # ffmpeg to inject a corrective non-square pixel aspect ratio when
            # scaled to a fixed 1080x1920, which some players mishandle.
            crop_h = (ih // 32) * 32
            crop_w = crop_h * 9 // 16

        track = _track_smart_crop_x(video_path, start, end) if vertical_mode == "smart" else []
        if track:
            print(f"Smart crop: tracking face across {len(track)} samples")
            keyframes = smooth_face_track(track, duration)
            sendcmd_path = out_path.with_suffix(".facecmd.txt")
            build_sendcmd_file(keyframes, crop_w, iw, sendcmd_path)
            sendcmd_str = str(sendcmd_path).replace("\\", "\\\\").replace("'", r"\'")
            vf = (
                f"sendcmd=f='{sendcmd_str}',"
                f"crop={crop_w}:{crop_h}:{(iw - crop_w) // 2}:{(ih - crop_h) // 2},"
                f"scale=1080:1920:flags=lanczos,unsharp=5:5:0.8:5:5:0.0,format=yuv420p"
            )
        else:
            if vertical_mode == "smart":
                print("Smart crop: no face detected, falling back to a static center crop")
            vf = (
                f"crop={crop_w}:{crop_h}:{(iw - crop_w) // 2}:{(ih - crop_h) // 2},"
                "scale=1080:1920:flags=lanczos,unsharp=5:5:0.8:5:5:0.0"
            )
    elif vertical:
        # blur mode: letterbox the full original frame (nothing cropped out)
        # over a blurred, zoomed-in copy of itself filling the 9:16 canvas -
        # the standard "TikTok repost" look for widescreen source video.
        vf = (
            "split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1080:1920,boxblur=40:20,eq=brightness=-0.08[bgblur];"
            "[fg]scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,"
            "unsharp=5:5:0.8:5:5:0.0[fgscaled];"
            "[bgblur][fgscaled]overlay=(W-w)/2:(H-h)/2,format=yuv420p"
        )

    cmd = [FFMPEG_BIN, "-y", "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{duration:.3f}"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac",
            "-movflags", "+faststart", str(out_path)]

    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# 5. Build plain rolling subtitles - rendered as PNG frames with Pillow and
#    composited onto the clip with one ffmpeg overlay. Uniform white text,
#    one size, no color cycling, no emoji, no caps/punctuation flourishes.
# ---------------------------------------------------------------------------
NORMAL_COLOR = (255, 255, 255, 255)    # white text, classic unobtrusive caption look
OUTLINE_COLOR = (0, 0, 0, 255)         # thin black outline for legibility over any footage - no background box


def _measure(draw, text: str, font) -> tuple:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox


def _wrap_words(draw, words: list, font, max_width: int) -> list:
    """Greedily wraps whole words onto lines no wider than max_width."""
    space_w, _, _ = _measure(draw, " ", font)
    lines, current, current_w = [], [], 0
    for word in words:
        word_w, _, _ = _measure(draw, word, font)
        extra = space_w if current else 0
        if current and current_w + extra + word_w > max_width:
            lines.append(current)
            current, current_w = [], 0
            extra = 0
        current.append(word)
        current_w += extra + word_w
    if current:
        lines.append(current)
    return lines


def _draw_plain_lines(draw, lines: list, width: int, height: int, anchor: str, margin: int, font, font_size: int):
    """Draws wrapped lines directly over the footage - no background box,
    just plain white text with a thin black outline for legibility against
    any footage (the classic, unobtrusive caption look). anchor='top'|'bottom'
    anchors that edge at `margin` px and grows the block inward, so a
    caption that wraps to many lines can't run off-frame."""
    if not lines:
        return
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    line_gap = int(font_size * 0.25)
    stroke_width = max(2, font_size // 14)

    line_texts = [" ".join(words) for words in lines]
    line_widths = [draw.textlength(t, font=font) for t in line_texts]
    block_h = line_height * len(lines) + line_gap * (len(lines) - 1)
    y = margin if anchor == "top" else height - margin - block_h

    for text, w in zip(line_texts, line_widths):
        x = (width - w) / 2
        draw.text((x, y), text, font=font, fill=NORMAL_COLOR, stroke_width=stroke_width, stroke_fill=OUTLINE_COLOR)
        y += line_height + line_gap


def render_subtitle_frame(width: int, height: int, group: list | None, font_size: int) -> Image.Image:
    """One frame's worth of the rolling subtitle: the current few-word group,
    plain and uniform - no per-word highlighting."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if not group:
        return img

    font = _get_text_font(font_size)
    lines = _wrap_words(draw, group, font, int(width * 0.9))
    _draw_plain_lines(draw, lines, width, height, "bottom", int(height * 0.08), font, font_size)
    return img


def render_caption_overlay(words: list, clip_start: float, clip_end: float, width: int, height: int,
                            frame_dir: Path, group_size: int = 3) -> Path:
    """Renders one transparent PNG per group of words and writes an ffmpeg
    concat list for them. Returns the path to that list."""
    font_size = max(22, height // 20)  # nonchalant, not obstructing
    clip_duration = clip_end - clip_start

    rel_words = []
    for w in words:
        s = w["start"] - clip_start
        e = w["end"] - clip_start
        if e <= 0 or s >= clip_duration:
            continue
        rel_words.append({"word": w["word"], "start": max(0.0, s), "end": min(clip_duration, e)})

    # (start, end, group-or-None) states, contiguous across the whole clip -
    # a leading/trailing gap just means no caption. One state per GROUP (not
    # per word), since there's no per-word visual difference to animate.
    states = []
    if rel_words and rel_words[0]["start"] > 0:
        states.append((0.0, rel_words[0]["start"], None))
    for gi in range(0, len(rel_words), group_size):
        group = rel_words[gi:gi + group_size]
        states.append((group[0]["start"], group[-1]["end"], [w["word"] for w in group]))
    if rel_words and rel_words[-1]["end"] < clip_duration:
        states.append((rel_words[-1]["end"], clip_duration, None))
    if not states:
        states = [(0.0, clip_duration, None)]

    list_path = frame_dir / "frames.txt"
    last_name = ""
    with open(list_path, "w", encoding="utf-8") as f:
        for idx, (start_t, end_t, group) in enumerate(states):
            img = render_subtitle_frame(width, height, group, font_size)
            png_path = frame_dir / f"{idx:04d}.png"
            img.save(png_path)
            duration = max(0.02, end_t - start_t)
            f.write(f"file '{png_path.name}'\nduration {duration:.3f}\n")
            last_name = png_path.name
        # concat demuxer needs the last file repeated without a duration line
        f.write(f"file '{last_name}'\n")

    return list_path


def burn_captions(clip_path: Path, frames_list: Path, out_path: Path):
    cmd = [
        FFMPEG_BIN, "-y", "-i", str(clip_path),
        "-f", "concat", "-safe", "0", "-i", str(frames_list),
        "-filter_complex", "[1:v]format=rgba[ovl];[0:v][ovl]overlay=format=auto,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def _render_segment(video_path: Path, start: float, end: float, seg_words: list,
                     out_path: Path, args, tag: str):
    """Clips [start, end] from video_path and (optionally) burns rolling
    subtitles onto it, writing the result to out_path. Used for both the
    hook teaser and the main highlight body, since they're rendered
    independently and then concatenated."""
    raw_clip = out_path.with_name(out_path.stem + f"_{tag}_raw.mp4")
    facecmd = raw_clip.with_suffix(".facecmd.txt")
    print("== Clipping video ==")
    clip_video(video_path, start, end, raw_clip, vertical=args.vertical, vertical_mode=args.vertical_mode)

    if args.subtitles:
        print("== Burning subtitles ==")
        width, height = get_resolution(raw_clip)
        frame_dir = out_path.with_name(out_path.stem + f"_{tag}_frames")
        frame_dir.mkdir(exist_ok=True)
        frames_list = render_caption_overlay(seg_words, start, end, width, height, frame_dir, args.subtitle_group_size)
        burn_captions(raw_clip, frames_list, out_path)
        if not args.keep_temp:
            shutil.rmtree(frame_dir, ignore_errors=True)
            raw_clip.unlink(missing_ok=True)
    else:
        raw_clip.rename(out_path)

    if not args.keep_temp:
        facecmd.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _process_highlight(idx: int, total: int, video_path: Path, highlight: tuple,
                        hook: tuple | None, args, stem: str, n_workers: int,
                        campaign: dict | None = None, video_title: str = ""):
    """Renders one highlight (hook + body, or just the body) end to end and
    returns (idx, final_path). Runs in its own process - highlights are
    independent given the same source video, so num_clips of them can be
    rendered concurrently instead of one at a time, which is where most of
    a multi-clip run's wall time goes (each involves its own face-tracking
    pass, the single most CPU-heavy step)."""
    import cv2 as _cv2
    import torch as _torch
    threads = max(1, (os.cpu_count() or 4) // max(1, n_workers))
    _torch.set_num_threads(threads)
    _cv2.setNumThreads(threads)

    start, end, highlight_words = highlight
    suffix = f"_{idx}" if total > 1 else ""
    print(f"\n--- Clip {idx}/{total}: {start:.1f}s -> {end:.1f}s ({end - start:.1f}s) ---")

    final_path = OUTPUT_DIR / f"{stem}_final{suffix}.mp4"

    if args.hook and hook is not None:
        hstart, hend, hook_words = hook
        print(f"Clip {idx}/{total}: hook {hstart:.1f}s -> {hend:.1f}s ({hend - hstart:.1f}s)")

        hook_path = OUTPUT_DIR / f"{stem}_hook{suffix}.mp4"
        body_path = OUTPUT_DIR / f"{stem}_body{suffix}.mp4"
        _render_segment(video_path, hstart, hend, hook_words, hook_path, args, f"hook{suffix}")
        _render_segment(video_path, start, end, highlight_words, body_path, args, f"body{suffix}")

        concat_clips([hook_path, body_path], final_path)
        if not args.keep_temp:
            hook_path.unlink(missing_ok=True)
            body_path.unlink(missing_ok=True)

        total_len = (hend - hstart) + (end - start)
        print(f"Clip {idx}/{total}: total length {total_len:.1f}s")
    else:
        _render_segment(video_path, start, end, highlight_words, final_path, args, f"clip{suffix}")

    caption = ""
    if args.captions:
        caption = generate_caption(highlight_words, campaign, video_title)
        final_path.with_suffix(".txt").write_text(caption, encoding="utf-8")

    print(f"=== Clip {idx}/{total} rendered ===")
    return idx, final_path, caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", default=None, help="YouTube video URL (omit if using --local-video)")
    parser.add_argument("--local-video", type=str, default=None,
                         help="Path to an already-downloaded video file to use instead of a YouTube URL - "
                              "skips the download step entirely. No YouTube heatmap or comments are "
                              "possible for a local file; every other signal still works.")
    parser.add_argument("--min-duration", type=float, default=30.0,
                         help="Minimum TOTAL clip length in seconds, hook included (default 30 - enough room "
                              "for actual context, not just a punchy fragment)")
    parser.add_argument("--max-duration", type=float, default=45.0,
                         help="Maximum TOTAL clip length in seconds, hook included (default 45 - each clip's "
                              "actual length is chosen within [min, max] to fit just the concentrated engaging "
                              "part, not padded out to the max)")
    parser.add_argument("--whisper-model", default="small",
                         choices=["tiny", "base", "small", "medium", "large"],
                         help="Whisper model size (bigger = more accurate, slower)")
    parser.add_argument("--vertical", action="store_true", help="Reframe output to 9:16 for TikTok")
    parser.add_argument("--vertical-mode", default="smart", choices=["smart", "crop", "blur"],
                         help="smart: face-tracking crop that pans to follow the detected face (default). "
                              "crop: fill the whole 9:16 frame with a fixed center-crop, no face tracking. "
                              "blur: keeps the full frame but letterboxes it over a blurred background")
    parser.add_argument("--subtitles", action=argparse.BooleanOptionalAction, default=True,
                         help="Plain rolling word-by-word subtitles, uniform size and color (default on)")
    parser.add_argument("--subtitle-group-size", type=int, default=3,
                         help="Words shown on screen at once in the rolling subtitles")
    parser.add_argument("--hook", action=argparse.BooleanOptionalAction, default=True,
                         help="Prepend a short teaser of the highlight's loudest/funniest instant before the "
                              "clip plays through from the top - the classic TikTok 'hook' (default on)")
    parser.add_argument("--hook-duration", type=float, default=2.0,
                         help="Length in seconds of the opening hook teaser (1-3s recommended)")
    parser.add_argument("--funniness", action=argparse.BooleanOptionalAction, default=True,
                         help="Ask the local LLM to flag funny/weird/surprising moments beyond just loud ones, "
                              "for both highlight picking and the hook (needs Ollama; auto-starts it; default on)")
    parser.add_argument("--comments", action=argparse.BooleanOptionalAction, default=True,
                         help="Parse the video's own YouTube comments for timestamp mentions (e.g. '2:15 lol') "
                              "as a real-audience-reaction signal for highlight/hook picking (default on)")
    parser.add_argument("--curiosity-gaps", action=argparse.BooleanOptionalAction, default=True,
                         help="Ask the local LLM to find hook/payoff (curiosity gap) pairs - a question or "
                              "tense moment and where it's later resolved - and prefer clips that span that "
                              "arc, with the hook teaser using the real question/tense line (default on)")
    parser.add_argument("--hook-lines", action=argparse.BooleanOptionalAction, default=True,
                         help="Ask the local LLM for actual QUESTION or bold/controversial STATEMENT lines "
                              "to use as the opening hook - matched to their complete real sentence, not a "
                              "fixed-length clip. Doesn't need a later payoff, unlike --curiosity-gaps, and "
                              "takes priority over it and over the plain loud/funny hook search (default on)")
    parser.add_argument("--captions", action=argparse.BooleanOptionalAction, default=True,
                         help="Auto-draft a posting caption per clip with the local LLM, saved next to the "
                              "clip as a .txt file (default on)")
    parser.add_argument("--campaign-pdf", type=str, default=None,
                         help="Path to a campaign brief PDF (brand-deal / ClipFarm style). Its rules - "
                              "required hashtag, a required on-screen segment/keyword, caption tone, things "
                              "to avoid - are extracted with the local LLM and applied to highlight picking "
                              "and captions. Omit for a plain, unconstrained run.")
    parser.add_argument("--num-clips", type=int, default=5,
                         help="Cut the top N best-scoring, non-overlapping highlights per video (default 5)")
    parser.add_argument("--max-height", type=int, default=1080,
                         help="Max video resolution to download (e.g. 1080, 1440). Higher = bigger download, "
                              "slower to decode/encode")
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate files")
    args = parser.parse_args()
    if not args.url and not args.local_video:
        parser.error("provide either a YouTube url or --local-video PATH")
    if args.url and args.local_video:
        parser.error("provide only one of a YouTube url or --local-video PATH, not both")

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.local_video:
        print("== 1/5 Using local video file ==")
        video_path, info = load_local_video(Path(args.local_video))
        print(f"Using: {video_path}")
    else:
        print("== 1/5 Downloading video ==")
        video_path, info = download_video(args.url, args.max_height, get_comments=args.comments)
        print(f"Downloaded: {video_path}")

    print("== 2/5 Transcribing ==")
    words = transcribe(video_path, args.whisper_model)
    print(f"Transcribed {len(words)} words")

    n_seconds = int(info.get("duration") or words[-1]["end"]) + 1  # type: ignore[union-attr]
    video_title: str = info.get("title") or ""  # type: ignore[union-attr]

    campaign = None
    if args.campaign_pdf:
        print(f"== Reading campaign brief: {args.campaign_pdf} ==")
        try:
            pdf_text = extract_pdf_text(Path(args.campaign_pdf))
            campaign = extract_campaign_rules(pdf_text)
        except Exception as e:
            print(f"Couldn't read campaign PDF ({e}) - continuing without campaign rules")
        if campaign is None:
            print("No usable campaign rules extracted - running unconstrained")
        else:
            (OUTPUT_DIR / f"{video_path.stem}_campaign.json").write_text(
                json.dumps(campaign, indent=2), encoding="utf-8")

    funniness = None
    if args.funniness:
        print("== 3/5 Scoring transcript for funny/weird moments ==")
        funniness = llm_funniness_scores(words, n_seconds, info.get("title", ""))  # type: ignore[union-attr]

    comment_scores = None
    if args.comments:
        comment_scores = comment_timestamp_scores(info, n_seconds)  # type: ignore[arg-type]

    curiosity_pairs = []
    if args.curiosity_gaps:
        curiosity_pairs = find_curiosity_gaps(words, n_seconds, video_title)

    hook_lines = []
    if args.hook_lines:
        # Enough candidates that every clip in the batch can get a real,
        # distinct one instead of the pool running out partway through and
        # the rest silently falling back to a lesser signal.
        hook_lines = find_hook_lines(words, n_seconds, video_title, min_candidates=max(8, args.num_clips * 2))

    hook_duration = args.hook_duration if args.hook else 0.0
    # min_main is NOT hook_duration-reduced (unlike max_main) - the hook is
    # additional on top, and word-boundary snapping (_snap_to_words) can trim
    # a candidate window slightly, so this leaves real margin between the
    # body's floor and the user's requested TOTAL floor instead of cutting
    # it exactly at the line, which measurably let a clip land under the
    # floor before this fix.
    min_main = max(5.0, args.min_duration)
    # max_main must reserve room for the WORST-CASE hook length, not just
    # the nominal hook_duration - a hook_lines match is anchored to a real
    # complete sentence and is trusted at its natural length up to the 15s
    # sanity ceiling in choose_hooks, and reserving only hook_duration for
    # it measurably let a clip's total run over the requested ceiling.
    max_hook_reserve = 15.0 if (args.hook and args.hook_lines) else (hook_duration + 2.0 if args.hook else 0.0)
    max_main = max(min_main + 1.0, args.max_duration - max_hook_reserve)

    required_keywords = campaign.get("required_keywords") if campaign else None

    print(f"== 4/5 Finding top {args.num_clips} highlight(s) ==")
    highlights, composite = find_top_highlights(video_path, words, info, min_main, max_main,  # type: ignore[arg-type]
                                                 args.num_clips, funniness, comment_scores,
                                                 required_keywords, curiosity_pairs)
    if not highlights:
        raise RuntimeError("Couldn't find any highlight windows in this video.")
    if len(highlights) < args.num_clips:
        print(f"Only found room for {len(highlights)} non-overlapping highlight(s) in this video.")

    hooks = (choose_hooks(composite, words, highlights, args.hook_duration, curiosity_pairs, hook_lines)
             if args.hook else [None] * len(highlights))

    stem = video_path.stem
    n_workers = max(1, min(len(highlights), 4))
    print(f"== 5/5 Rendering {len(highlights)} clip(s) ({n_workers} at a time) ==")

    results = {}
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_process_highlight, i, len(highlights), video_path,
                                 highlight, hooks[i - 1], args, stem, n_workers, campaign, video_title)
                for i, highlight in enumerate(highlights, 1)
            ]
            for fut in as_completed(futures):
                idx, path, caption = fut.result()
                results[idx] = (path, caption)
    else:
        idx, path, caption = _process_highlight(1, 1, video_path, highlights[0], hooks[0], args, stem, 1,
                                                  campaign, video_title)
        results[idx] = (path, caption)

    final_paths = [results[i] for i in range(1, len(highlights) + 1)]

    print(f"\nDone! {len(final_paths)} clip(s):")
    for p, caption in final_paths:
        print(f"  {p}")
        if caption:
            print(f"    caption: {caption}")


if __name__ == "__main__":
    main()
