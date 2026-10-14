# Clipping App

A local pipeline that turns a long-form video into a batch of short, vertical,
captioned TikTok/Reels/Shorts clips — with a local web UI, brand-campaign
compliance support, and no cloud services or API keys required. Everything
(transcription, content scoring, caption writing) runs on-device.

## What it does

Given a YouTube URL (or an uploaded video file), the pipeline:

1. Downloads the video (or uses the uploaded file directly).
2. Transcribes it with Whisper (word-level timestamps).
3. Scores the whole video for "highlight-worthiness" using several signals
   at once, and picks the best non-overlapping windows.
4. For each pick, finds a short opening "hook" — the loudest/funniest/most
   attention-grabbing instant, pulled from anywhere in the video if that
   beats the highlight's own content — and prepends it as a teaser before
   the clip plays through from the top.
5. Reframes each clip to vertical 9:16 with face/speaker-tracking (crops and
   pans to follow whoever's talking).
6. Burns in plain, TikTok-style rolling word-by-word captions.
7. Drafts a posting caption per clip with a local LLM.
8. Optionally applies a brand/campaign brief's rules (from an uploaded PDF)
   to both clip selection and captions.

Output: N short (30–45s) vertical `.mp4` clips per run, each with a caption
`.txt` sidecar, ready to post.

## Why it's built this way

- **Fully local.** Whisper for transcription, a local Ollama model
  (`llama3.2:3b`) for all "understanding" tasks (funniness scoring, caption
  writing, campaign-rule extraction), OpenCV/S3FD + a vendored Light-ASD
  model for face/speaker tracking. No API keys, no per-call cost.
- **Portable.** ffmpeg, Deno, and Ollama binaries are bundled under `.bin/`
  and `.ollama/` rather than relying on a system-wide install.
- **Honest about its limits.** Where a signal is a heuristic rather than a
  guarantee (e.g. "did the required product actually appear on screen,"
  or "will this clip go viral"), the tool says so explicitly instead of
  presenting a guess as certainty.

## Highlight-selection signals

Each candidate window is scored by a weighted blend of whichever of these
are available for a given video:

| Signal | What it captures |
|---|---|
| **Comment timestamps** *(dominant, 45%)* | Real viewers pointing at a timestamp in the video's own YouTube comments (e.g. "2:15 lol") — the most direct "this part is good" signal available. |
| **YouTube "most replayed" heatmap** | The official audience-engagement graph under the seek bar, when YouTube has generated one. |
| **LLM funniness** | The local LLM reads the actual transcript and flags funny/weird/surprising moments — catches jokes and twists that aren't simply loud. |
| **Audio energy** | Loudness envelope — reactions, shouting, music stings. |
| **Transcript keywords / speech density** | Excitement words, exclamations, talking pace. |
| **Curiosity gaps (hook/story/payoff)** | The LLM finds a question/tense moment and a *later* moment where it's resolved, and clips are preferred that span that whole arc — with the opening teaser using the real hook line, not just whatever's loudest. |

Each clip's own **length** (30–45s, configurable) is chosen per-clip, not
fixed — the picker tries several candidate lengths and keeps whichever one
is most tightly concentrated on the actual engaging moment, so clips aren't
padded out to a maximum just because they can be.

A **relative "hook strength" ranking** is printed per batch (e.g. `#1 Clip 2:
100%`) — explicitly a comparison of a video's own clips against each other
using the signals above, not a virality prediction (there's no legitimate
way to know that without access to TikTok's own algorithm).

## Campaign-brief mode (monetized "clip farm" style deals)

Upload a campaign brief PDF alongside the video and the local LLM extracts
the rules it can actually act on:

- a required hashtag
- a required on-screen segment/keyword every clip must contain
- the caption tone
- what gets a clip rejected

Highlight-picking then **hard-requires** the transcript to mention the
required keyword (a transcript proxy for an on-screen visual requirement —
if the video never says it out loud, this prints a clear warning and falls
back to unconstrained selection rather than silently failing). Every
caption gets the required hashtag enforced. No PDF uploaded → runs exactly
as normal, with captions as a bonus.

## Face/speaker tracking

The vertical-crop pan follows whoever's actually speaking, using a vendored
copy of **Light-ASD** (CVPR 2023): an S3FD face detector plus an
audio-visual active-speaker-detection model, with a debounce/smoothing pass
so the crop glides between speakers instead of jumping. Falls back to a
plain Haar-cascade face tracker if the ASD model can't be used. A detection
smaller than ~8% of the frame width is treated as a facecam-overlay-sized
face (e.g. gameplay commentary) and ignored, so the crop centers on the
actual game footage instead of chasing a tiny webcam box.

## Project structure

```
clip.py                  Main pipeline (single file, run directly or via the web UI)
webui/
  server.py              Flask app: job queue, progress parsing, caption chatbox API
  index.html             Single-page UI: URL/file/PDF inputs, progress bar, clip gallery
asd_model/
  infer.py               Light-ASD inference entry point (face tracking)
  model/, weight/        Vendored Light-ASD model code + pretrained weights
  loss.py
downloads/                Downloaded/uploaded source videos (cleared per job by the web UI)
output/                   Generated clips + caption .txt + campaign .json sidecars
campaigns/                Uploaded campaign-brief PDFs (one per job)
.bin/, .ollama/, .python312/   Bundled portable binaries (ffmpeg, Ollama, Python)
requirements.txt
```

## Running it

### Web UI (recommended)

```
./venv/bin/python webui/server.py
```

Open **http://127.0.0.1:5050**. Paste a YouTube URL *or* upload a video
file, optionally attach a campaign brief PDF, and click Generate. Each
result clip has a small chatbox under it — type something like "make it
funnier" or "shorter" and the caption gets revised on the spot (still
enforcing any campaign hashtag/tone rules underneath).

### CLI

```
./venv/bin/python clip.py "https://youtube.com/watch?v=..." --vertical
```

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `--local-video PATH` | — | Use a local file instead of a YouTube URL |
| `--min-duration` / `--max-duration` | `30` / `45` | Total clip length bounds (hook included) |
| `--num-clips` | `5` | How many highlights to cut |
| `--whisper-model` | `small` | `tiny`/`base`/`small`/`medium`/`large` |
| `--vertical` / `--vertical-mode` | off / `smart` | 9:16 reframe; `smart`/`crop`/`blur` |
| `--hook` / `--hook-duration` | on / `2.0` | Opening teaser toggle + length |
| `--funniness` | on | LLM funniness signal |
| `--comments` | on | YouTube comment-timestamp signal |
| `--curiosity-gaps` | on | Hook/story/payoff arc detection |
| `--captions` | on | Auto-drafted posting caption per clip |
| `--campaign-pdf PATH` | — | Apply a campaign brief's rules |
| `--subtitles` / `--subtitle-group-size` | on / `3` | Burned-in caption toggle + words-on-screen |
| `--max-height` | `1080` | Download resolution ceiling |
| `--keep-temp` | off | Keep intermediate render files |

Everything (comments, funniness, curiosity-gaps, captions) degrades
gracefully to "off"/unconstrained if the local LLM or YouTube data isn't
available — nothing hard-fails the run.

## Setup

This repo ships source only — the bundled runtimes and vendored model
weights are excluded from git (see `.gitignore`) because they're large,
platform-specific, or easy to re-fetch:

- `venv/` — create with `python3 -m venv venv && ./venv/bin/pip install -r requirements.txt`
- `.bin/`, `.ollama/` — portable ffmpeg/Deno/Ollama binaries; install Ollama
  normally and pull the model with `ollama pull llama3.2:3b`, or point
  `clip.py` at a system ffmpeg instead
- `asd_model/weight/pretrain_AVA_CVPR.model` and
  `asd_model/model/faceDetector/s3fd/sfd_face.pth` — pretrained Light-ASD /
  S3FD weights, available from the
  [Light-ASD repo](https://github.com/Junhua-Liao/Light-ASD); the app falls
  back to a plain Haar-cascade face tracker if these aren't present

## Performance notes

Clips render **in parallel** (up to 4 at a time) rather than one at a time,
since each involves its own face-tracking pass — the single most CPU-heavy
step. A full 5-clip run typically finishes in a few minutes on this
machine.
