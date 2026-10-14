"""
Minimal-inference-only integration of Light-ASD (CVPR 2023) for active-
speaker detection: given a clip and a time range, figures out which
detected face is ACTUALLY TALKING at each instant using a real trained
audio-visual model (audio/lip-sync correlation), not a size/position
heuristic - so a smart-crop can follow whoever's speaking instead of just
the biggest or most-recently-tracked face.

Original model + face detector: https://github.com/Junhua-Liao/Light-ASD
(MIT License). Only the inference path is vendored here (no training/
eval-benchmark code); model/loss/S3FD files are used close to verbatim,
with the S3FD weight path fixed to be relative to this package instead of
the caller's cwd.
"""
from __future__ import annotations

import os
import sys
import warnings
from collections import Counter

import cv2
import numpy as np
import torch
from python_speech_features import mfcc
from scipy.signal import medfilt

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model.Model import ASD_Model  # noqa: E402
from model.faceDetector.s3fd import S3FD  # noqa: E402
from loss import lossAV  # noqa: E402

_WEIGHT_PATH = os.path.join(_THIS_DIR, "weight", "pretrain_AVA_CVPR.model")
_ANALYSIS_FPS = 25
_WORK_WIDTH = 256
_DETECT_EVERY = 4  # run the face detector every Nth frame, hold its result for the skipped ones - CPU
                    # inference is the bottleneck. Safe to hold longer than you'd think: the debounce
                    # window downstream is 13 samples (~0.5s) and the track's own sample rate (25/s,
                    # matching the sendcmd keyframe density burned into the final video) doesn't change
                    # with this - only how often the held detection gets refreshed.
_MIN_FACE_FRAC = 0.08  # a detection narrower than this fraction of frame width is treated as a
                        # facecam-overlay-sized face (e.g. gameplay commentary), not the main subject
_CROP_SCALE = 0.40  # padding factor around each detected face box, matches upstream

_state: dict = {}


def _get_models(device: str = "cpu"):
    if "model" in _state:
        return _state["model"], _state["lossav"], _state["detector"]

    full_state = torch.load(_WEIGHT_PATH, map_location=device)
    model_state = {k[len("model."):]: v for k, v in full_state.items() if k.startswith("model.")}
    lossav_state = {k[len("lossAV."):]: v for k, v in full_state.items() if k.startswith("lossAV.")}

    model = ASD_Model().to(device)
    model.load_state_dict(model_state)
    model.eval()

    lav = lossAV().to(device)
    lav.load_state_dict(lossav_state)
    lav.eval()

    detector = S3FD(device=device)

    _state["model"], _state["lossav"], _state["detector"] = model, lav, detector
    return model, lav, detector


def _iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter + 1e-6)


def _build_tracks(frame_dets, iou_thresh=0.5, max_gap=10, min_track=13):
    """Greedy IOU tracking across frames (adapted from Light-ASD's
    track_shot). frame_dets[i] is a list of bboxes (x1,y1,x2,y2) detected in
    frame i. Returns [{'frames': [...], 'bboxes': [...]}], one per identity
    seen for at least min_track frames."""
    pool = [list(dets) for dets in frame_dets]
    tracks = []
    while True:
        track_frames, track_bboxes = [], []
        for i, dets in enumerate(pool):
            if not track_frames:
                if dets:
                    track_bboxes.append(dets.pop(0))
                    track_frames.append(i)
                continue
            if i - track_frames[-1] > max_gap:
                break
            if not dets:
                continue
            best_j, best_iou = -1, iou_thresh
            for j, bbox in enumerate(dets):
                v = _iou(bbox, track_bboxes[-1])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                track_bboxes.append(dets.pop(best_j))
                track_frames.append(i)
        if not track_frames:
            break
        if len(track_frames) >= min_track:
            tracks.append({"frames": track_frames, "bboxes": track_bboxes})
    return tracks


def _crop_face_sequence(frames: list, track: dict) -> tuple:
    """Builds the (T,112,112) grayscale visual feature for one track,
    matching Light-ASD's exact preprocessing: median-filtered box center/
    size -> padded square crop -> resize 224x224 -> center-crop 112x112."""
    bboxes = np.array(track["bboxes"], dtype=np.float64)
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    size = np.maximum(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]) / 2
    k = min(13, len(cx) - (1 - len(cx) % 2))  # medfilt needs an odd kernel <= len
    k = max(1, k if k % 2 else k - 1)
    cx, cy, size = medfilt(cx, k), medfilt(cy, k), medfilt(size, k)

    faces = []
    for i, frame_idx in enumerate(track["frames"]):
        frame = frames[frame_idx]
        bs = size[i]
        bsi = int(bs * (1 + 2 * _CROP_SCALE))
        padded = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)), mode="constant", constant_values=110)
        my, mx = cy[i] + bsi, cx[i] + bsi
        y0, y1 = int(my - bs), int(my + bs * (1 + 2 * _CROP_SCALE))
        x0, x1 = int(mx - bs * (1 + _CROP_SCALE)), int(mx + bs * (1 + _CROP_SCALE))
        face = padded[max(0, y0):y1, max(0, x0):x1]
        if face.size == 0:
            face = np.zeros((10, 10, 3), dtype=np.uint8)
        face = cv2.resize(face, (224, 224))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = gray[56:168, 56:168]  # center 112x112 - the model's actual input size
        faces.append(gray)
    return np.array(faces), cx, cy


def _score_track(model, lav, visual: np.ndarray, audio_signal: np.ndarray, device: str) -> np.ndarray:
    """Runs one track's (T,112,112) face crops + its aligned raw audio
    signal through the model, returns a per-frame speaking score."""
    audio_feat = mfcc(audio_signal, 16000, numcep=13, winlen=0.025, winstep=0.010)
    length = min((audio_feat.shape[0] - audio_feat.shape[0] % 4) / 100, visual.shape[0] / 25) * 25
    length = int(round(length))
    if length <= 0:
        return np.zeros(visual.shape[0])
    audio_feat = audio_feat[:int(round(length / 25 * 100)), :]
    video_feat = visual[:length]

    with torch.no_grad():
        inputA = torch.FloatTensor(audio_feat).unsqueeze(0).to(device)
        inputV = torch.FloatTensor(video_feat).unsqueeze(0).to(device)
        embedA = model.forward_audio_frontend(inputA)
        embedV = model.forward_visual_frontend(inputV)
        out = model.forward_audio_visual_backend(embedA, embedV)
        scores = lav.forward(out, labels=None)
    if len(scores) < visual.shape[0]:
        scores = np.pad(scores, (0, visual.shape[0] - len(scores)), mode="edge")
    return scores


def track_active_speaker_x(video_path, start: float, end: float, device: str = "cpu"):
    """Returns [(t_relative, x_frac)] for whichever identity is judged to be
    actively speaking at each ~1/25s instant - the real audio-visual signal,
    not just face size/position. Empty if no faces were tracked at all
    (caller should fall back to a plain static crop)."""
    model, lav, detector = _get_models(device)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    # Seek ONCE to `start`, then decode forward sequentially and keep the
    # native frame nearest each analysis timestamp. Re-seeking (cap.set)
    # before every single read - the original approach - forces the
    # decoder back to the nearest keyframe and back on EVERY frame, which
    # measured ~25x slower than sequential decode on AV1 source video (the
    # single biggest cost in the whole pipeline). cap.grab() (decode, no
    # colorspace conversion/copy) is used for the frames we skip over.
    frames = []
    step = 1.0 / _ANALYSIS_FPS
    duration = end - start
    native_idx = 0
    t_next = 0.0
    while t_next < duration:
        target_native = round(t_next * src_fps)
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
        if w > _WORK_WIDTH:
            frame = cv2.resize(frame, (_WORK_WIDTH, int(_WORK_WIDTH * h / w)))
        frames.append(frame)
        t_next += step
    cap.release()
    if not frames:
        return []

    work_w = frames[0].shape[1]
    min_face_w = _MIN_FACE_FRAC * work_w

    frame_dets = []
    last_dets = []
    for i, frame in enumerate(frames):
        if i % _DETECT_EVERY != 0:
            frame_dets.append(last_dets)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = detector.detect_faces(rgb, conf_th=0.9, scales=[1])
        # Drop facecam-overlay-sized detections (e.g. a small webcam box in
        # the corner of gameplay footage) - tracking those instead of the
        # main subject is what dragged the crop off the actual game action.
        last_dets = [tuple(b[:4]) for b in bboxes if (b[2] - b[0]) >= min_face_w]
        frame_dets.append(last_dets)

    tracks = _build_tracks(frame_dets)
    if not tracks:
        return []

    # Whole-clip mono 16kHz audio, sliced per track below.
    audio = _decode_audio_16k_mono(video_path, start, end)

    n_frames = len(frames)
    frame_w = frames[0].shape[1]
    all_scores = np.full((len(tracks), n_frames), -1e9)
    all_x = np.zeros((len(tracks), n_frames))

    for ti, track in enumerate(tracks):
        visual, cx, _cy = _crop_face_sequence(frames, track)
        f0, f1 = track["frames"][0], track["frames"][-1]
        a0 = int(f0 / _ANALYSIS_FPS * 16000)
        a1 = int((f1 + 1) / _ANALYSIS_FPS * 16000)
        audio_slice = audio[a0:a1] if len(audio) >= a1 else audio[a0:]
        if len(audio_slice) < 400:  # need at least ~25ms for a single MFCC frame
            continue
        scores = _score_track(model, lav, visual, audio_slice, device)
        for i, frame_idx in enumerate(track["frames"]):
            if i < len(scores):
                all_scores[ti, frame_idx] = scores[i]
                all_x[ti, frame_idx] = cx[i] / frame_w

    winners = np.argmax(all_scores, axis=0)
    has_any = np.any(all_scores > -1e8, axis=0)

    # Debounce: majority-vote the winner over a short rolling window so a
    # single noisy frame can't flip the target - this is what actually
    # fixes "choppy" switching, on top of the position glide already
    # applied downstream by smooth_face_track.
    window = 13  # ~0.5s at 25fps
    debounced = winners.copy()
    for i in range(n_frames):
        if not has_any[i]:
            continue
        lo, hi = max(0, i - window // 2), min(n_frames, i + window // 2 + 1)
        candidates = [winners[j] for j in range(lo, hi) if has_any[j]]
        debounced[i] = Counter(candidates).most_common(1)[0][0]

    # No qualifying (large-enough) face at all -> treat as "standard POV":
    # settle toward frame-center instead of holding a stale prior position.
    # This is what keeps gameplay stretches (no real subject on screen, just
    # a small facecam or none) centered on the actual game action rather
    # than drifting toward wherever a face was last tracked before the
    # gameplay segment started.
    track_out = [
        (i / _ANALYSIS_FPS, float(all_x[debounced[i], i]) if has_any[i] else 0.5)
        for i in range(n_frames)
    ]

    return track_out


def _decode_audio_16k_mono(video_path, start: float, end: float) -> np.ndarray:
    import subprocess

    ffmpeg_bin = os.environ.get("ASD_FFMPEG_BIN", "ffmpeg")
    cmd = [
        ffmpeg_bin, "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{end - start:.3f}",
        "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.int16)
