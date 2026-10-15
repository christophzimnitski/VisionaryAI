# VisionaryAI

A modular, tested multi-object tracking pipeline. It follows an arbitrary number
of moving objects across video frames, assigns each a stable identity, survives
collisions and short occlusions, holds an object's identity by **object
permanence** while it is off-screen (pointing an arrow at where it is expected to
be), and **re-identifies** a returning object by its remembered intrinsic
features rather than recomputing it from scratch.

This project began as two divergent single-file scripts (`main.py`, `main2.py`)
and was refactored into a layered package with a clear separation between
detection, motion modelling, data association, lifecycle management, rendering,
and input/output.

## Architecture

```
src/visionary/
  configuration.py            # TrackerConfiguration: every tunable knob, one frozen dataclass
  geometry.py                 # IoU, NMS, and off-screen expectation (arrow + bearing) -- pure
  logging_configuration.py    # structured logging instead of print()
  detection/
    base.py                   # ObjectDetector protocol + Detection dataclass
    motion_difference_detector.py   # canonical default detector (frame differencing)
    template_matching_detector.py   # alternative detector for a known template
  features/
    appearance_traits.py      # appearance descriptors, type tokens, similarity
    object_feature_memory.py  # intrinsic-feature "object file" + partial-match re-identification
  tracking/
    kalman_filter.py          # constant-velocity Kalman filter wrapper
    track.py                  # TrackedObject + lifecycle + feature memory
    association.py            # Hungarian assignment fusing IoU + Mahalanobis + appearance
    anticipatory_focus_tracker.py   # focus layer: per-track local search, occlusion ordering
    camera_motion_compensation.py   # ORB/RANSAC homography (applied to predictions)
    multi_object_tracker.py   # the orchestrator; returns structured results, draws nothing
  io/
    video_sources.py          # video file, webcam, and deterministic synthetic scene (ground truth)
  visualization/
    overlay_renderer.py       # all OpenCV drawing, isolated here (incl. off-screen pointer)
  evaluation/
    mot_metrics_evaluation.py            # MOTA / IDF1 (optional motmetrics dependency)
    identity_consistency_evaluation.py   # dependency-free ID-switch / recall scoring
  application.py              # command-line entry point that wires everything together
```

### Canonical design decisions

- **Detector:** motion-difference detection is the default because it needs no
  per-object template; template matching is a drop-in alternative. Both satisfy
  the `ObjectDetector` protocol, so swapping in a learned detector later is local.
- **Association:** optimal Hungarian assignment over a combined cost fusing
  spatial overlap (IoU), motion consistency (Mahalanobis distance under the Kalman
  innovation covariance), and appearance similarity. This replaced the original
  greedy matching and genuinely honours the configured gate thresholds.
- **Two layers, focus then filter:** a *focus layer* re-locates each known object
  in a small window around its prediction (cheap, and the key to holding identity
  through a collision); a *filter layer* then scans the whole frame for changes
  that no tracked object already explains, to discover or re-identify objects.
- **Object permanence:** a track whose expected position leaves the frame is held
  (not retired) and reported with an off-screen pointer, then restored on return.
- **Active memory:** each object keeps a small, brightness-invariant feature
  memory; a returning object is recognised by a partial match against it, so its
  identity survives even when its predicted position can no longer be trusted.
- **Rendering is separate from tracking:** the tracker returns a drawing-free
  `FrameProcessingResult`, so it runs headless and is unit-testable.

### Bugs fixed during the refactor

- The camera-motion homography is now **applied** to predictions (it was computed
  and discarded before).
- The IoU gate threshold passed to association is **honoured** (it was ignored).
- Each detection's appearance descriptor is **computed once** and reused (it was
  recomputed several times per detection).
- `main.py` / `main2.py` duplication was collapsed into one package.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,evaluation]"
```

## Running

```bash
# Deterministic synthetic demo (no camera needed)
python -m visionary.application --source synthetic

# Live webcam
python -m visionary.application --source webcam --camera-index 0

# A video file, headless (e.g. on a server)
python -m visionary.application --source video --video-path clip.mp4 --headless --max-frames 500
```

## Testing and evaluation

```bash
pytest                                   # unit + integration tests
python -m visionary.evaluation.identity_consistency_evaluation   # ID switches + recall (no extra deps)
python -m visionary.evaluation.mot_metrics_evaluation            # MOTA / IDF1 (needs motmetrics)
```

## Roadmap

- Stronger core: optional ByteTrack / OC-SORT association, then a learned detector
  (YOLO / RT-DETR) and learned re-identification embeddings.
- Multi-class support and a GPU detection backend.
- Batch / offline processing mode and a small live dashboard.

## License

The code is released under the [MIT License](LICENSE).

The synthetic demo's rocket sprite (`src/visionary/assets/rocket.png`) is the
rocket emoji (U+1F680) from Google's [Noto Emoji](https://github.com/googlefonts/noto-emoji)
project, copyright Google LLC, used under the Apache License 2.0. Its license
texts ship next to it in `src/visionary/assets/`.
