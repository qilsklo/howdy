# Howdy ONNX pipeline prototype

`compare_onnx_prototype.py` is a self-contained prototype of a Windows-Hello-style
verification pipeline intended to replace the dlib stack in `compare.py`:

| Stage | Old (`compare.py`) | New (prototype) |
|---|---|---|
| Detection | dlib HOG / MMOD CNN | SCRFD (InsightFace), 5-point landmarks |
| Alignment | dlib 5-landmark predictor | Similarity transform to canonical 112x112 ArcFace crop |
| Embedding | dlib ResNet, 128-D | ArcFace ResNet-50, 512-D on a hypersphere |
| Matching | Euclidean distance | Cosine similarity, threshold τ (default 0.40) |
| PAD / liveness | none | LBP micro-texture + landmark parallax on the IR stream |
| Compute | CPU only | AMD iGPU via ONNX Runtime (ROCm/MIGraphX) or OpenCV DNN (Vulkan) |

## Setup

1. **Weights** (SCRFD + ArcFace, open source, InsightFace model zoo):

   ```sh
   ./onnx-data/install.sh            # buffalo_l: SCRFD-10G + ArcFace R50
   ./onnx-data/install.sh buffalo_s  # smaller/faster alternative
   ```

2. **Inference engine.**

   ```sh
   pip install --user --break-system-packages onnxruntime
   ```

   To move inference onto the Radeon 780M, install the ROCm stack and the
   ROCm build of ONNX Runtime; the ROCm/MIGraphX execution providers are then
   picked up automatically:

   ```sh
   pip install onnxruntime-rocm --extra-index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/
   # RDNA3 iGPUs (gfx1103) may need: export HSA_OVERRIDE_GFX_VERSION=11.0.0
   ```

   The engine is picked automatically (ORT GPU EP → ORT CPU → OpenCV DNN) and
   can be forced with `HOWDY_ONNX_ENGINE=ort|ort-cpu|cv-vulkan|cv-cpu`. The
   chosen backend is printed at startup.

   *Why not Vulkan?* OpenCV 5's new DNN graph engine currently ignores
   `setPreferableBackend` (it warns and runs on CPU), and its classic engine
   cannot parse SCRFD's dynamic-shape nodes — where it does run (ArcFace) it
   measured ~5x slower than the CPU path on this machine. Until OpenCV's
   Vulkan backend matures, ONNX Runtime with the ROCm EP is the supported
   iGPU route; `cv-vulkan` remains available as a forced experimental option.

## Usage

```sh
python3 compare_onnx_prototype.py <user> --enroll   # look at the IR camera, stores 512-D vector
python3 compare_onnx_prototype.py <user>            # authenticate (exit code 0 on success)
python3 compare_onnx_prototype.py <user> --test     # live similarity/liveness readout
```

The IR camera defaults to `/dev/video2` (8-bit `GREY`, 640x360); override with
`--device` or `HOWDY_IR_DEVICE`.

Exit codes follow `compare.py` (0 ok, 10 no model, 11 timeout/no match,
12 no username, 13 all frames dark) plus **14 = presentation attack suspected**
(the face matched but liveness rejected it) so the PAM layer can message this
distinctly.

## Presentation attack detection

There is no `Z16` depth stream on Linux, so liveness is derived from the 2D IR
stream with two independent signals that must both pass:

* **LBP micro-texture** — uniform LBP(8,1) histogram of the aligned IR crop.
  Skin under active IR illumination exhibits subsurface scattering with a broad
  texture spectrum; paper is flat, and phone/laptop screens emit almost no IR
  (they appear black to the IR camera). Train the proper classifier with:

  ```sh
  python3 train_liveness_onnx.py --capture onnx-data/pad/real  --seconds 30   # your face
  python3 train_liveness_onnx.py --capture onnx-data/pad/spoof --seconds 30   # prints/screens
  python3 train_liveness_onnx.py --real onnx-data/pad/real --spoof onnx-data/pad/spoof
  ```

  This writes `onnx-data/liveness_svm.xml`, which the prototype loads
  automatically. Until then a conservative entropy/sharpness heuristic is used
  (a warning is printed).

* **Landmark parallax** — the 5 landmarks are tracked across frames; a rigid
  similarity transform is fitted to the outer four points (eyes + mouth) and
  the nose tip's deviation from that planar prediction is measured. A flat
  photo moves as a plane (near-zero residual); a real face shows perspective
  parallax under natural head micro-motion. Requires a few frames of visible
  motion before it will vote, so authentication takes at least ~4-5 frames.

`--no-pad` disables both checks for debugging. Thresholds are class constants
on `LivenessAnalyzer` and are deliberately conservative defaults — tune them
with `--test` against your own spoof media before trusting them.

## PAM integration plan

The prototype mirrors `compare.py`'s contract (argv[1] = username, exit-code
protocol) so the C PAM wrapper needs no changes beyond invoking this script
and mapping exit code 14 to a "spoof suspected" message. Remaining work before
it can replace `compare.py`:

* store enrollments in `paths_factory.user_model_path()` format (versioned,
  multiple models per user) instead of `onnx-data/enroll_<user>.json`
* read thresholds/device from Howdy's `config.ini` instead of CLI flags
* wire `howdy-gtk` status messages and snapshot capture back in
* package the ONNX weights via `onnx-data/install.sh` the same way
  `dlib-data/install.sh` is handled by the meson build
