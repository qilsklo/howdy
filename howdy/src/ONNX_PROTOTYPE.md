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

   To move inference onto the Radeon 780M, install the ROCm stack plus the
   MIGraphX build of ONNX Runtime (for ROCm 7.x AMD ships the MIGraphX
   execution provider; the old ROCm EP wheels target ROCm 6.x). The wheel's
   `rocm-rel-*` index version **must match the system ROCm version**, only
   cp310/cp312 wheels exist (hence the 3.12 venv), and on glibc >= 2.41 the
   wheel needs an execstack fix before it will import:

   ```sh
   sudo pacman -S --needed rocm-hip-runtime migraphx     # Arch; match rel index below
   python3.12 -m venv ~/.venvs/howdy-rocm
   ~/.venvs/howdy-rocm/bin/pip install numpy opencv-python
   ~/.venvs/howdy-rocm/bin/pip install onnxruntime-migraphx \
       --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/ \
       --extra-index-url https://pypi.org/simple
   python3 onnx-data/fix-execstack.py ~/.venvs/howdy-rocm
   # RDNA3 iGPUs (gfx1103) are not an official ROCm target:
   export HSA_OVERRIDE_GFX_VERSION=11.0.0
   ```

   MIGraphX compiles GPU kernels at session creation, which takes minutes
   for these models. The engine passes `migraphx_model_cache_dir`
   (`onnx-data/mxr-cache/`) so this cost is paid once; subsequent starts
   load the compiled programs in under a second. The cache is tied to the
   GPU/driver combination — delete it after ROCm upgrades if inference
   misbehaves.

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

## Integration plan (WebAuthn daemon)

The consumer of this pipeline is `howdy-webauthn`, the virtual FIDO2
authenticator: face verification gates passkey assertions. The prototype
mirrors `compare.py`'s contract (argv[1] = username, exit-code protocol), so
the internal verification API can shell out to it — or import it — the same
way it drives `compare.py`, mapping exit code 14 to a distinct
"presentation attack suspected" UV failure.

### Running the pipeline on the GPU inside the daemon

`howdy-webauthn.service` is hardened, and as shipped the GPU is not reachable
from inside it. A drop-in (`sudo systemctl edit howdy-webauthn`) needs:

```ini
[Service]
# ROCm environment: gfx1103 override must live here, not in a shell profile,
# because systemd services inherit nothing from user shells
Environment=HSA_OVERRIDE_GFX_VERSION=11.0.0
# ROCm compute + graphics device nodes (root bypasses the render group,
# but DeviceAllow still filters them)
DeviceAllow=/dev/kfd rw
DeviceAllow=char-drm rw
```

Two more hardening directives can bite and may need relaxing *only if* GPU
init fails in the journal: `MemoryDenyWriteExecute=true` (GPU runtimes map
code buffers) and `ProtectHome=true` (blocks a venv living under `/home` —
for production, install the ONNX runtime somewhere system-wide instead).
The engine falls back to the CPU provider automatically in both cases, so a
misconfigured GPU degrades speed, never availability.

### Remaining work

* store enrollments in `paths_factory.user_model_path()` format (versioned,
  multiple models per user) instead of `onnx-data/enroll_<user>.json`
* read thresholds/device from Howdy's `config.ini` (`[webauthn]`) instead of
  CLI flags
* call the verification routine in-process from the daemon so the models
  stay loaded between assertions (~0.3s saved per authentication)
* package the ONNX weights via `onnx-data/install.sh` the same way
  `dlib-data/install.sh` is handled by the meson build
