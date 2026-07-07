# Prototype of a modern ONNX-based recognition pipeline for Howdy
#
# Replaces the dlib HOG/CNN + Euclidean pipeline in compare.py with:
#   1. Liveness / PAD on the raw IR stream (LBP micro-texture + landmark parallax)
#   2. SCRFD face detection with 5-point landmarks (InsightFace)
#   3. Similarity-transform alignment to the canonical 112x112 ArcFace crop
#   4. ArcFace 512-D embedding + cosine similarity verification
#
# Inference runs through ONNX Runtime (ROCm/MIGraphX execution providers when
# available) or OpenCV DNN on the Vulkan backend, so the AMD Radeon 780M iGPU
# does the heavy lifting instead of the CPU.
#
# Usage:
#   python3 compare_onnx_prototype.py <user> --enroll   capture + store enrollment vector
#   python3 compare_onnx_prototype.py <user>            authenticate against stored vector
#   python3 compare_onnx_prototype.py <user> --test     live similarity readout, no exit codes
#
# Exit codes match compare.py so this can slot into the PAM wrapper later:
#   0 success, 10 no model, 11 timeout, 12 no username, 13 all frames dark,
#   14 presentation attack suspected (new)

import time

timings = {"st": time.time()}

import argparse
import json
import os
import sys
from collections import deque

import cv2
import numpy as np

# Directory holding the ONNX weights and enrollment data, next to this script
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx-data")

# Canonical 5-point destination template used by ArcFace for a 112x112 crop
ARCFACE_DST = np.array([
	[38.2946, 51.6963],  # left eye
	[73.5318, 51.5014],  # right eye
	[56.0252, 71.7366],  # nose tip
	[41.5493, 92.3655],  # left mouth corner
	[70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)


def build_uniform_lbp_lut():
	"""Map the 256 raw LBP codes to the 59 uniform-pattern bins"""
	lut = np.zeros(256, dtype=np.uint8)
	next_bin = 0
	for code in range(256):
		bits = [(code >> i) & 1 for i in range(8)]
		transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
		if transitions <= 2:
			lut[code] = next_bin
			next_bin += 1
		else:
			lut[code] = 58
	return lut


UNIFORM_LBP_LUT = build_uniform_lbp_lut()


class InferenceEngine:
	"""Run an ONNX model on the best available backend.

	Priority: ONNX Runtime with an AMD GPU execution provider (MIGraphX/ROCm)
	when the ROCm stack is installed, then ONNX Runtime on CPU, then OpenCV
	DNN. Override with HOWDY_ONNX_ENGINE=ort|ort-cpu|cv-vulkan|cv-cpu.

	Note on Vulkan: OpenCV 5's new DNN graph engine currently ignores
	setPreferableBackend, and the classic engine cannot parse SCRFD's dynamic
	shapes, so cv-vulkan is only honored when forced and may silently run on
	the CPU. The supported iGPU path is ONNX Runtime's ROCm/MIGraphX EP.
	"""

	def __init__(self, model_path):
		self.model_path = model_path
		self.backend = None
		forced = os.environ.get("HOWDY_ONNX_ENGINE", "")

		if forced in ("", "ort", "ort-cpu"):
			try:
				import onnxruntime as ort
				available = ort.get_available_providers()
				wanted = ["MIGraphXExecutionProvider", "ROCMExecutionProvider"]
				gpu = [] if forced == "ort-cpu" else [p for p in wanted if p in available]
				# MIGraphX compiles GPU kernels at session creation, which
				# takes minutes for these models; a cache dir makes every
				# start after the first load in well under a second
				providers = []
				for p in gpu:
					if p == "MIGraphXExecutionProvider":
						cache_dir = os.path.join(DATA_DIR, "mxr-cache")
						os.makedirs(cache_dir, exist_ok=True)
						providers.append((p, {"migraphx_model_cache_dir": cache_dir}))
					else:
						providers.append(p)
				providers.append("CPUExecutionProvider")
				options = ort.SessionOptions()
				options.log_severity_level = 4
				try:
					self.session = ort.InferenceSession(
						model_path, sess_options=options, providers=providers)
				except Exception:
					# GPU provider failed to initialize (missing ROCm libs,
					# unsupported gfx target, ...): retry on the CPU alone
					if not gpu:
						raise
					gpu = []
					self.session = ort.InferenceSession(
						model_path, sess_options=options,
						providers=["CPUExecutionProvider"])
				self.input_name = self.session.get_inputs()[0].name
				self.backend = "ort:" + self.session.get_providers()[0]
				return
			except ImportError:
				if forced:
					raise

		if forced == "cv-vulkan":
			self.net = cv2.dnn.readNetFromONNX(model_path)
			self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_VKCOM)
			self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_VULKAN)
			self.backend = "opencv:vulkan(experimental)"
		else:
			self.net = cv2.dnn.readNetFromONNX(model_path)
			self.backend = "opencv:cpu"
		self.out_names = self.net.getUnconnectedOutLayersNames()

	def run(self, blob):
		"""Return the list of output tensors for a NCHW float blob"""
		if self.backend.startswith("ort"):
			try:
				return self.session.run(None, {self.input_name: blob})
			except Exception:
				# A GPU provider can also fail at inference time (e.g. a HIP
				# kernel error on an unsupported gfx target). Rebuild the
				# session on the CPU once and keep going rather than failing
				# an authentication attempt.
				if self.session.get_providers()[0] == "CPUExecutionProvider":
					raise
				import onnxruntime as ort
				print("GPU inference failed on %s, falling back to CPU" % self.backend)
				self.session = ort.InferenceSession(
					self.model_path, providers=["CPUExecutionProvider"])
				self.backend = "ort:CPUExecutionProvider(runtime-fallback)"
				return self.session.run(None, {self.input_name: blob})
		self.net.setInput(blob)
		outs = self.net.forward(self.out_names)
		return list(outs) if isinstance(outs, (list, tuple)) else [outs]


class SCRFD:
	"""SCRFD detector returning bounding boxes and 5 facial landmarks"""

	def __init__(self, model_path, conf_thresh=0.5, nms_thresh=0.4, input_size=640):
		self.engine = InferenceEngine(model_path)
		self.conf_thresh = conf_thresh
		self.nms_thresh = nms_thresh
		self.input_size = input_size

	def detect(self, img):
		"""Detect faces in a BGR image, returns [(bbox, score, kps5x2), ...]"""
		size = self.input_size
		scale = size / max(img.shape[:2])
		resized = cv2.resize(img, None, fx=scale, fy=scale)
		padded = np.zeros((size, size, 3), dtype=np.uint8)
		padded[:resized.shape[0], :resized.shape[1]] = resized

		blob = cv2.dnn.blobFromImage(
			padded, 1.0 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
		outputs = self.engine.run(blob)

		# Group the 9 output tensors by anchor count so we don't depend on
		# backend-specific output ordering: per stride there is a (N,1) score,
		# (N,4) bbox and (N,10) keypoint tensor
		groups = {}
		for out in outputs:
			out = out.reshape(out.shape[-2], out.shape[-1]) if out.ndim == 3 else out
			groups.setdefault(out.shape[0], {})[out.shape[1]] = out

		bboxes, scores, kpss = [], [], []
		for stride in (8, 16, 32):
			cells = size // stride
			num_anchors = 2
			n = cells * cells * num_anchors
			if n not in groups or len(groups[n]) < 3:
				continue
			score = groups[n][1].ravel()
			bbox = groups[n][4] * stride
			kps = groups[n][10] * stride

			centers = np.stack(np.mgrid[:cells, :cells][::-1], axis=-1)
			centers = (centers * stride).reshape(-1, 2)
			centers = np.repeat(centers, num_anchors, axis=0).astype(np.float32)

			keep = score >= self.conf_thresh
			if not np.any(keep):
				continue
			c, b, k = centers[keep], bbox[keep], kps[keep]

			# distance2bbox: predictions are distances from the anchor center
			boxes = np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1],
							  c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], axis=-1)
			# distance2kps: 5 (x, y) offsets from the anchor center
			pts = k.reshape(-1, 5, 2) + c[:, None, :]

			bboxes.append(boxes)
			scores.append(score[keep])
			kpss.append(pts)

		if not bboxes:
			return []

		bboxes = np.concatenate(bboxes) / scale
		scores = np.concatenate(scores)
		kpss = np.concatenate(kpss) / scale

		wh = np.stack([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]], axis=-1)
		rects = np.concatenate([bboxes[:, :2], wh], axis=-1)
		keep = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh)
		keep = np.array(keep).ravel()
		return [(bboxes[i], float(scores[i]), kpss[i].astype(np.float32)) for i in keep]


class ArcFace:
	"""ArcFace embedder producing L2-normalized 512-D vectors"""

	def __init__(self, model_path):
		self.engine = InferenceEngine(model_path)

	@staticmethod
	def align(img, kps):
		"""Warp a face to the canonical 112x112 crop with a similarity transform"""
		matrix, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
		return cv2.warpAffine(img, matrix, (112, 112), borderValue=0)

	def embed(self, img, kps):
		"""Return (embedding, aligned_crop) for a face given its 5 landmarks"""
		aligned = self.align(img, kps)
		blob = cv2.dnn.blobFromImage(
			aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
		embedding = self.engine.run(blob)[0].ravel().astype(np.float32)
		return embedding / (np.linalg.norm(embedding) + 1e-10), aligned


class LivenessAnalyzer:
	"""Presentation attack detection from the 2D IR stream only.

	Two independent signals, both must pass:
	  * Micro-texture: uniform LBP histogram of the aligned IR crop. Skin under
	    active IR illumination scatters subsurface light and produces a broad
	    texture distribution; paper prints and screens are flat or emit almost
	    no IR. A trained SVM (onnx-data/liveness_svm.xml) is used when present,
	    otherwise a conservative entropy + sharpness heuristic.
	  * Parallax: track the 5 landmarks over consecutive frames, fit the rigid
	    similarity transform on the outer 4 points (eyes + mouth) and measure
	    how far the nose tip deviates from that planar prediction. A flat photo
	    moves as a plane (residual ~ 0), a real 3D face shows perspective
	    distortion when the head micro-rotates.
	"""

	# Tunable defaults, expressed in interocular-distance units where relevant
	MOTION_FLOOR = 0.010      # minimum inter-frame motion to carry parallax signal
	PARALLAX_RATIO = 0.05     # nose residual / motion ratio separating flat from 3D
	MIN_MOTION_PAIRS = 3      # frame pairs with usable motion before judging
	LBP_ENTROPY_MIN = 4.1     # bits; flat prints fall well below live skin
	SHARPNESS_MIN = 18.0      # variance of Laplacian; screen replays in IR are dim/blurred

	def __init__(self):
		self.kps_history = deque(maxlen=16)
		self.texture_votes = deque(maxlen=16)
		self.parallax_ratios = []
		self.svm = None
		svm_path = os.path.join(DATA_DIR, "liveness_svm.xml")
		if os.path.isfile(svm_path):
			self.svm = cv2.ml.SVM_load(svm_path)

	@staticmethod
	def lbp_histogram(gray):
		"""Uniform LBP(8,1) histogram, normalized to a distribution"""
		g = gray.astype(np.int16)
		center = g[1:-1, 1:-1]
		codes = np.zeros(center.shape, dtype=np.uint8)
		offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
		for i, (dy, dx) in enumerate(offsets):
			neighbor = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
			codes |= ((neighbor >= center).astype(np.uint8) << i)
		hist = np.bincount(UNIFORM_LBP_LUT[codes.ravel()], minlength=59).astype(np.float32)
		return hist / (hist.sum() + 1e-10)

	def update(self, aligned_gray, kps):
		"""Feed one frame's aligned grayscale crop and raw landmark positions"""
		hist = self.lbp_histogram(aligned_gray)
		if self.svm is not None:
			live = self.svm.predict(hist[None, :])[1].ravel()[0] > 0
		else:
			entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
			sharpness = float(cv2.Laplacian(aligned_gray, cv2.CV_64F).var())
			live = entropy >= self.LBP_ENTROPY_MIN and sharpness >= self.SHARPNESS_MIN
		self.texture_votes.append(live)

		self.kps_history.append(kps.copy())
		if len(self.kps_history) < 2:
			return
		prev, curr = self.kps_history[-2], self.kps_history[-1]
		interocular = np.linalg.norm(curr[1] - curr[0]) + 1e-10
		outer = [0, 1, 3, 4]

		matrix, _ = cv2.estimateAffinePartial2D(prev[outer], curr[outer], method=cv2.LMEDS)
		if matrix is None:
			return
		motion = float(np.mean(np.linalg.norm(curr[outer] - prev[outer], axis=1))) / interocular
		if motion < self.MOTION_FLOOR:
			return
		nose_pred = matrix[:, :2] @ prev[2] + matrix[:, 2]
		residual = float(np.linalg.norm(curr[2] - nose_pred)) / interocular
		self.parallax_ratios.append(residual / motion)

	def texture_ok(self):
		votes = list(self.texture_votes)
		return len(votes) >= 3 and sum(votes) > len(votes) / 2

	def parallax_ok(self):
		"""None while there is not enough motion to judge, else the verdict"""
		if len(self.parallax_ratios) < self.MIN_MOTION_PAIRS:
			return None
		return float(np.median(self.parallax_ratios)) >= self.PARALLAX_RATIO

	def verdict(self):
		return self.texture_ok() and self.parallax_ok() is True


def find_model(*patterns):
	"""Locate an ONNX file in DATA_DIR matching any of the given prefixes"""
	if os.path.isdir(DATA_DIR):
		for name in sorted(os.listdir(DATA_DIR)):
			if name.endswith(".onnx") and any(name.startswith(p) for p in patterns):
				return os.path.join(DATA_DIR, name)
	print("Missing ONNX weights in " + DATA_DIR)
	print("Run: " + os.path.join(DATA_DIR, "install.sh"))
	sys.exit(1)


def enrollment_path(user):
	return os.path.join(DATA_DIR, "enroll_" + user + ".json")


def open_camera(device):
	"""Open the IR camera in native 8-bit grayscale mode"""
	cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GREY"))
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
	cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
	if not cap.isOpened():
		print("Cannot open camera " + device)
		sys.exit(1)
	return cap


def read_gray(cap):
	"""Read one frame and return it as single-channel grayscale, or None"""
	ret, frame = cap.read()
	if not ret or frame is None:
		return None
	if frame.ndim == 3:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	return frame


def main():
	parser = argparse.ArgumentParser(description="Howdy ONNX pipeline prototype")
	parser.add_argument("user", nargs="?", help="username to enroll or verify")
	parser.add_argument("--enroll", action="store_true", help="capture and store an enrollment vector")
	parser.add_argument("--test", action="store_true", help="live similarity readout without authenticating")
	parser.add_argument("--device", default=os.environ.get("HOWDY_IR_DEVICE", "/dev/video2"))
	parser.add_argument("--timeout", type=float, default=5.0, help="seconds before giving up")
	parser.add_argument("--certainty", type=float, default=0.40, help="cosine similarity threshold")
	parser.add_argument("--dark-threshold", type=float, default=60.0)
	parser.add_argument("--no-pad", action="store_true", help="disable liveness checks (debugging only)")
	args = parser.parse_args()

	if not args.user:
		sys.exit(12)

	# In verification mode the enrollment must exist before we spin anything up
	enrolled = None
	if not args.enroll:
		try:
			with open(enrollment_path(args.user)) as f:
				stored = json.load(f)
			enrolled = np.array(stored["embedding"], dtype=np.float32)
		except (FileNotFoundError, KeyError, ValueError):
			print("No enrollment for " + args.user + ", run with --enroll first")
			sys.exit(10)

	timings["ll"] = time.time()
	detector = SCRFD(find_model("det_", "scrfd"))
	embedder = ArcFace(find_model("w600k", "arcface"))
	liveness = LivenessAnalyzer()
	timings["ll"] = time.time() - timings["ll"]

	print("Detector backend:  " + detector.engine.backend)
	print("Embedder backend:  " + embedder.engine.backend)
	if liveness.svm is None and not args.no_pad:
		print("PAD texture stage: heuristic mode (no liveness_svm.xml trained yet)")

	timings["ic"] = time.time()
	cap = open_camera(args.device)
	timings["ic"] = time.time() - timings["ic"]

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

	frames = 0
	dark_frames = 0
	valid_frames = 0
	best_similarity = -1.0
	embeddings = []
	timings["fr"] = time.time()

	while time.time() - timings["fr"] < args.timeout:
		gray = read_gray(cap)
		if gray is None:
			continue
		frames += 1

		# Skip black/dark frames the same way compare.py does: IR emitters
		# often need a few frames to power up
		hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
		total = float(np.sum(hist))
		darkness = hist[0] / total * 100 if total else 100.0
		if darkness >= args.dark_threshold:
			dark_frames += 1
			continue
		valid_frames += 1

		gray = clahe.apply(gray)
		bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

		faces = detector.detect(bgr)
		if not faces:
			continue
		# Largest face wins
		bbox, score, kps = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))

		embedding, aligned = embedder.embed(bgr, kps)
		liveness.update(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), kps)

		if args.enroll:
			embeddings.append(embedding)
			if len(embeddings) >= 12 and (args.no_pad or liveness.verdict()):
				mean = np.mean(embeddings, axis=0)
				mean /= np.linalg.norm(mean)
				os.makedirs(DATA_DIR, exist_ok=True)
				with open(enrollment_path(args.user), "w") as f:
					json.dump({
						"user": args.user,
						"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
						"model": os.path.basename(embedder.engine.model_path),
						"samples": len(embeddings),
						"embedding": [round(float(x), 7) for x in mean],
					}, f)
				print("Enrolled %s from %d samples -> %s" % (args.user, len(embeddings), enrollment_path(args.user)))
				sys.exit(0)
			continue

		similarity = float(np.dot(embedding, enrolled))
		best_similarity = max(best_similarity, similarity)

		if args.test:
			print("sim=%.3f texture=%s parallax=%s" % (similarity, liveness.texture_ok(), liveness.parallax_ok()))
			continue

		if similarity >= args.certainty and (args.no_pad or liveness.verdict()):
			total_time = time.time() - timings["st"]
			print("Match for %s: similarity %.3f >= %.2f" % (args.user, similarity, args.certainty))
			print("Liveness: texture ok, parallax ratio %.3f" % (np.median(liveness.parallax_ratios) if liveness.parallax_ratios else -1))
			print("Frames: %d (%d dark skipped), scan %.2fs, total %.2fs (init %.2fs, cam %.2fs)" % (
				frames, dark_frames, time.time() - timings["fr"], total_time, timings["ll"], timings["ic"]))
			sys.exit(0)

	# Timed out: distinguish darkness, spoof suspicion and plain no-match
	if valid_frames == 0 and frames > 0:
		print("All frames were too dark")
		sys.exit(13)
	if not args.enroll and not args.test and best_similarity >= args.certainty:
		print("Face matched (%.3f) but liveness rejected it: texture=%s parallax=%s" % (
			best_similarity, liveness.texture_ok(), liveness.parallax_ok()))
		sys.exit(14)
	if not args.test:
		print("No match within %.1fs (best similarity %.3f)" % (args.timeout, best_similarity))
	sys.exit(11)


if __name__ == "__main__":
	main()
