# Modern ONNX-based face recognition pipeline for Howdy
#
# Replaces the dlib HOG/CNN + Euclidean pipeline in compare.py with:
#   1. Liveness / PAD on the IR stream: an anti-spoofing CNN (MiniFASNet
#      architecture) or LBP micro-texture classifier, plus landmark-parallax
#      analysis that separates flat spoof media from real 3D faces
#   2. SCRFD face detection with 5-point landmarks (InsightFace)
#   3. Similarity-transform alignment to the canonical 112x112 ArcFace crop
#   4. ArcFace 512-D embedding + cosine similarity verification
#
# Inference runs through ONNX Runtime (ROCm/MIGraphX execution providers when
# available) or OpenCV DNN, so an AMD iGPU can do the heavy lifting.
#
# This module is a library: the CLI lives in compare_onnx.py and the
# howdy-webauthn daemon calls verify_user() in-process via verification.py.

import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

# Directory holding the ONNX weights, next to this script
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx-data")

# Exit/status codes, matching compare.py and pam/main.hh (VerificationResult)
STATUS_SUCCESS = 0
STATUS_NO_FACE_MODEL = 10
STATUS_TIMEOUT_REACHED = 11
STATUS_ABORT = 12
STATUS_TOO_DARK = 13
# New in the ONNX pipeline: the face matched but liveness rejected the attempt
# (14 and 15 are already taken by INVALID_DEVICE and RUBBERSTAMP)
STATUS_PRESENTATION_ATTACK = 16

# Canonical 5-point destination template used by ArcFace for a 112x112 crop
ARCFACE_DST = np.array([
	[38.2946, 51.6963],  # left eye
	[73.5318, 51.5014],  # right eye
	[56.0252, 71.7366],  # nose tip
	[41.5493, 92.3655],  # left mouth corner
	[70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)


class Settings:
	"""Pipeline configuration, populated from Howdy's config.ini when present.

	Every value can also be set directly, which is what the CLI flags do.
	"""

	def __init__(self):
		self.device_path = os.environ.get("HOWDY_IR_DEVICE", "/dev/video2")
		self.timeout = 5.0
		self.dark_threshold = 60.0
		self.certainty = 0.40           # minimum cosine similarity
		self.detection_confidence = 0.5
		self.pad = True                 # require liveness checks
		self.pad_threshold = 0.60       # minimum live probability from the CNN
		# Texture engine: "auto" uses the trained LBP SVM when present, else
		# the heuristic. "cnn" opts into the RGB anti-spoofing CNN, which we
		# measured to have no discriminative power on active-IR imagery (it
		# scores real and spoofed IR faces alike) - only use it on RGB cameras
		self.pad_engine = "auto"
		self.frame_width = 640
		self.frame_height = 360

	@classmethod
	def from_config(cls, config=None):
		"""Build settings from a ConfigParser (loads Howdy's config if None)"""
		settings = cls()
		if config is None:
			config = load_config()
		if config is None:
			return settings
		device = config.get("video", "device_path", fallback="").strip()
		# The stock config ships with the placeholder "none"
		if device and device.lower() != "none":
			settings.device_path = device
		settings.timeout = config.getfloat("video", "timeout", fallback=settings.timeout)
		settings.dark_threshold = config.getfloat("video", "dark_threshold", fallback=settings.dark_threshold)
		settings.certainty = config.getfloat("onnx", "certainty", fallback=settings.certainty)
		settings.detection_confidence = config.getfloat("onnx", "detection_confidence", fallback=settings.detection_confidence)
		settings.pad = config.getboolean("onnx", "pad", fallback=settings.pad)
		settings.pad_threshold = config.getfloat("onnx", "pad_threshold", fallback=settings.pad_threshold)
		settings.pad_engine = config.get("onnx", "pad_engine", fallback=settings.pad_engine).strip().lower()
		return settings


def load_config():
	"""Read Howdy's config.ini, or None if it cannot be found"""
	import configparser
	try:
		import paths_factory
		path = paths_factory.config_file_path()
	except ImportError:
		path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
	if not os.path.isfile(path):
		return None
	config = configparser.ConfigParser()
	config.read(path)
	return config


def user_model_path(user):
	"""Path of a user's ONNX face model, beside Howdy's dlib model store.

	Falls back to onnx-data/ in a development checkout where the generated
	paths module does not exist.
	"""
	try:
		import paths_factory
		return os.path.join(str(paths_factory.user_models_dir_path()), "onnx", user + ".dat")
	except ImportError:
		return os.path.join(DATA_DIR, "enroll_" + user + ".json")


def load_user_models(user):
	"""Return the list of stored model dicts for a user, or None"""
	try:
		with open(user_model_path(user)) as f:
			models = json.load(f)
	except (FileNotFoundError, ValueError):
		return None
	# Earlier prototype enrollments stored a single dict
	if isinstance(models, dict):
		models = [{"label": "prototype", "id": 0, "data": [models["embedding"]]}]
	return models if models else None


def save_user_model(user, embedding, label):
	"""Append an enrollment to the user's model store, compare.py style"""
	models = load_user_models(user) or []
	models.append({
		"time": int(time.time()),
		"label": label,
		"id": max((m.get("id", 0) for m in models), default=-1) + 1,
		"engine": "arcface",
		"data": [[round(float(x), 7) for x in embedding]],
	})
	path = user_model_path(user)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as f:
		json.dump(models, f)
	return path


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


class DeepPAD:
	"""Anti-spoofing CNN (MiniFASNet architecture, Silent-Face family).

	Takes the full frame plus the detected bounding box, crops a square
	region 1.5x the face size (the network was trained to see background
	context around the face) and returns the probability that the face is
	live rather than a print/replay attack. Class 0 of the softmax output
	is "live" in these models.
	"""

	INPUT_SIZE = 128
	BBOX_INC = 1.5

	def __init__(self, model_path):
		self.engine = InferenceEngine(model_path)

	@classmethod
	def crop(cls, img, bbox):
		"""Square crop of BBOX_INC times the face size, padded at the borders"""
		x1, y1, x2, y2 = bbox[:4]
		side = int(max(x2 - x1, y2 - y1) * cls.BBOX_INC)
		cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
		x, y = int(cx - side / 2), int(cy - side / 2)
		h, w = img.shape[:2]
		cx1, cy1 = max(x, 0), max(y, 0)
		cx2, cy2 = min(x + side, w), min(y + side, h)
		crop = img[cy1:cy2, cx1:cx2]
		# Pad the parts of the square that fell outside the frame
		return cv2.copyMakeBorder(
			crop, cy1 - y, y + side - cy2, cx1 - x, x + side - cx2,
			cv2.BORDER_CONSTANT, value=[0, 0, 0])

	def live_probability(self, img, bbox):
		"""P(live) for the face at bbox in a BGR frame"""
		crop = self.crop(img, bbox)
		size = self.INPUT_SIZE
		scale = size / max(crop.shape[:2])
		resized = cv2.resize(crop, None, fx=scale, fy=scale)
		padded = np.zeros((size, size, 3), dtype=np.uint8)
		top = (size - resized.shape[0]) // 2
		left = (size - resized.shape[1]) // 2
		padded[top:top + resized.shape[0], left:left + resized.shape[1]] = resized

		blob = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
		logits = self.engine.run(blob)[0].ravel()
		exp = np.exp(logits - logits.max())
		return float(exp[0] / exp.sum())


class LivenessAnalyzer:
	"""Presentation attack detection from the 2D IR stream only.

	Two independent signals, both must pass:
	  * Texture/appearance: the anti-spoofing CNN when its weights are
	    installed (preferred), otherwise a trained LBP SVM
	    (onnx-data/liveness_svm.xml), otherwise a conservative entropy +
	    sharpness heuristic on the LBP histogram.
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

	def __init__(self, pad_threshold=0.60, cnn=None):
		self.kps_history = deque(maxlen=16)
		self.texture_votes = deque(maxlen=16)
		self.parallax_ratios = []
		self.pad_threshold = pad_threshold

		self.cnn = cnn
		self.svm = None
		if self.cnn is None:
			svm_path = os.path.join(DATA_DIR, "liveness_svm.xml")
			if os.path.isfile(svm_path):
				try:
					self.svm = cv2.ml.SVM_load(svm_path)
				except AttributeError:
					# PyPI opencv-python 5.x dropped cv2.ml (it moved to the
					# contrib packages); run the heuristic rather than dying
					print("This OpenCV build has no cv2.ml, LBP SVM disabled "
						  "(install opencv-python<5 or opencv-contrib-python)")

	def texture_mode(self):
		if self.cnn is not None:
			return "cnn"
		return "svm" if self.svm is not None else "heuristic"

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

	def update(self, frame, bbox, aligned_gray, kps):
		"""Feed one frame: full BGR frame, face bbox, aligned crop, landmarks"""
		if self.cnn is not None:
			live = self.cnn.live_probability(frame, bbox) >= self.pad_threshold
		elif self.svm is not None:
			hist = self.lbp_histogram(aligned_gray)
			live = self.svm.predict(hist[None, :])[1].ravel()[0] > 0
		else:
			hist = self.lbp_histogram(aligned_gray)
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


class FacePipeline:
	"""The loaded models, reusable across verifications.

	Construction is expensive (session creation, GPU kernel cache load);
	daemons should build one instance and keep it for the process lifetime.
	"""

	def __init__(self, settings=None):
		self.settings = settings or Settings.from_config()
		self.detector = SCRFD(
			find_model("det_", "scrfd"),
			conf_thresh=self.settings.detection_confidence)
		self.embedder = ArcFace(find_model("w600k", "arcface"))
		# The anti-spoofing CNN is opt-in: it is trained on RGB imagery and
		# has no discriminative power on active-IR frames
		self.pad_cnn = None
		if self.settings.pad_engine == "cnn":
			cnn_path = os.path.join(DATA_DIR, "antispoof_bin.onnx")
			if os.path.isfile(cnn_path):
				self.pad_cnn = DeepPAD(cnn_path)

	def backends(self):
		return "detector=%s embedder=%s pad=%s" % (
			self.detector.engine.backend, self.embedder.engine.backend,
			self.pad_cnn.engine.backend if self.pad_cnn else "-")

	def make_liveness(self):
		return LivenessAnalyzer(
			pad_threshold=self.settings.pad_threshold, cnn=self.pad_cnn)

	def process_frame(self, bgr):
		"""Detect the largest face, returns (bbox, kps, embedding, aligned) or None"""
		faces = self.detector.detect(bgr)
		if not faces:
			return None
		bbox, _score, kps = max(
			faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
		embedding, aligned = self.embedder.embed(bgr, kps)
		return bbox, kps, embedding, aligned

	def verify_user(self, user, timeout=None, cancel_check=None, on_frame=None):
		"""Authenticate a user against their stored models.

		Returns a STATUS_* code. cancel_check is polled between frames so a
		caller (CTAPHID CANCEL) can abort; on_frame, if given, receives
		(similarity, liveness) per processed face for diagnostics.
		"""
		settings = self.settings
		models = load_user_models(user)
		if not models:
			return STATUS_NO_FACE_MODEL
		encodings = np.array(
			[row for m in models for row in m["data"]], dtype=np.float32)

		liveness = self.make_liveness()
		try:
			camera = open_camera(settings)
		except RuntimeError:
			return STATUS_ABORT
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

		deadline = time.time() + (timeout or settings.timeout)
		frames = valid_frames = 0
		best_similarity = -1.0
		try:
			while time.time() < deadline:
				if cancel_check is not None and cancel_check():
					return STATUS_ABORT
				gray = read_gray(camera)
				if gray is None:
					continue
				frames += 1

				# Skip black/dark frames: IR emitters need a few frames to power up
				hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
				total = float(np.sum(hist))
				darkness = hist[0] / total * 100 if total else 100.0
				if darkness >= settings.dark_threshold:
					continue
				valid_frames += 1

				gray = clahe.apply(gray)
				bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
				result = self.process_frame(bgr)
				if result is None:
					continue
				bbox, kps, embedding, aligned = result
				liveness.update(bgr, bbox, cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), kps)

				similarity = float(np.max(encodings @ embedding))
				best_similarity = max(best_similarity, similarity)
				if on_frame is not None:
					on_frame(similarity, liveness)

				if similarity >= settings.certainty and (not settings.pad or liveness.verdict()):
					return STATUS_SUCCESS
		finally:
			camera.release()

		if frames > 0 and valid_frames == 0:
			return STATUS_TOO_DARK
		if best_similarity >= settings.certainty and settings.pad:
			return STATUS_PRESENTATION_ATTACK
		return STATUS_TIMEOUT_REACHED

	def enroll_user(self, user, label="onnx", samples=12, timeout=30.0, cancel_check=None):
		"""Capture an enrollment for a user, returns (STATUS_*, model_path)"""
		settings = self.settings
		liveness = self.make_liveness()
		try:
			camera = open_camera(settings)
		except RuntimeError:
			return STATUS_ABORT, None
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

		deadline = time.time() + timeout
		embeddings = []
		try:
			while time.time() < deadline:
				if cancel_check is not None and cancel_check():
					return STATUS_ABORT, None
				gray = read_gray(camera)
				if gray is None:
					continue
				gray = clahe.apply(gray)
				bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
				result = self.process_frame(bgr)
				if result is None:
					continue
				bbox, kps, embedding, aligned = result
				liveness.update(bgr, bbox, cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), kps)
				embeddings.append(embedding)

				if len(embeddings) >= samples and (not settings.pad or liveness.verdict()):
					mean = np.mean(embeddings, axis=0)
					mean /= np.linalg.norm(mean)
					return STATUS_SUCCESS, save_user_model(user, mean, label)
		finally:
			camera.release()
		return STATUS_TIMEOUT_REACHED, None


def find_model(*patterns):
	"""Locate an ONNX file in DATA_DIR matching any of the given prefixes.

	Raises RuntimeError rather than exiting: in-process callers (the webauthn
	daemon) must translate this into a verification failure, not die.
	"""
	if os.path.isdir(DATA_DIR):
		for name in sorted(os.listdir(DATA_DIR)):
			if name.endswith(".onnx") and any(name.startswith(p) for p in patterns):
				return os.path.join(DATA_DIR, name)
	raise RuntimeError(
		"Missing ONNX weights in %s, run %s" % (DATA_DIR, os.path.join(DATA_DIR, "install.sh")))


def open_camera(settings):
	"""Open the IR camera in native 8-bit grayscale mode"""
	cap = cv2.VideoCapture(settings.device_path, cv2.CAP_V4L2)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GREY"))
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
	cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
	if not cap.isOpened():
		raise RuntimeError("Cannot open camera " + settings.device_path)
	return cap


def read_gray(cap):
	"""Read one frame and return it as single-channel grayscale, or None"""
	ret, frame = cap.read()
	if not ret or frame is None:
		return None
	if frame.ndim == 3:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	return frame
