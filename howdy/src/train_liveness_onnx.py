# Train the LBP micro-texture liveness classifier for the ONNX prototype
#
# Collect training data with the built-in capture mode, pointing the IR camera
# at a real face and then at spoof media (printed photos, phone/laptop screens):
#
#   python3 train_liveness_onnx.py --capture onnx-data/pad/real  --seconds 30
#   python3 train_liveness_onnx.py --capture onnx-data/pad/spoof --seconds 30
#
# Then train the SVM that compare_onnx_prototype.py picks up automatically:
#
#   python3 train_liveness_onnx.py --real onnx-data/pad/real --spoof onnx-data/pad/spoof

import argparse
import os
import sys
import time

import cv2
import numpy as np

from onnx_face import (
	DATA_DIR, SCRFD, ArcFace, LivenessAnalyzer, Settings, find_model,
	open_camera, read_gray)

SVM_PATH = os.path.join(DATA_DIR, "liveness_svm.xml")


def aligned_crops(detector, image):
	"""Yield aligned 112x112 grayscale crops for every face in a BGR image"""
	for _bbox, _score, kps in detector.detect(image):
		aligned = ArcFace.align(image, kps)
		yield cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)


def capture(detector, out_dir, device, seconds):
	"""Save aligned face crops from the IR camera into out_dir"""
	os.makedirs(out_dir, exist_ok=True)
	settings = Settings.from_config()
	if device:
		settings.device_path = device
	cap = open_camera(settings)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	start = time.time()
	saved = 0
	while time.time() - start < seconds:
		gray = read_gray(cap)
		if gray is None:
			continue
		bgr = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
		for crop in aligned_crops(detector, bgr):
			path = os.path.join(out_dir, "crop_%d_%03d.png" % (int(start), saved))
			cv2.imwrite(path, crop)
			saved += 1
	print("Saved %d aligned crops to %s" % (saved, out_dir))


def load_features(detector, directory):
	"""LBP histograms plus the capture session each image came from.

	Frames within one capture run are nearly identical, so accuracy must be
	measured on a session that contributed no training data at all. Session
	identity comes from the timestamp --capture puts in the filename.
	"""
	feats, sessions = [], []
	for name in sorted(os.listdir(directory)):
		img = cv2.imread(os.path.join(directory, name))
		if img is None:
			continue
		parts = name.split("_")
		session = parts[1] if name.startswith("crop_") and len(parts) > 2 else "other"
		if img.shape[:2] == (112, 112):
			crops = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]
		else:
			crops = list(aligned_crops(detector, img))
		for crop in crops:
			feats.append(LivenessAnalyzer.lbp_histogram(crop))
			sessions.append(session)
	return feats, sessions


def split_holdout(feats, sessions):
	"""Return (train, test) holding out the newest capture session if possible"""
	groups = sorted(set(sessions))
	if len(groups) >= 2:
		held = groups[-1]
		train = [f for f, s in zip(feats, sessions) if s != held]
		test = [f for f, s in zip(feats, sessions) if s == held]
		return train, test, True
	# Single session: a random split is all we can do, but consecutive video
	# frames are heavily correlated so the resulting number is optimistic
	rng = np.random.default_rng(42)
	order = rng.permutation(len(feats))
	cut = max(1, len(feats) // 5)
	test = [feats[i] for i in order[:cut]]
	train = [feats[i] for i in order[cut:]]
	return train, test, False


def train_svm(real, spoof):
	samples = np.array(real + spoof, dtype=np.float32)
	labels = np.array([1] * len(real) + [-1] * len(spoof), dtype=np.int32)
	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_RBF)
	svm.trainAuto(samples, cv2.ml.ROW_SAMPLE, labels)
	return svm


def recall(svm, feats, expected):
	if not feats:
		return float("nan")
	pred = svm.predict(np.array(feats, dtype=np.float32))[1].ravel()
	return float(np.mean(pred == expected))


def main():
	parser = argparse.ArgumentParser(description="Train the IR liveness SVM")
	parser.add_argument("--capture", metavar="DIR", help="capture aligned crops into DIR instead of training")
	parser.add_argument("--seconds", type=float, default=30.0)
	parser.add_argument("--device", help="override the IR camera device path")
	parser.add_argument("--real", metavar="DIR", help="directory of genuine-face images/crops")
	parser.add_argument("--spoof", metavar="DIR", help="directory of spoof images/crops")
	args = parser.parse_args()

	detector = SCRFD(find_model("det_", "scrfd"))

	if args.capture:
		capture(detector, args.capture, args.device, args.seconds)
		return

	if not args.real or not args.spoof:
		parser.error("either --capture DIR, or both --real and --spoof")

	real, real_sessions = load_features(detector, args.real)
	spoof, spoof_sessions = load_features(detector, args.spoof)
	print("Features: %d real (%d sessions), %d spoof (%d sessions)" % (
		len(real), len(set(real_sessions)), len(spoof), len(set(spoof_sessions))))
	if len(real) < 20 or len(spoof) < 20:
		sys.exit("Need at least 20 samples per class for a usable classifier")

	# Evaluate on held-out data first, then retrain on everything for the
	# model that actually gets saved
	real_train, real_test, real_by_session = split_holdout(real, real_sessions)
	spoof_train, spoof_test, spoof_by_session = split_holdout(spoof, spoof_sessions)
	eval_svm = train_svm(real_train, spoof_train)
	live_recall = recall(eval_svm, real_test, 1)
	spoof_recall = recall(eval_svm, spoof_test, -1)

	kind = "held-out session" if real_by_session and spoof_by_session else "random 20% split"
	print("Evaluation on %s (%d real / %d spoof samples):" % (kind, len(real_test), len(spoof_test)))
	print("  live faces accepted:  %.1f%%" % (live_recall * 100))
	print("  spoofs rejected:      %.1f%%" % (spoof_recall * 100))
	if not (real_by_session and spoof_by_session):
		print("  WARNING: only one capture session per class; consecutive video")
		print("  frames are near-duplicates, so this estimate is OPTIMISTIC.")
		print("  Capture more sessions (different lighting, distances, spoof")
		print("  media) and retrain for a number you can trust.")

	svm = train_svm(real, spoof)
	svm.save(SVM_PATH)
	print("Saved " + SVM_PATH + " (trained on all %d samples)" % (len(real) + len(spoof)))
	print("Final check: run compare_onnx_prototype.py --test against your real spoof media")


if __name__ == "__main__":
	main()
