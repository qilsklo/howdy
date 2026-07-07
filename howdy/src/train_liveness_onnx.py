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

from compare_onnx_prototype import (
	DATA_DIR, SCRFD, ArcFace, LivenessAnalyzer, find_model, open_camera, read_gray)

SVM_PATH = os.path.join(DATA_DIR, "liveness_svm.xml")


def aligned_crops(detector, image):
	"""Yield aligned 112x112 grayscale crops for every face in a BGR image"""
	for _bbox, _score, kps in detector.detect(image):
		aligned = ArcFace.align(image, kps)
		yield cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)


def capture(detector, out_dir, device, seconds):
	"""Save aligned face crops from the IR camera into out_dir"""
	os.makedirs(out_dir, exist_ok=True)
	cap = open_camera(device)
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
	"""LBP histograms for every image in a directory (aligning full frames)"""
	feats = []
	for name in sorted(os.listdir(directory)):
		img = cv2.imread(os.path.join(directory, name))
		if img is None:
			continue
		if img.shape[:2] == (112, 112):
			crops = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]
		else:
			crops = list(aligned_crops(detector, img))
		for crop in crops:
			feats.append(LivenessAnalyzer.lbp_histogram(crop))
	return feats


def main():
	parser = argparse.ArgumentParser(description="Train the IR liveness SVM")
	parser.add_argument("--capture", metavar="DIR", help="capture aligned crops into DIR instead of training")
	parser.add_argument("--seconds", type=float, default=30.0)
	parser.add_argument("--device", default=os.environ.get("HOWDY_IR_DEVICE", "/dev/video2"))
	parser.add_argument("--real", metavar="DIR", help="directory of genuine-face images/crops")
	parser.add_argument("--spoof", metavar="DIR", help="directory of spoof images/crops")
	args = parser.parse_args()

	detector = SCRFD(find_model("det_", "scrfd"))

	if args.capture:
		capture(detector, args.capture, args.device, args.seconds)
		return

	if not args.real or not args.spoof:
		parser.error("either --capture DIR, or both --real and --spoof")

	real = load_features(detector, args.real)
	spoof = load_features(detector, args.spoof)
	print("Features: %d real, %d spoof" % (len(real), len(spoof)))
	if len(real) < 20 or len(spoof) < 20:
		sys.exit("Need at least 20 samples per class for a usable classifier")

	samples = np.array(real + spoof, dtype=np.float32)
	labels = np.array([1] * len(real) + [-1] * len(spoof), dtype=np.int32)

	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_RBF)
	svm.trainAuto(samples, cv2.ml.ROW_SAMPLE, labels)

	predictions = svm.predict(samples)[1].ravel()
	accuracy = float(np.mean(predictions == labels))
	svm.save(SVM_PATH)
	print("Training accuracy: %.1f%% (validate with held-out spoof attempts!)" % (accuracy * 100))
	print("Saved " + SVM_PATH)


if __name__ == "__main__":
	main()
