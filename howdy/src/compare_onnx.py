# CLI for the ONNX face pipeline, mirroring compare.py's contract:
# argv[1] is the username, the exit code is the verification status
# (see onnx_face.STATUS_* / verification.VerificationResult).
#
#   python3 compare_onnx.py <user> --enroll   capture + store an enrollment
#   python3 compare_onnx.py <user>            authenticate (exit 0 on success)
#   python3 compare_onnx.py <user> --test     live similarity/liveness readout

import argparse
import sys
import time

import onnx_face


def main():
	parser = argparse.ArgumentParser(description="Howdy ONNX face verification")
	parser.add_argument("user", nargs="?", help="username to enroll or verify")
	parser.add_argument("--enroll", action="store_true", help="capture and store an enrollment")
	parser.add_argument("--label", default="onnx", help="label for the stored enrollment")
	parser.add_argument("--test", action="store_true", help="live readout without authenticating")
	parser.add_argument("--device", help="override the IR camera device path")
	parser.add_argument("--timeout", type=float, help="seconds before giving up")
	parser.add_argument("--certainty", type=float, help="cosine similarity threshold")
	parser.add_argument("--no-pad", action="store_true", help="disable liveness checks (debugging only)")
	args = parser.parse_args()

	if not args.user:
		sys.exit(onnx_face.STATUS_ABORT)

	settings = onnx_face.Settings.from_config()
	if args.device:
		settings.device_path = args.device
	if args.timeout:
		settings.timeout = args.timeout
	if args.certainty:
		settings.certainty = args.certainty
	if args.no_pad:
		settings.pad = False

	start = time.time()
	pipeline = onnx_face.FacePipeline(settings)
	print("Backends: %s (loaded in %.2fs)" % (pipeline.backends(), time.time() - start))
	if settings.pad:
		print("PAD texture engine: " + pipeline.make_liveness().texture_mode())

	if args.enroll:
		status, path = pipeline.enroll_user(
			args.user, label=args.label, timeout=settings.timeout + 25)
		if status == onnx_face.STATUS_SUCCESS:
			print("Enrolled %s -> %s" % (args.user, path))
		else:
			print("Enrollment failed with status %d" % status)
		sys.exit(status)

	if args.test:
		settings.timeout = args.timeout or 30.0

		def report(similarity, liveness):
			print("sim=%.3f texture=%s parallax=%s (%s)" % (
				similarity, liveness.texture_ok(), liveness.parallax_ok(),
				liveness.texture_mode()))

		status = pipeline.verify_user(args.user, on_frame=report)
		print("status: %d" % status)
		sys.exit(status)

	status = pipeline.verify_user(args.user)
	scan = time.time() - start
	if status == onnx_face.STATUS_SUCCESS:
		print("Verified %s in %.2fs total" % (args.user, scan))
	elif status == onnx_face.STATUS_PRESENTATION_ATTACK:
		print("Face matched but liveness rejected the attempt")
	else:
		print("Verification failed with status %d after %.2fs" % (status, scan))
	sys.exit(status)


if __name__ == "__main__":
	main()
