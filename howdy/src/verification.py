# Internal face verification API
# Wraps the compare.py subprocess contract also used by the PAM module,
# so callers get behaviour identical to PAM authentication

import enum
import os
import subprocess
import sys
import threading


class VerificationResult(enum.IntEnum):
	"""Exit status codes of the compare process, mirrors CompareError in pam/main.hh"""
	SUCCESS = 0
	NO_FACE_MODEL = 10
	TIMEOUT_REACHED = 11
	ABORT = 12
	TOO_DARK = 13
	INVALID_DEVICE = 14
	RUBBERSTAMP = 15
	# ONNX pipeline only: the face matched but liveness rejected the attempt
	PRESENTATION_ATTACK = 16
	# Not produced by compare.py itself: the caller cancelled or hard-killed it
	CANCELLED = 250
	UNKNOWN_ERROR = 251

	@classmethod
	def from_exit_code(cls, code):
		"""Map a compare.py exit status to a VerificationResult"""
		try:
			return cls(code)
		except ValueError:
			return cls.UNKNOWN_ERROR


class FaceVerifier:
	"""Runs Howdy face verification for a user by spawning the compare script

	The compare script only communicates through its exit code, which keeps
	this wrapper equivalent to the PAM module and free of side effects on it.
	A verifier instance runs at most one verification at a time and can be
	cancelled from another thread (used by the CTAPHID CANCEL command).
	"""

	def __init__(self, compare_script=None, python_executable=None):
		self.compare_script = compare_script or os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.py")
		self.python_executable = python_executable or sys.executable
		self._process = None
		self._lock = threading.Lock()
		self._cancelled = False

	def verify(self, user, timeout=None):
		"""Verify the face of the given user, blocking until a result is known

		timeout is a hard kill deadline in seconds on top of the internal
		timeout compare.py already applies from [video] timeout in the config.
		"""
		if not user or user == "root":
			return VerificationResult.ABORT

		with self._lock:
			if self._process is not None:
				# A verification is already running on this instance
				return VerificationResult.ABORT
			self._cancelled = False
			try:
				self._process = subprocess.Popen(
					[self.python_executable, self.compare_script, user],
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL)
			except OSError:
				self._process = None
				return VerificationResult.UNKNOWN_ERROR
			process = self._process

		try:
			exit_code = process.wait(timeout=timeout)
		except subprocess.TimeoutExpired:
			process.terminate()
			try:
				process.wait(timeout=2)
			except subprocess.TimeoutExpired:
				process.kill()
				process.wait()
			exit_code = VerificationResult.TIMEOUT_REACHED
		finally:
			with self._lock:
				self._process = None
				cancelled = self._cancelled

		if cancelled:
			return VerificationResult.CANCELLED

		# A negative return code means the process died from a signal
		if exit_code < 0:
			return VerificationResult.ABORT

		return VerificationResult.from_exit_code(exit_code)

	def cancel(self):
		"""Abort a running verification from another thread"""
		with self._lock:
			self._cancelled = True
			if self._process is not None:
				self._process.terminate()


class BoundVerifier:
	"""A FaceVerifier fixed to one user, the interface the authenticator expects"""

	def __init__(self, user, compare_script=None):
		self.user = user
		self._verifier = FaceVerifier(compare_script=compare_script)

	def verify(self, timeout=None):
		return self._verifier.verify(self.user, timeout=timeout)

	def cancel(self):
		self._verifier.cancel()


class OnnxBoundVerifier:
	"""In-process verifier running the ONNX pipeline (onnx_face.py).

	Unlike BoundVerifier there is no subprocess: the models are loaded once
	and stay resident, so repeated verifications skip session creation and
	the GPU kernel-cache load. Cancellation is polled between camera frames.
	"""

	def __init__(self, user, preload=False):
		self.user = user
		self._pipeline = None
		self._lock = threading.Lock()
		self._cancel = threading.Event()
		if preload:
			threading.Thread(target=self._preload, daemon=True).start()

	def _preload(self):
		try:
			self._get_pipeline()
		except Exception as err:
			print("ONNX pipeline preload failed: %s" % err)

	def _get_pipeline(self):
		with self._lock:
			if self._pipeline is None:
				import onnx_face
				self._pipeline = onnx_face.FacePipeline()
			return self._pipeline

	def verify(self, timeout=None):
		if not self.user or self.user == "root":
			return VerificationResult.ABORT
		self._cancel.clear()
		try:
			pipeline = self._get_pipeline()
			status = pipeline.verify_user(
				self.user, timeout=timeout, cancel_check=self._cancel.is_set)
		except Exception as err:
			print("ONNX verification failed: %s" % err)
			return VerificationResult.UNKNOWN_ERROR
		if self._cancel.is_set():
			return VerificationResult.CANCELLED
		return VerificationResult.from_exit_code(status)

	def cancel(self):
		self._cancel.set()


def create_verifier(user, config=None):
	"""Build the verifier selected by [onnx] enabled in the config.

	Defaults to the classic compare.py subprocess when the section is absent
	or the ONNX dependencies are not importable.
	"""
	use_onnx = False
	if config is not None:
		use_onnx = config.getboolean("onnx", "enabled", fallback=False)
	if use_onnx:
		try:
			import onnx_face  # noqa: F401 -- probe the dependency stack early
			return OnnxBoundVerifier(user, preload=True)
		except ImportError as err:
			print("ONNX pipeline unavailable (%s), using compare.py" % err)
	return BoundVerifier(user)


def verify_face(user, timeout=None):
	"""Verify the face of the given user, returns a VerificationResult"""
	return FaceVerifier().verify(user, timeout=timeout)
