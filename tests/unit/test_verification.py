import os
import stat
import sys
import threading
import time

import pytest

from verification import FaceVerifier, VerificationResult, verify_face


def make_stub_compare(tmp_path, body):
	"""Write a stand-in compare.py that runs the given python body"""
	script = tmp_path / "compare_stub.py"
	script.write_text(body)
	return str(script)


def make_verifier(tmp_path, body):
	return FaceVerifier(compare_script=make_stub_compare(tmp_path, body), python_executable=sys.executable)


def test_success_exit_code(tmp_path):
	verifier = make_verifier(tmp_path, "import sys; sys.exit(0)")
	assert verifier.verify("alice") == VerificationResult.SUCCESS


@pytest.mark.parametrize("code,result", [
	(10, VerificationResult.NO_FACE_MODEL),
	(11, VerificationResult.TIMEOUT_REACHED),
	(12, VerificationResult.ABORT),
	(13, VerificationResult.TOO_DARK),
	(14, VerificationResult.INVALID_DEVICE),
	(15, VerificationResult.RUBBERSTAMP),
])
def test_compare_error_codes(tmp_path, code, result):
	verifier = make_verifier(tmp_path, f"import sys; sys.exit({code})")
	assert verifier.verify("alice") == result


def test_unknown_exit_code(tmp_path):
	verifier = make_verifier(tmp_path, "import sys; sys.exit(42)")
	assert verifier.verify("alice") == VerificationResult.UNKNOWN_ERROR


def test_user_is_passed_to_compare(tmp_path):
	marker = tmp_path / "user.txt"
	verifier = make_verifier(tmp_path, f"import sys; open({str(marker)!r}, 'w').write(sys.argv[1]); sys.exit(0)")
	assert verifier.verify("bob") == VerificationResult.SUCCESS
	assert marker.read_text() == "bob"


def test_rejects_empty_user(tmp_path):
	verifier = make_verifier(tmp_path, "import sys; sys.exit(0)")
	assert verifier.verify("") == VerificationResult.ABORT


def test_rejects_root(tmp_path):
	verifier = make_verifier(tmp_path, "import sys; sys.exit(0)")
	assert verifier.verify("root") == VerificationResult.ABORT


def test_hard_timeout_kills_process(tmp_path):
	verifier = make_verifier(tmp_path, "import time; time.sleep(30)")
	start = time.monotonic()
	assert verifier.verify("alice", timeout=0.5) == VerificationResult.TIMEOUT_REACHED
	assert time.monotonic() - start < 5


def test_cancel_from_other_thread(tmp_path):
	verifier = make_verifier(tmp_path, "import time; time.sleep(30)")
	results = []
	thread = threading.Thread(target=lambda: results.append(verifier.verify("alice")))
	thread.start()
	# Give the subprocess a moment to start before cancelling
	time.sleep(0.5)
	verifier.cancel()
	thread.join(timeout=5)
	assert not thread.is_alive()
	assert results == [VerificationResult.CANCELLED]


def test_concurrent_verify_rejected(tmp_path):
	verifier = make_verifier(tmp_path, "import time; time.sleep(5)")
	results = []
	thread = threading.Thread(target=lambda: results.append(verifier.verify("alice")))
	thread.start()
	time.sleep(0.5)
	# Second call on the same instance must not spawn a second camera process
	assert verifier.verify("alice") == VerificationResult.ABORT
	verifier.cancel()
	thread.join(timeout=5)


def test_missing_compare_script():
	verifier = FaceVerifier(compare_script="/nonexistent/compare.py", python_executable="/nonexistent/python")
	assert verifier.verify("alice") == VerificationResult.UNKNOWN_ERROR


def test_module_level_helper_uses_real_script_path():
	# The default script path must point at the installed compare.py location
	verifier = FaceVerifier()
	assert verifier.compare_script.endswith(os.path.join("src", "compare.py")) or verifier.compare_script.endswith("compare.py")
