"""End-to-end integration test.

Drives the full Howdy WebAuthn stack with python-fido2's *client* CTAP2
implementation acting as the browser:

	Ctap2 client -> CTAPHID framing -> LoopbackConnection -> our CtapHidDevice
	transport -> Authenticator -> SoftwareKeyStore (real ECDSA signing)

No root and no /dev/uhid needed: the two 64-byte report streams are wired
together in memory through a pair of queues, exactly the bytes the kernel
would carry. This proves our authenticator interoperates with a real,
independent CTAP2 client.
"""

import hashlib
import queue
import threading

import pytest

fido2 = pytest.importorskip("fido2")

from fido2.ctap2 import Ctap2  # noqa: E402
from fido2.hid.base import CtapHidConnection, HidDescriptor  # noqa: E402
from fido2.hid import CtapHidDevice as ClientHidDevice  # noqa: E402
from fido2.webauthn import AuthenticatorData  # noqa: E402

from verification import VerificationResult  # noqa: E402
from webauthn.authenticator import Authenticator  # noqa: E402
from webauthn.ctaphid import CtapHidDevice as ServerHidDevice  # noqa: E402
from webauthn.keystore import SoftwareKeyStore  # noqa: E402
from webauthn.store import CredentialStore  # noqa: E402


class AlwaysVerifies:
	def verify(self, timeout=None):
		return VerificationResult.SUCCESS

	def cancel(self):
		pass


class LoopbackConnection(CtapHidConnection):
	"""Wires a client HID connection to our server-side transport in memory"""

	def __init__(self, authenticator):
		self._to_client = queue.Queue()
		self._server = ServerHidDevice(authenticator, self._to_client.put)

	def write_packet(self, data):
		# The client writes a 64-byte report to the "device"
		self._server.feed_report(bytes(data))

	def read_packet(self):
		return self._to_client.get(timeout=5)

	def close(self):
		pass


def make_client(tmp_path):
	store = CredentialStore(str(tmp_path / "creds.json"))
	keystore = SoftwareKeyStore(str(tmp_path / "keystore"))
	authenticator = Authenticator(store, keystore, AlwaysVerifies())
	descriptor = HidDescriptor("loopback", 0x1209, 0xF1D0, 64, 64, "Howdy", None)
	connection = LoopbackConnection(authenticator)
	return ClientHidDevice(descriptor, connection)


def test_ctap2_getinfo(tmp_path):
	ctap = Ctap2(make_client(tmp_path))
	info = ctap.get_info()
	assert "FIDO_2_1" in info.versions
	assert info.options.get("uv") is True


def test_full_registration_and_assertion(tmp_path):
	device = make_client(tmp_path)
	ctap = Ctap2(device)

	rp = {"id": "example.com", "name": "Example"}
	user = {"id": b"user-1", "name": "alice", "displayName": "Alice"}
	client_data_hash = hashlib.sha256(b"reg").digest()

	attestation = ctap.make_credential(
		client_data_hash,
		rp,
		user,
		[{"type": "public-key", "alg": -7}],
		options={"rk": True},
	)
	auth_data = attestation.auth_data
	assert auth_data.is_user_verified()
	credential_id = auth_data.credential_data.credential_id
	public_key = auth_data.credential_data.public_key

	# The self-attestation signature verifies against the credential key
	public_key.verify(bytes(auth_data) + client_data_hash, attestation.att_stmt["sig"])

	# Now assert (sign a fresh challenge) and verify like an RP would
	assertion_hash = hashlib.sha256(b"login-challenge").digest()
	assertions = ctap.get_assertions("example.com", assertion_hash)
	assertion = assertions[0]
	assert assertion.credential["id"] == credential_id
	assert AuthenticatorData(assertion.auth_data).counter == 1

	public_key.verify(bytes(assertion.auth_data) + assertion_hash, assertion.signature)


def test_counter_increments_over_the_wire(tmp_path):
	device = make_client(tmp_path)
	ctap = Ctap2(device)
	ctap.make_credential(
		hashlib.sha256(b"reg").digest(),
		{"id": "example.com", "name": "Example"},
		{"id": b"user-1", "name": "alice", "displayName": "Alice"},
		[{"type": "public-key", "alg": -7}],
		options={"rk": True},
	)
	counters = []
	for i in range(3):
		assertion = ctap.get_assertions("example.com", hashlib.sha256(f"c{i}".encode()).digest())[0]
		counters.append(AuthenticatorData(assertion.auth_data).counter)
	assert counters == [1, 2, 3]
