import hashlib
import os
import time

import pytest
from fido2 import cbor
from fido2.cose import CoseKey
from fido2.ctap import CtapError
from fido2.webauthn import AttestedCredentialData, AuthenticatorData

from verification import VerificationResult
from webauthn import AAGUID
from webauthn.authenticator import (
	CMD_GET_ASSERTION,
	CMD_GET_INFO,
	CMD_MAKE_CREDENTIAL,
	Authenticator,
)
from webauthn.keystore import SoftwareKeyStore
from webauthn.store import CredentialStore

ERR = CtapError.ERR
CLIENT_DATA_HASH = hashlib.sha256(b"client data").digest()
RP = {"id": "example.com", "name": "Example"}
USER = {"id": b"user-handle-1", "name": "alice", "displayName": "Alice"}
ES256_PARAM = [{"type": "public-key", "alg": -7}]


class FakeVerifier:
	"""Scripted face verification outcomes, records how often it ran"""

	def __init__(self, result=VerificationResult.SUCCESS):
		self.result = result
		self.calls = 0

	def verify(self, timeout=None):
		self.calls += 1
		return self.result

	def cancel(self):
		pass


@pytest.fixture
def verifier():
	return FakeVerifier()


@pytest.fixture
def authenticator(tmp_path, verifier):
	store = CredentialStore(str(tmp_path / "creds.json"))
	keystore = SoftwareKeyStore(str(tmp_path / "keystore"))
	return Authenticator(store, keystore, verifier)


def make_credential_params(overrides=None):
	params = {
		1: CLIENT_DATA_HASH,
		2: dict(RP),
		3: dict(USER),
		4: list(ES256_PARAM),
		7: {"rk": True},
	}
	params.update(overrides or {})
	return params


def get_assertion_params(overrides=None):
	params = {
		1: RP["id"],
		2: CLIENT_DATA_HASH,
	}
	params.update(overrides or {})
	return params


def register(authenticator, overrides=None):
	"""Run MakeCredential and return the parsed attestation pieces"""
	response = authenticator.make_credential(make_credential_params(overrides))
	auth_data = AuthenticatorData(response[2])
	return response, auth_data, AttestedCredentialData(auth_data.credential_data)


# --- MakeCredential / credential creation ---

def test_make_credential_success(authenticator, verifier):
	response, auth_data, attested = register(authenticator)

	assert response[1] == "packed"
	assert verifier.calls == 1
	assert auth_data.is_user_present() and auth_data.is_user_verified() and auth_data.is_attested()
	assert auth_data.rp_id_hash == hashlib.sha256(b"example.com").digest()
	assert attested.aaguid == AAGUID

	# Self attestation signature verifies with the credential public key
	public_key = CoseKey.parse(attested.public_key)
	assert response[3]["alg"] == -7
	public_key.verify(response[2] + CLIENT_DATA_HASH, response[3]["sig"])


def test_make_credential_stores_credential(authenticator):
	_, _, attested = register(authenticator)
	stored = authenticator.store.find_by_id(attested.credential_id)
	assert stored is not None
	assert stored.rp_id == "example.com"
	assert stored.user_handle == USER["id"]
	assert stored.resident is True
	assert stored.sign_count == 0


def test_make_credential_face_verification_failure(authenticator, verifier):
	verifier.result = VerificationResult.TIMEOUT_REACHED
	with pytest.raises(CtapError) as err:
		authenticator.make_credential(make_credential_params())
	assert err.value.code == ERR.USER_ACTION_TIMEOUT
	assert authenticator.store.list_all() == []

	verifier.result = VerificationResult.ABORT
	with pytest.raises(CtapError) as err:
		authenticator.make_credential(make_credential_params())
	assert err.value.code == ERR.OPERATION_DENIED
	assert authenticator.store.list_all() == []


def test_make_credential_unsupported_algorithm(authenticator, verifier):
	with pytest.raises(CtapError) as err:
		authenticator.make_credential(make_credential_params({4: [{"type": "public-key", "alg": -8}]}))
	assert err.value.code == ERR.UNSUPPORTED_ALGORITHM
	assert verifier.calls == 0


def test_make_credential_missing_parameter(authenticator):
	params = make_credential_params()
	del params[2]
	with pytest.raises(CtapError) as err:
		authenticator.make_credential(params)
	assert err.value.code == ERR.MISSING_PARAMETER


def test_make_credential_exclude_list(authenticator, verifier):
	_, _, attested = register(authenticator)
	exclude = [{"type": "public-key", "id": attested.credential_id}]
	with pytest.raises(CtapError) as err:
		authenticator.make_credential(make_credential_params({5: exclude}))
	assert err.value.code == ERR.CREDENTIAL_EXCLUDED
	# Presence was still required before disclosing existence
	assert verifier.calls == 2


def test_make_credential_exclude_list_other_rp_ignored(authenticator):
	_, _, attested = register(authenticator)
	exclude = [{"type": "public-key", "id": attested.credential_id}]
	params = make_credential_params({2: {"id": "other.org", "name": "Other"}, 5: exclude})
	response = authenticator.make_credential(params)
	assert response[1] == "packed"


def test_make_credential_rejects_pin_auth(authenticator):
	with pytest.raises(CtapError) as err:
		authenticator.make_credential(make_credential_params({8: b"\x00" * 16}))
	assert err.value.code == ERR.PIN_AUTH_INVALID


def test_make_credential_cred_protect_echoed(authenticator):
	response = authenticator.make_credential(make_credential_params({6: {"credProtect": 2}}))
	auth_data = AuthenticatorData(response[2])
	assert auth_data.extensions == {"credProtect": 2}


def test_make_credential_replaces_same_rp_user(authenticator):
	_, _, first = register(authenticator)
	_, _, second = register(authenticator)
	assert authenticator.store.find_by_id(first.credential_id) is None
	assert authenticator.store.find_by_id(second.credential_id) is not None


# --- GetAssertion / challenge signing ---

def test_get_assertion_success_allow_list(authenticator, verifier):
	_, _, attested = register(authenticator)
	allow = [{"type": "public-key", "id": attested.credential_id}]
	response = authenticator.get_assertion(get_assertion_params({3: allow}))

	assert response[1]["id"] == attested.credential_id
	auth_data = AuthenticatorData(response[2])
	assert auth_data.is_user_present() and auth_data.is_user_verified()
	assert auth_data.counter == 1

	public_key = CoseKey.parse(attested.public_key)
	public_key.verify(response[2] + CLIENT_DATA_HASH, response[3])
	assert verifier.calls == 2  # once for registration, once for the assertion


def test_get_assertion_discoverable_without_allow_list(authenticator):
	_, _, attested = register(authenticator)
	response = authenticator.get_assertion(get_assertion_params())
	assert response[1]["id"] == attested.credential_id
	assert response[4]["id"] == USER["id"]


def test_get_assertion_signature_binds_client_data(authenticator):
	_, _, attested = register(authenticator)
	response = authenticator.get_assertion(get_assertion_params())
	public_key = CoseKey.parse(attested.public_key)
	from cryptography.exceptions import InvalidSignature
	with pytest.raises((InvalidSignature, Exception)):
		public_key.verify(response[2] + hashlib.sha256(b"other client data").digest(), response[3])


def test_get_assertion_invalid_rp_id(authenticator, verifier):
	register(authenticator)
	with pytest.raises(CtapError) as err:
		authenticator.get_assertion(get_assertion_params({1: "evil.com"}))
	assert err.value.code == ERR.NO_CREDENTIALS
	# The camera is never involved when there is nothing to sign with
	assert verifier.calls == 1


def test_get_assertion_allow_list_wrong_rp(authenticator):
	# A credential id that exists but belongs to another RP must not sign
	_, _, attested = register(authenticator)
	allow = [{"type": "public-key", "id": attested.credential_id}]
	with pytest.raises(CtapError) as err:
		authenticator.get_assertion(get_assertion_params({1: "evil.com", 3: allow}))
	assert err.value.code == ERR.NO_CREDENTIALS


def test_get_assertion_empty_store(authenticator):
	with pytest.raises(CtapError) as err:
		authenticator.get_assertion(get_assertion_params())
	assert err.value.code == ERR.NO_CREDENTIALS


def test_get_assertion_face_verification_failure(authenticator, verifier):
	_, _, attested = register(authenticator)
	verifier.result = VerificationResult.TIMEOUT_REACHED
	with pytest.raises(CtapError) as err:
		authenticator.get_assertion(get_assertion_params())
	assert err.value.code == ERR.USER_ACTION_TIMEOUT
	# No signature was produced and the counter did not move
	assert authenticator.store.find_by_id(attested.credential_id).sign_count == 0


def test_get_assertion_counter_updates_and_persists(authenticator, tmp_path):
	_, _, attested = register(authenticator)
	for expected in (1, 2, 3):
		response = authenticator.get_assertion(get_assertion_params())
		assert AuthenticatorData(response[2]).counter == expected
	reloaded = CredentialStore(str(tmp_path / "creds.json"))
	assert reloaded.find_by_id(attested.credential_id).sign_count == 3


def test_get_assertion_up_false_rejected(authenticator):
	register(authenticator)
	with pytest.raises(CtapError) as err:
		authenticator.get_assertion(get_assertion_params({5: {"up": False}}))
	assert err.value.code == ERR.UNSUPPORTED_OPTION


def test_get_next_assertion(authenticator):
	register(authenticator)
	register(authenticator, {3: {"id": b"user-handle-2", "name": "bob", "displayName": "Bob"}})
	first = authenticator.get_assertion(get_assertion_params())
	assert first[5] == 2
	second = authenticator.get_next_assertion()
	assert 5 not in second
	assert first[1]["id"] != second[1]["id"]
	assert {first[4]["id"], second[4]["id"]} == {b"user-handle-1", b"user-handle-2"}
	with pytest.raises(CtapError) as err:
		authenticator.get_next_assertion()
	assert err.value.code == ERR.NOT_ALLOWED


def test_get_next_assertion_without_get_assertion(authenticator):
	with pytest.raises(CtapError) as err:
		authenticator.get_next_assertion()
	assert err.value.code == ERR.NOT_ALLOWED


# --- GetInfo / dispatch / reset ---

def test_get_info_contents(authenticator):
	info = authenticator.get_info()
	assert "FIDO_2_1" in info[1]
	assert info[3] == AAGUID
	assert info[4]["uv"] is True and info[4]["rk"] is True and info[4]["plat"] is False
	assert "clientPin" not in info[4]


def test_handle_cbor_get_info_roundtrip(authenticator):
	response = authenticator.handle_cbor(bytes([CMD_GET_INFO]))
	assert response[0] == 0x00
	info = cbor.decode(response[1:])
	assert info[3] == AAGUID


def test_handle_cbor_full_make_and_assert(authenticator):
	request = bytes([CMD_MAKE_CREDENTIAL]) + cbor.encode(make_credential_params())
	response = authenticator.handle_cbor(request)
	assert response[0] == 0x00

	request = bytes([CMD_GET_ASSERTION]) + cbor.encode(get_assertion_params())
	response = authenticator.handle_cbor(request)
	assert response[0] == 0x00
	assertion = cbor.decode(response[1:])
	assert AuthenticatorData(assertion[2]).counter == 1


def test_handle_cbor_error_status(authenticator):
	request = bytes([CMD_GET_ASSERTION]) + cbor.encode(get_assertion_params())
	response = authenticator.handle_cbor(request)
	assert response == bytes([ERR.NO_CREDENTIALS])


def test_handle_cbor_unknown_command(authenticator):
	assert authenticator.handle_cbor(bytes([0x42])) == bytes([ERR.INVALID_COMMAND])


def test_handle_cbor_garbage_cbor(authenticator):
	assert authenticator.handle_cbor(bytes([CMD_MAKE_CREDENTIAL]) + b"\xff\xff") == bytes([ERR.INVALID_CBOR])


def test_reset_inside_window(authenticator):
	register(authenticator)
	authenticator.reset()
	assert authenticator.store.list_all() == []


def test_reset_after_window_rejected(authenticator):
	register(authenticator)
	authenticator._started = time.monotonic() - 60
	with pytest.raises(CtapError) as err:
		authenticator.reset()
	assert err.value.code == ERR.NOT_ALLOWED
	assert len(authenticator.store.list_all()) == 1
