import json
import os

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from webauthn.keystore import KeyStoreError, SoftwareKeyStore


@pytest.fixture
def keystore(tmp_path):
	return SoftwareKeyStore(str(tmp_path / "keystore"))


def test_generate_sign_verify_roundtrip(keystore):
	key_ref, public_key = keystore.generate_key()
	message = b"authenticator data || client data hash"
	signature = keystore.sign(key_ref, message)
	# Raises InvalidSignature on failure
	public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))


def test_signature_bound_to_message(keystore):
	key_ref, public_key = keystore.generate_key()
	signature = keystore.sign(key_ref, b"message one")
	with pytest.raises(InvalidSignature):
		public_key.verify(signature, b"message two", ec.ECDSA(hashes.SHA256()))


def test_keys_are_distinct(keystore):
	ref_a, public_a = keystore.generate_key()
	ref_b, public_b = keystore.generate_key()
	assert ref_a != ref_b
	signature = keystore.sign(ref_a, b"msg")
	with pytest.raises(InvalidSignature):
		public_b.verify(signature, b"msg", ec.ECDSA(hashes.SHA256()))


def test_key_ref_contains_no_plaintext_key(keystore):
	key_ref, public_key = keystore.generate_key()
	ref = json.loads(key_ref)
	assert set(ref) == {"backend", "id", "nonce", "blob"}
	# The blob must not decrypt without the master key: check it is not a
	# DER PKCS8 EC key in disguise
	import base64
	blob = base64.b64decode(ref["blob"])
	from cryptography.hazmat.primitives import serialization
	with pytest.raises(ValueError):
		serialization.load_der_private_key(blob, password=None)


def test_master_key_file_permissions(keystore):
	keystore.generate_key()
	path = keystore._master_key_path()
	assert os.stat(path).st_mode & 0o777 == 0o600


def test_tampered_blob_rejected(keystore):
	key_ref, _ = keystore.generate_key()
	ref = json.loads(key_ref)
	import base64
	blob = bytearray(base64.b64decode(ref["blob"]))
	blob[0] ^= 0xFF
	ref["blob"] = base64.b64encode(bytes(blob)).decode()
	with pytest.raises(KeyStoreError):
		keystore.sign(json.dumps(ref), b"msg")


def test_swapped_aad_rejected(keystore):
	# Wrapping is bound to the key id via AEAD associated data
	key_ref, _ = keystore.generate_key()
	ref = json.loads(key_ref)
	ref["id"] = "different-id"
	with pytest.raises(KeyStoreError):
		keystore.sign(json.dumps(ref), b"msg")


def test_malformed_key_ref_rejected(keystore):
	with pytest.raises(KeyStoreError):
		keystore.sign("not json", b"msg")


def test_destroy_all_removes_master_key(keystore):
	key_ref, _ = keystore.generate_key()
	keystore.destroy_all()
	assert not os.path.exists(keystore._master_key_path())
	# After destruction old refs are useless even though a new master key
	# gets generated transparently
	with pytest.raises(KeyStoreError):
		keystore.sign(key_ref, b"msg")


def test_persistence_across_instances(tmp_path):
	directory = str(tmp_path / "keystore")
	key_ref, public_key = SoftwareKeyStore(directory).generate_key()
	signature = SoftwareKeyStore(directory).sign(key_ref, b"msg")
	public_key.verify(signature, b"msg", ec.ECDSA(hashes.SHA256()))
