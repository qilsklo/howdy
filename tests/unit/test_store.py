import os

import pytest

from webauthn.store import Credential, CredentialStore, StoreError


def make_credential(cred_id=b"\x01" * 16, rp_id="example.com", user_handle=b"user-1", resident=True):
	return Credential(
		credential_id=cred_id,
		rp_id=rp_id,
		rp_name="Example",
		user_handle=user_handle,
		user_name="alice",
		user_display_name="Alice",
		cose_public_key=b"\xa5\x01\x02",
		key_ref='{"backend": "software"}',
		resident=resident,
	)


@pytest.fixture
def store(tmp_path):
	return CredentialStore(str(tmp_path / "creds" / "alice.json"))


def test_roundtrip_persistence(store):
	credential = make_credential()
	store.add(credential)
	reloaded = CredentialStore(store.path)
	assert reloaded.find_by_id(credential.credential_id) == credential


def test_find_by_id_missing(store):
	store.add(make_credential())
	assert store.find_by_id(b"\x02" * 16) is None


def test_find_for_rp_filters_rp_id(store):
	store.add(make_credential(cred_id=b"\x01" * 16, rp_id="example.com"))
	store.add(make_credential(cred_id=b"\x02" * 16, rp_id="other.org", user_handle=b"user-2"))
	matches = store.find_for_rp("example.com")
	assert [c.rp_id for c in matches] == ["example.com"]
	assert store.find_for_rp("missing.net") == []


def test_find_for_rp_allow_list(store):
	a = make_credential(cred_id=b"\x01" * 16, user_handle=b"u1")
	b = make_credential(cred_id=b"\x02" * 16, user_handle=b"u2")
	store.add(a)
	store.add(b)
	matches = store.find_for_rp("example.com", allow_ids=[a.credential_id, b"\x99" * 16])
	assert [c.credential_id for c in matches] == [a.credential_id]


def test_find_for_rp_without_allow_list_returns_resident_only(store):
	resident = make_credential(cred_id=b"\x01" * 16, user_handle=b"u1", resident=True)
	server_side = make_credential(cred_id=b"\x02" * 16, user_handle=b"u2", resident=False)
	store.add(resident)
	store.add(server_side)
	assert [c.credential_id for c in store.find_for_rp("example.com")] == [resident.credential_id]
	# But the non resident credential is reachable through an allow list
	matches = store.find_for_rp("example.com", allow_ids=[server_side.credential_id])
	assert [c.credential_id for c in matches] == [server_side.credential_id]


def test_add_replaces_same_rp_and_user(store):
	old = make_credential(cred_id=b"\x01" * 16)
	new = make_credential(cred_id=b"\x02" * 16)
	store.add(old)
	store.add(new)
	assert store.find_by_id(old.credential_id) is None
	assert len(store.list_all()) == 1


def test_counter_increments_and_persists(store):
	credential = make_credential()
	store.add(credential)
	assert store.increment_counter(credential.credential_id) == 1
	assert store.increment_counter(credential.credential_id) == 2
	reloaded = CredentialStore(store.path)
	assert reloaded.find_by_id(credential.credential_id).sign_count == 2


def test_counter_unknown_credential(store):
	with pytest.raises(StoreError):
		store.increment_counter(b"\xff" * 16)


def test_remove(store):
	credential = make_credential()
	store.add(credential)
	assert store.remove(credential.credential_id) is True
	assert store.remove(credential.credential_id) is False
	assert CredentialStore(store.path).list_all() == []


def test_destroy_all(store):
	store.add(make_credential())
	store.destroy_all()
	assert store.list_all() == []
	assert not os.path.exists(store.path)


def test_corrupt_store_raises(tmp_path):
	path = tmp_path / "creds.json"
	path.write_text("{not json")
	with pytest.raises(StoreError):
		CredentialStore(str(path))


def test_unsupported_version_raises(tmp_path):
	path = tmp_path / "creds.json"
	path.write_text('{"version": 99, "credentials": []}')
	with pytest.raises(StoreError):
		CredentialStore(str(path))


def test_file_permissions(store):
	store.add(make_credential())
	assert os.stat(store.path).st_mode & 0o777 == 0o600


def test_atomic_write_leaves_no_temp_file(store):
	store.add(make_credential())
	directory = os.path.dirname(store.path)
	assert [f for f in os.listdir(directory) if f.endswith(".tmp")] == []
