# Howdy WebAuthn: optional FIDO2/WebAuthn authenticator gated on face verification
# See docs/webauthn-design.md for the architecture

AAGUID = bytes.fromhex("8ba870f8e63c4d38a04bcd471faf1839")


def create_keystore(kind, directory):
	"""Instantiate a keystore backend by its configured kind"""
	if kind == "tpm":
		from webauthn.keystore_tpm import TpmKeyStore
		return TpmKeyStore(directory)
	from webauthn.keystore import SoftwareKeyStore
	return SoftwareKeyStore(directory)
