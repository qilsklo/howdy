# Howdy WebAuthn daemon
# Creates the virtual FIDO HID device and pumps host reports through the
# CTAPHID transport into the authenticator. Run as root (needs /dev/uhid,
# the camera and the root-only credential store).

import configparser
import logging
import os
import pwd
import signal
import sys

import paths_factory

from i18n import _
from verification import BoundVerifier
from webauthn import create_keystore
from webauthn.authenticator import Authenticator
from webauthn.ctaphid import CtapHidDevice
from webauthn.store import CredentialStore
from webauthn.uhid import UHidDevice


def _resolve_user(config):
	"""The user whose face unlocks credentials"""
	configured = config.get("webauthn", "user", fallback="").strip()
	if configured:
		return configured

	# Fall back to the owner of an active graphical session via logind
	try:
		import glob
		for base in ("/run/user/*",):
			for path in glob.glob(base):
				uid = int(os.path.basename(path))
				if uid >= 1000:
					return pwd.getpwuid(uid).pw_name
	except (OSError, KeyError, ValueError):
		pass
	return ""


def run_service(state):
	# Route the authenticator's operational logging to stderr, which systemd
	# captures into the journal (journalctl -u howdy-webauthn)
	logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

	config = configparser.ConfigParser()
	config.read(paths_factory.config_file_path())

	if not config.getboolean("webauthn", "enabled", fallback=False):
		print(_("Howdy WebAuthn is disabled in the config ([webauthn] enabled = false)"))
		sys.exit(1)

	user = _resolve_user(config)
	if not user:
		print(_("Could not determine which user's face to use, set [webauthn] user"))
		sys.exit(1)

	verify_timeout = config.getfloat("webauthn", "verify_timeout", fallback=10.0)

	store = CredentialStore(paths_factory.webauthn_credentials_path(user))
	keystore = create_keystore(state["keystore"], paths_factory.webauthn_keystore_dir_path())
	verifier = BoundVerifier(user)
	authenticator = Authenticator(store, keystore, verifier, verify_timeout=verify_timeout)

	try:
		device = UHidDevice()
	except PermissionError:
		print(_("Cannot open /dev/uhid, the service must run as root"))
		sys.exit(1)
	except FileNotFoundError:
		print(_("/dev/uhid not present, load the uhid kernel module"))
		sys.exit(1)

	transport = CtapHidDevice(authenticator, device.send_input)

	stopping = {"flag": False}

	def handle_signal(signum, frame):
		stopping["flag"] = True
		try:
			device.close()
		except OSError:
			pass

	signal.signal(signal.SIGTERM, handle_signal)
	signal.signal(signal.SIGINT, handle_signal)

	print(_("Howdy WebAuthn authenticator running for user {} (keystore: {})").format(user, state["keystore"]))

	try:
		while not stopping["flag"]:
			try:
				event_type, report = device.read_event()
			except OSError:
				break
			if report:
				transport.feed_report(report)
	finally:
		device.close()
