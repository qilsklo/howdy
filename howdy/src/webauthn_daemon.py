# Entry point for the Howdy WebAuthn daemon.
#
# This is deliberately separate from the interactive `howdy` CLI: that CLI
# refuses to run as root (it expects a human sudo'ing for their own account),
# but the daemon *must* be root to open /dev/uhid, the camera and the
# root-only credential store. The service resolves which user's face to use
# from config.ini on its own, so it never needs the CLI's user handling.

import json
import sys

import paths_factory

from i18n import _


def main():
	try:
		import fido2  # noqa: F401
	except ImportError:
		print(_("The Howdy WebAuthn feature requires the python-fido2 package"))
		sys.exit(1)

	try:
		with open(paths_factory.webauthn_state_path()) as file:
			state = json.load(file)
	except FileNotFoundError:
		print(_("Howdy WebAuthn has not been set up yet, run: sudo howdy webauthn init"))
		sys.exit(1)

	from webauthn.service import run_service
	run_service(state)


if __name__ == "__main__":
	main()
