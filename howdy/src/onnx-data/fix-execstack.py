#!/usr/bin/env python3
# Clear the executable-stack flag on AMD's onnxruntime wheels.
#
# The onnxruntime-migraphx/-rocm wheels from repo.radeon.com ship a shared
# object whose PT_GNU_STACK program header requests an executable stack,
# which glibc >= 2.41 refuses to load ("cannot enable executable stack as
# shared object requires: Invalid argument"). Clearing the PF_X bit is safe
# (nothing in the library actually executes from the stack) and is what
# patchelf --clear-execstack (>= 0.18) does.
#
# Usage: python3 fix-execstack.py <path-to-.so | venv-dir>

import glob
import os
import struct
import sys

PT_GNU_STACK = 0x6474E551


def clear_execstack(path):
	with open(path, "r+b") as f:
		head = f.read(64)
		if head[:4] != b"\x7fELF" or head[4] != 2:
			print("skipped (not ELF64): " + path)
			return
		phoff = struct.unpack_from("<Q", head, 0x20)[0]
		phentsize = struct.unpack_from("<H", head, 0x36)[0]
		phnum = struct.unpack_from("<H", head, 0x38)[0]
		for i in range(phnum):
			f.seek(phoff + i * phentsize)
			p_type, p_flags = struct.unpack("<II", f.read(8))
			if p_type == PT_GNU_STACK:
				if not p_flags & 0x1:
					print("already clear: " + path)
					return
				f.seek(phoff + i * phentsize + 4)
				f.write(struct.pack("<I", p_flags & ~0x1))
				print("cleared execstack: " + path)
				return
		print("no PT_GNU_STACK header: " + path)


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit(__doc__ or "usage: fix-execstack.py <so-file | venv-dir>")
	target = sys.argv[1]
	if os.path.isdir(target):
		pattern = os.path.join(target, "**", "onnxruntime", "capi", "*.so")
		files = glob.glob(pattern, recursive=True)
		if not files:
			sys.exit("no onnxruntime .so files under " + target)
		for f in files:
			clear_execstack(f)
	else:
		clear_execstack(target)
