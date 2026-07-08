#!/usr/bin/env bash
# Deploy the ONNX face pipeline for the howdy-webauthn daemon.
#
#   sudo ./deploy-webauthn-onnx.sh          # CPU: system onnxruntime
#   sudo ./deploy-webauthn-onnx.sh --gpu    # AMD GPU: ROCm venv + systemd drop-in
#
# Idempotent: safe to re-run after upgrades. Run it from anywhere; when run
# from a git checkout that already has downloaded weights/kernel caches they
# are copied instead of re-downloaded.
set -euo pipefail

HOWDY_LIB="/usr/local/lib/howdy"
DATA="$HOWDY_LIB/onnx-data"
VENV="/opt/howdy-onnx"
ROCM_REL="rocm-rel-7.2.4"   # must match the installed ROCm version
GFX_OVERRIDE="11.0.0"       # HSA override for RDNA3 iGPUs (gfx1103)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ "$(id -u)" -eq 0 ] || { echo "Run with sudo"; exit 1; }
[ -f "$HOWDY_LIB/onnx_face.py" ] || {
	echo "ERROR: $HOWDY_LIB/onnx_face.py not installed - run 'sudo meson install -C build' first"; exit 1; }

GPU=0
[ "${1:-}" = "--gpu" ] && GPU=1

echo "== 1/5 ONNX weights =="
mkdir -p "$DATA"
if ls "$SCRIPT_DIR"/*.onnx >/dev/null 2>&1 && [ "$SCRIPT_DIR" != "$DATA" ]; then
	cp -n "$SCRIPT_DIR"/*.onnx "$DATA/" 2>/dev/null || true
	echo "copied weights from $SCRIPT_DIR"
fi
if ! ls "$DATA"/det_*.onnx >/dev/null 2>&1; then
	"$DATA/install.sh"
fi
# Reuse an existing MIGraphX kernel cache and liveness classifier if present
[ -d "$SCRIPT_DIR/mxr-cache" ] && [ "$SCRIPT_DIR" != "$DATA" ] && cp -rn "$SCRIPT_DIR/mxr-cache" "$DATA/" 2>/dev/null || true
[ -f "$SCRIPT_DIR/liveness_svm.xml" ] && cp -n "$SCRIPT_DIR/liveness_svm.xml" "$DATA/" 2>/dev/null || true
ls "$DATA"/*.onnx | sed 's/^/  /'

echo "== 2/5 Python runtime =="
if [ "$GPU" -eq 1 ]; then
	command -v python3.12 >/dev/null || { echo "python3.12 required for the ROCm wheel"; exit 1; }
	[ -d /opt/rocm ] || { echo "ROCm not installed (pacman -S rocm-hip-runtime migraphx)"; exit 1; }
	python3.12 -m venv "$VENV"
	# opencv-python must stay <5: the PyPI 5.x wheels dropped cv2.ml, which
	# the LBP liveness classifier needs (system/contrib builds still have it)
	"$VENV/bin/pip" install --quiet --upgrade numpy 'opencv-python>=4.8,<5' fido2 cryptography
	"$VENV/bin/pip" install --quiet onnxruntime-migraphx \
		--index-url "https://repo.radeon.com/rocm/manylinux/$ROCM_REL/" \
		--extra-index-url https://pypi.org/simple
	python3 "$DATA/fix-execstack.py" "$VENV"
	"$VENV/bin/python" - <<-'EOF'
	import onnxruntime as ort
	assert "MIGraphXExecutionProvider" in ort.get_available_providers(), ort.get_available_providers()
	print("  venv onnxruntime %s with MIGraphX EP" % ort.__version__)
	EOF
else
	python3 -c "import onnxruntime" 2>/dev/null || {
		echo "system onnxruntime missing: pip install --break-system-packages onnxruntime"; exit 1; }
	echo "  system onnxruntime OK"
fi

echo "== 3/5 systemd drop-in =="
DROPIN=/etc/systemd/system/howdy-webauthn.service.d
mkdir -p "$DROPIN"
if [ "$GPU" -eq 1 ]; then
	cat > "$DROPIN/onnx.conf" <<-EOF
	[Service]
	# Run the daemon on the ROCm venv interpreter (lives in /opt because
	# ProtectHome=true hides /home from the service)
	ExecStart=
	ExecStart=$VENV/bin/python $HOWDY_LIB/webauthn_daemon.py
	# gfx1103 is not an official ROCm target
	Environment=HSA_OVERRIDE_GFX_VERSION=$GFX_OVERRIDE
	# GPU compute + render nodes, blocked by the stock DeviceAllow list
	DeviceAllow=/dev/kfd rw
	DeviceAllow=char-drm rw
	# The HIP runtime maps executable memory; without this GPU init fails
	# and the pipeline silently runs on the CPU provider
	MemoryDenyWriteExecute=false
	# MIGraphX writes compiled-kernel caches here (ProtectSystem=strict
	# makes /usr read-only otherwise)
	ReadWritePaths=$DATA/mxr-cache
	EOF
	mkdir -p "$DATA/mxr-cache"
else
	rm -f "$DROPIN/onnx.conf"
fi
systemctl daemon-reload
echo "  $([ "$GPU" -eq 1 ] && echo installed "$DROPIN/onnx.conf" || echo "no drop-in needed (CPU)")"

echo "== 4/5 enrollment =="
TARGET_USER="${SUDO_USER:-}"
if [ -z "$TARGET_USER" ]; then
	TARGET_USER="$(python3 -c "
import configparser; c = configparser.ConfigParser(); c.read('/usr/local/etc/howdy/config.ini')
print(c.get('webauthn', 'user', fallback=''))")"
fi
PYBIN=$([ "$GPU" -eq 1 ] && echo "$VENV/bin/python" || echo python3)
if [ -n "$TARGET_USER" ]; then
	if "$PYBIN" - "$TARGET_USER" <<-'EOF'
	import sys; sys.path.insert(0, "/usr/local/lib/howdy")
	import onnx_face
	sys.exit(0 if onnx_face.load_user_models(sys.argv[1]) else 1)
	EOF
	then
		echo "  $TARGET_USER already enrolled"
	else
		echo "  Enrolling $TARGET_USER - LOOK AT THE IR CAMERA (starting in 3s)..."
		sleep 3
		if ! "$PYBIN" "$HOWDY_LIB/compare_onnx.py" "$TARGET_USER" --enroll; then
			echo "  WARNING: enrollment did not complete. Re-run it any time with:"
			echo "    sudo $PYBIN $HOWDY_LIB/compare_onnx.py $TARGET_USER --enroll"
		fi
	fi
fi

echo "== 5/5 restart service =="
systemctl restart howdy-webauthn
sleep 2
systemctl --no-pager --lines 6 status howdy-webauthn || true
echo
echo "Done. Verify with: journalctl -u howdy-webauthn -f  then use a passkey at https://webauthn.io"
