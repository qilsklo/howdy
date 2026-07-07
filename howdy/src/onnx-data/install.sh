#!/usr/bin/env bash
# Download the open-source ONNX weights for the Howdy ONNX prototype.
#
# Fetches an InsightFace model pack and keeps the two files the pipeline
# needs: the SCRFD detector (with 5-point landmarks) and the ArcFace
# recognition backbone.
#
#   ./install.sh            # buffalo_l: SCRFD-10G + ArcFace ResNet-50 (best accuracy)
#   ./install.sh buffalo_s  # SCRFD-500M + MobileFaceNet (fast, low power)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK="${1:-buffalo_l}"
URL="https://github.com/deepinsight/insightface/releases/download/v0.7/${PACK}.zip"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading ${PACK} from ${URL}"
curl -L --fail --progress-bar -o "$TMP/pack.zip" "$URL"
unzip -o -q "$TMP/pack.zip" -d "$TMP"

# Packs may or may not nest files inside a directory named after the pack
SRC="$TMP"
[ -d "$TMP/$PACK" ] && SRC="$TMP/$PACK"

copied=0
for f in "$SRC"/det_*.onnx "$SRC"/scrfd*.onnx "$SRC"/w600k_*.onnx; do
	[ -e "$f" ] || continue
	cp "$f" "$DIR/"
	echo "Installed $(basename "$f") ($(du -h "$f" | cut -f1))"
	copied=$((copied + 1))
done

if [ "$copied" -lt 2 ]; then
	echo "ERROR: expected a det_*/scrfd* detector and a w600k_* recognizer in the pack" >&2
	exit 1
fi

echo "Done. Weights are in $DIR"
