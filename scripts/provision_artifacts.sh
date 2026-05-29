#!/usr/bin/env bash
#
# provision_artifacts.sh — stage the model artifacts the backend loads at
# runtime into ./artifacts/, which docker-compose.prod.yml mounts read-only at
# /artifacts inside the backend container.
#
# WHY THIS EXISTS
#   Two artifacts the server's student-retrieval tier needs are NOT in a normal
#   git checkout of the backend:
#     - student.onnx        (86 MB) — exceeds GitHub's 100 MB file limit, so it
#                            is gitignored and lives only at
#                            mobile/assets/models/student.onnx.
#     - gallery_index.json  (13 MB) — IS tracked in git, but at
#                            mobile/assets/models/gallery_index.json. The
#                            backend's default path is backend/data/gallery_index.json,
#                            which is gitignored (a server copy). The two files
#                            are byte-identical.
#   Without them the student tier self-disables silently and scans fall back to
#   Brickognize/Gemini only — a quality regression, not a crash. This script
#   makes provisioning explicit and verifiable.
#
# USAGE
#   Local source (default) — copy from the in-repo mobile bundle:
#       ./scripts/provision_artifacts.sh
#
#   Remote source — fetch from object storage / a release asset (set URLs):
#       STUDENT_ONNX_URL="https://.../student.onnx" \
#       GALLERY_URL="https://.../gallery_index.json" \
#       ./scripts/provision_artifacts.sh
#
#   Optional integrity check — set the expected sha256 and the script aborts on
#   mismatch (recommended for remote fetches):
#       STUDENT_ONNX_SHA256=7ce9d155...61f6 ./scripts/provision_artifacts.sh
#
# Run from the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ARTIFACTS_DIR:-$ROOT/artifacts}"

# Known-good checksums of the artifacts currently in the repo. Used as the
# DEFAULT expected hashes so a local copy is always verified; override the
# *_SHA256 env vars when you provision a newer build.
DEFAULT_STUDENT_SHA256="7ce9d15589eea0be0b2f0fe2a0c537ef0bc3b35947cfe72ac715eecbb4261f61"
DEFAULT_GALLERY_SHA256="fa891f9c558d2f89e44fc3d0db533e30278bc824d6a328d3b940bdad5492d5ca"

STUDENT_SRC="${STUDENT_ONNX_SRC:-$ROOT/mobile/assets/models/student.onnx}"
GALLERY_SRC="${GALLERY_SRC:-$ROOT/mobile/assets/models/gallery_index.json}"

STUDENT_ONNX_URL="${STUDENT_ONNX_URL:-}"
GALLERY_URL="${GALLERY_URL:-}"

STUDENT_ONNX_SHA256="${STUDENT_ONNX_SHA256:-$DEFAULT_STUDENT_SHA256}"
GALLERY_SHA256="${GALLERY_SHA256:-$DEFAULT_GALLERY_SHA256}"

mkdir -p "$DEST"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'   # macOS
  fi
}

# fetch_or_copy <name> <url> <local_src> <dest_path>
fetch_or_copy() {
  local name="$1" url="$2" src="$3" dest="$4"
  if [[ -n "$url" ]]; then
    echo "→ $name: downloading from $url"
    curl -fL --retry 3 -o "$dest" "$url"
  elif [[ -f "$src" ]]; then
    echo "→ $name: copying from $src"
    cp "$src" "$dest"
  else
    echo "✗ $name: no URL set and local source not found at $src" >&2
    echo "  Set ${name}_URL=... to fetch it, or place the file at $src." >&2
    return 1
  fi
}

verify() {
  local name="$1" path="$2" expected="$3"
  [[ -z "$expected" ]] && return 0
  local got; got="$(sha256_of "$path")"
  if [[ "$got" != "$expected" ]]; then
    echo "✗ $name: sha256 mismatch" >&2
    echo "    expected $expected" >&2
    echo "    got      $got" >&2
    return 1
  fi
  echo "  ✓ $name sha256 verified"
}

echo "Provisioning backend model artifacts → $DEST"
echo ""

fetch_or_copy "STUDENT_ONNX" "$STUDENT_ONNX_URL" "$STUDENT_SRC" "$DEST/student.onnx"
verify "student.onnx" "$DEST/student.onnx" "$STUDENT_ONNX_SHA256"

fetch_or_copy "GALLERY" "$GALLERY_URL" "$GALLERY_SRC" "$DEST/gallery_index.json"
verify "gallery_index.json" "$DEST/gallery_index.json" "$GALLERY_SHA256"

echo ""
echo "Done. Contents of $DEST:"
ls -lh "$DEST"
echo ""
echo "These are mounted read-only at /artifacts by docker-compose.prod.yml;"
echo "the backend reads them via STUDENT_ONNX_PATH and STUDENT_GALLERY_PATH."
echo "Verify the tier loaded after boot by grepping the logs for the load line:"
echo "    docker compose -f docker-compose.prod.yml logs backend | grep student_retrieval"
echo "  Expect: 'student_retrieval: loaded gallery N exemplars ... unique parts'."
echo "  (There is no runtime status endpoint yet — see DEPLOY.md punch list #M3.)"
