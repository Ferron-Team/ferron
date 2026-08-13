#!/usr/bin/env bash
#
# Download Crytek's Sponza atrium into assets/sponza/, for `ORRIN_SCENE=sponza`.
#
# The model is not in this repository and should not be: it is ~40 MB of somebody
# else's asset under a licence that asks for attribution, and nothing under
# version control here needs to carry it. assets/sponza/ is gitignored.
#
# What is fetched is Khronos' glTF 2.0 re-release of Crytek's original — PBR
# metallic-roughness materials, tangents, and an alpha-masked plant, which is what
# the importer in crates/core/src/scene/model.rs is written against. CC BY 3.0;
# the README and LICENSE that state the attribution are copied in beside it.
#
# A sparse, blobless clone rather than ~30 raw.githubusercontent.com URLs: the
# file list is the upstream repository's business, not this script's, and a
# hard-coded list rots silently into a half-downloaded model.

set -euo pipefail

readonly REPO="https://github.com/KhronosGroup/glTF-Sample-Assets.git"
readonly SUBDIR="Models/Sponza"
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TARGET="$ROOT/assets/sponza"

force=0
[[ "${1:-}" == "--force" ]] && force=1

if [[ -f "$TARGET/Sponza.gltf" && $force -eq 0 ]]; then
    echo "sponza: already at $TARGET (pass --force to re-download)"
    exit 0
fi

command -v git >/dev/null || {
    echo "sponza: needs git on PATH" >&2
    exit 1
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "sponza: cloning $SUBDIR from $REPO"
git clone --depth 1 --filter=blob:none --sparse "$REPO" "$work/assets" >/dev/null 2>&1
git -C "$work/assets" sparse-checkout set "$SUBDIR" >/dev/null

source="$work/assets/$SUBDIR"
[[ -f "$source/glTF/Sponza.gltf" ]] || {
    echo "sponza: upstream no longer has $SUBDIR/glTF/Sponza.gltf" >&2
    exit 1
}

rm -rf "$TARGET"
mkdir -p "$TARGET"
cp -R "$source/glTF/." "$TARGET/"
# Attribution travels with the asset, not with this script.
for doc in "$source/README.md" "$work/assets/LICENSE.md"; do
    [[ -f "$doc" ]] && cp "$doc" "$TARGET/$(basename "$doc")"
done

size="$(du -sh "$TARGET" | cut -f1)"
echo "sponza: $size in $TARGET"
echo "sponza: run it with"
echo "    ORRIN_SCENE=sponza cargo run --release -p orrin-core"
