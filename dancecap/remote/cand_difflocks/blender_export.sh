#!/usr/bin/env bash
# Optional: DiffLocks' own Blender export (inference/npz2blender.py, needs Blender 4.1.1 exactly) of the FIRST input's
# strands -> <pod_out>/blender/{blender_scene.blend,hair.abc}. Official blender.org tarball; headless (--background).
#   bash remote/cand_difflocks/blender_export.sh <pod_out dir, relative to dancecap/>
set -uo pipefail
OUT=$(realpath -m "${1:?pod_out dir}")/blender
ROOT=/workspace/difflocks_cand
REPO=$ROOT/difflocks
NPZ=$ROOT/tmp/difflocks_output_strands.npz
URL=${BLENDER_URL:-https://download.blender.org/release/Blender4.1/blender-4.1.1-linux-x64.tar.xz}
[ -f "$NPZ" ] || { echo "[blender] no float32 strands at $NPZ"; exit 2; }
mkdir -p "$OUT"
t0=$(date +%s)
cd "$ROOT" || exit 3
if [ ! -x blender-4.1.1-linux-x64/blender ]; then
  curl -sSfL --retry 3 -o blender.tar.xz "$URL" || { echo "[blender] download failed"; exit 4; }
  tar -xJf blender.tar.xz || exit 5
  rm -f blender.tar.xz
fi
(apt-get install -y -qq --no-install-recommends libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 \
   libice6 libx11-6 libgl1 > /dev/null 2>&1 || { apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq \
   --no-install-recommends libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 libx11-6 libgl1 \
   > /dev/null 2>&1; }) || echo "[blender] apt X libs failed (continuing)"
echo "[blender] $(blender-4.1.1-linux-x64/blender --version 2>/dev/null | head -1) ready in $(( $(date +%s) - t0 )) s"
cd "$REPO" || exit 6
timeout 720 "$ROOT/blender-4.1.1-linux-x64/blender" -t 8 --background --python ./inference/npz2blender.py -- \
  --input_npz "$NPZ" --out_path "$OUT" --strands_subsample 1.0 --vertex_subsample 0.5 --alembic_resolution 7 \
  --export_alembic > "$OUT/blender.log" 2>&1
rc=$?
tail -5 "$OUT/blender.log"
python - "$OUT" "$rc" "$ROOT/tmp/which.json" <<'EOF'
import json, os, sys
out, rc, which = sys.argv[1], int(sys.argv[2]), sys.argv[3]
files = {f: os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)}
d = {"rc": rc, "files": files, "input": json.load(open(which)) if os.path.exists(which) else None,
     "settings": {"strands_subsample": 1.0, "vertex_subsample": 0.5, "alembic_resolution": 7, "blender": "4.1.1"}}
json.dump(d, open(os.path.join(out, "blender.json"), "w"), indent=1)
print("[blender]", d)
EOF
echo "[blender] done in $(( $(date +%s) - t0 )) s (rc $rc)"
[ -f "$OUT/hair.abc" ] || [ -f "$OUT/blender_scene.blend" ]
