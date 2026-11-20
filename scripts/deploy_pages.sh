#!/usr/bin/env bash
# Deploy portfolio showcase + sim demo ke GitHub Pages (branch gh-pages).
# Pakai: bash scripts/deploy_pages.sh   (dari mana saja; butuh git + akses push ke origin)
#
# Hasil di https://daruoktab.github.io/tempograph/ :
#   /         <- web/showcase/index.html
#   /sim/     <- hasil build web/sim (dibuild otomatis kalau web/sim/dist belum ada)
#   /assets/  <- logo untuk preview link (og:image)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && (pwd -W 2>/dev/null || pwd))"
TMP="$ROOT/.pages-tmp"
URL="$(git -C "$ROOT" remote get-url origin)"

if [ ! -d "$ROOT/web/sim/dist" ]; then
  echo "web/sim/dist belum ada, build dulu..."
  (cd "$ROOT/web/sim" && npm ci && npm run build)
fi

rm -rf "$TMP"
trap 'rm -rf "$TMP"' EXIT
git clone --depth 1 --branch gh-pages --single-branch "$URL" "$TMP"

cp "$ROOT/web/showcase/index.html" "$TMP/index.html"
rm -rf "$TMP/sim" "$TMP/assets"
cp -r "$ROOT/web/sim/dist" "$TMP/sim"
mkdir -p "$TMP/assets"
cp "$ROOT/assets/logo-16-by-9.png" "$ROOT/assets/logo.png" "$TMP/assets/"
touch "$TMP/.nojekyll"

git -C "$TMP" add -A
if git -C "$TMP" diff --cached --quiet; then
  echo "Tidak ada perubahan — tidak perlu deploy."
  exit 0
fi
git -C "$TMP" commit -q -m "Redeploy showcase $(date +%Y-%m-%d)"
git -C "$TMP" push -q
echo "Deployed. Live: https://daruoktab.github.io/tempograph/ (build ~30 detik)"
