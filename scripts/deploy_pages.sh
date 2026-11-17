#!/usr/bin/env bash
# Deploy portfolio showcase + sim demo ke GitHub Pages (branch gh-pages).
# Pakai: bash scripts/deploy_pages.sh   (dari mana saja; butuh git + gh auth)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -W)"
TMP="$ROOT/.pages-tmp"
URL="$(git -C "$ROOT" remote get-url origin)"

rm -rf "$TMP"
trap 'rm -rf "$TMP"' EXIT
git clone --depth 1 --branch gh-pages --single-branch "$URL" "$TMP"

cp "$ROOT/Tempograph_Portfolio_Showcase_v2.html" "$TMP/index.html"
rm -rf "$TMP/sim"
cp -r "$ROOT/web/sim/dist" "$TMP/sim"
touch "$TMP/.nojekyll"

git -C "$TMP" add -A
if git -C "$TMP" diff --cached --quiet; then
  echo "Tidak ada perubahan — tidak perlu deploy."
  exit 0
fi
git -C "$TMP" commit -q -m "Redeploy showcase $(date +%Y-%m-%d)"
git -C "$TMP" push -q
echo "Deployed. Live: https://daruoktab.github.io/tempograph/ (build ~30 detik)"
