#!/usr/bin/env bash
# Fetch a heavyweight payload from the arcade-assets R2 bucket into
# web/app/.asset-cache/<logical-path>, verify its sha256 against
# web/app/asset-manifest.json, and print the cached file's absolute path.
# The on-disk consumers (tests, examples, smoke) all resolve payload bytes
# through this; the browser app resolves the same manifest via assetUrl.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
path="${1:?usage: fetch-asset.sh <logical-path>}"
manifest="$root/web/app/asset-manifest.json"
cache="$root/web/app/.asset-cache/$path"

digest=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$manifest" "$path")

sha() {
  python3 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"
}

if [ ! -f "$cache" ] || [ "$(sha "$cache")" != "$digest" ]; then
  dir=$(dirname "$path")
  base=$(basename "$path")
  url="https://arcade-assets.henrilemoine.com/$dir/${base%.*}.$digest.bin"
  mkdir -p "$(dirname "$cache")"
  curl -fsSL "$url" -o "$cache.tmp"
  [ "$(sha "$cache.tmp")" = "$digest" ] || { echo "checksum mismatch for $url" >&2; rm -f "$cache.tmp"; exit 1; }
  mv "$cache.tmp" "$cache"
fi

echo "$cache"
