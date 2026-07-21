#!/usr/bin/env bash
# Fetches the two third-party calibration dependencies into vendor/ (which is
# gitignored — ~45k lines of upstream Java have no business in this repo's
# history) at the exact pinned revisions the pure_r4 Elo anchor was measured
# against. Both upstreams have been dormant for years; the pins are their
# current HEADs, byte-identical to the previously-vendored copies.
#
#   ./fetch_vendor.sh          # populates vendor/{stratego,strategoevaluator}
#
# Licenses: braathwaate/stratego (Demon of Ignorance) is GPL — its license
# ships inside the clone; strategoevaluator is BSD-3 (COPYRIGHT in-tree).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DOI_REPO=https://github.com/braathwaate/stratego
DOI_PIN=fd95e0c79c497521417918cc2dc77e661acfc37d
EVAL_REPO=https://github.com/braathwaate/strategoevaluator
EVAL_PIN=54f0978c5763f9d40e7b7a0d703b739b729ce81c

fetch() {
  local repo=$1 pin=$2 dest=$3
  if [ -e "$dest/.vendor-pin" ] && [ "$(cat "$dest/.vendor-pin")" = "$pin" ]; then
    echo "$dest already at $pin"
    return
  fi
  rm -rf "$dest"
  git init -q "$dest"
  git -C "$dest" fetch -q --depth 1 "$repo" "$pin"
  git -C "$dest" checkout -q FETCH_HEAD
  rm -rf "$dest/.git"
  echo "$pin" > "$dest/.vendor-pin"
  echo "$dest fetched at $pin"
}

mkdir -p vendor
fetch "$DOI_REPO" "$DOI_PIN" vendor/stratego
fetch "$EVAL_REPO" "$EVAL_PIN" vendor/strategoevaluator
