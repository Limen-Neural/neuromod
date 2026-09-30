#!/usr/bin/env bash
set -euo pipefail

tag="${RELEASE_TAG:?RELEASE_TAG is required}"
if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "expected a stable vX.Y.Z tag, got $tag" >&2
  exit 1
fi

version=$(awk '
  $0 == "[package]" { in_package = 1; next }
  in_package && /^\[/ { exit }
  in_package && $1 == "version" && $2 == "=" {
    quote = substr($3, 1, 1)
    if (quote == "\"" || quote == sprintf("%c", 39)) {
      $3 = substr($3, 2, length($3) - 2)
    }
    print $3
    exit
  }
' Cargo.toml)
if [[ -z "$version" ]]; then
  echo "could not read [package].version from Cargo.toml" >&2
  exit 1
fi

if [[ "$tag" != "v$version" ]]; then
  echo "tag $tag does not match package version v$version" >&2
  exit 1
fi

printf 'Publishing neuromod %s from tag %s\n' "$version" "$tag"
