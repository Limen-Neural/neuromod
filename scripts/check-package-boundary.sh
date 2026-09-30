#!/usr/bin/env bash
set -euo pipefail

package_files=$(cargo package --list --locked --all-features --allow-dirty)
unexpected_scripts=$(grep -E '^scripts/' <<<"$package_files" || true)

if [[ -n "$unexpected_scripts" ]]; then
  printf 'Cargo package includes repository-only scripts:\n%s\n' "$unexpected_scripts" >&2
  exit 1
fi

echo 'Cargo package excludes scripts/'
