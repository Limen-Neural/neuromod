#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GITHUB_SHA:-}" || -z "${GITHUB_REPOSITORY:-}" || -z "${GH_TOKEN:-}" ]]; then
  echo "GITHUB_SHA, GITHUB_REPOSITORY, and GH_TOKEN are required" >&2
  exit 2
fi

git fetch --no-tags origin main
if ! git merge-base --is-ancestor "$GITHUB_SHA" origin/main; then
  echo "release commit $GITHUB_SHA is not reachable from main" >&2
  exit 1
fi

runs=$(gh run list \
  --repo "$GITHUB_REPOSITORY" \
  --workflow ci.yml \
  --commit "$GITHUB_SHA" \
  --limit 100 \
  --json status,conclusion)

if ! jq -e 'any(.[]; .status == "completed" and .conclusion == "success")' <<<"$runs" >/dev/null; then
  echo "no successful CI run is recorded for release commit $GITHUB_SHA" >&2
  exit 1
fi

echo "release commit $GITHUB_SHA is on main and has a successful CI run"
