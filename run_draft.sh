#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

# Keep draft preview output separate from the production `_site` directory.
# This prevents a later deployment from accidentally reusing rendered drafts.
preview_dir="$(mktemp -d "${TMPDIR:-/tmp}/al-folio-drafts.XXXXXX")"

cleanup() {
  rm -rf "$preview_dir"
}
trap cleanup EXIT INT TERM

bundle exec jekyll serve --drafts --destination "$preview_dir"
