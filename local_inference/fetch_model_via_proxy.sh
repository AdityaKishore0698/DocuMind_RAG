#!/bin/bash
# Mirror one model from registry.ollama.ai into ~/.ollama/models exactly as `ollama pull` lays it out,
# using curl (which honours the HTTPS proxy). Every blob is sha256-verified before it is moved into place;
# the manifest is written last, so a failed run leaves no visible/partial model.
set -euo pipefail
name="$1"; tag="${2:-latest}"
store="$HOME/.ollama/models"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
accept="application/vnd.docker.distribution.manifest.v2+json"
curl -sSfL -m 60 -H "Accept: $accept" "https://registry.ollama.ai/v2/library/$name/manifests/$tag" -o "$tmp/manifest"
digests=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(' '.join([m['config']['digest']]+[l['digest'] for l in m['layers']]))" "$tmp/manifest")
for d in $digests; do
  hex="${d#sha256:}"; dest="$store/blobs/sha256-$hex"
  if [ -f "$dest" ] && [ "$(shasum -a 256 "$dest" | cut -d' ' -f1)" = "$hex" ]; then echo "reuse  $d"; continue; fi
  echo "fetch  $d"
  curl -sSfL --retry 5 --retry-delay 3 -C - -m 7200 "https://registry.ollama.ai/v2/library/$name/blobs/$d" -o "$tmp/$hex"
  got="$(shasum -a 256 "$tmp/$hex" | cut -d' ' -f1)"
  [ "$got" = "$hex" ] || { echo "DIGEST MISMATCH for $d (got $got)"; exit 1; }
  mv "$tmp/$hex" "$dest"
done
mkdir -p "$store/manifests/registry.ollama.ai/library/$name"
mv "$tmp/manifest" "$store/manifests/registry.ollama.ai/library/$name/$tag"
echo "installed $name:$tag"
