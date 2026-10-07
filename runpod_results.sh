#!/bin/bash
# Copy training results from a RunPod pod to your machine with runpodctl.
#
# The pod has no persistent volume — its container disk is erased when the pod
# stops — so pull results before stopping it. runpodctl transfers peer-to-peer
# with a one-time code; no SSH setup needed.
#
# On the pod (bundles checkpoints/ and logs/, prints a code, waits for receiver):
#   bash runpod_results.sh send                  # name defaults to a timestamp
#   bash runpod_results.sh send run005           # name the bundle
#   bash runpod_results.sh send run005 extra.txt # add more paths
#
# On your machine (downloads and unpacks into runs/<bundle-name>/):
#   bash runpod_results.sh receive <code>
#
# runpodctl: preinstalled on RunPod pods. On macOS:
#   brew install runpod/runpodctl/runpodctl

set -euo pipefail
cd "$(dirname "$0")"

die() { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

command -v runpodctl >/dev/null 2>&1 || die "runpodctl not found.
  macOS:  brew install runpod/runpodctl/runpodctl
  Linux:  see https://github.com/runpod/runpodctl#install"

MODE="${1:-}"; shift || true

case "$MODE" in
send)
    NAME="${1:-$(date +%Y%m%d_%H%M%S)}"; shift || true
    PATHS=()
    for p in checkpoints logs "$@"; do
        [ -e "$p" ] && PATHS+=("$p")
    done
    [ ${#PATHS[@]} -gt 0 ] || die "nothing to send — no checkpoints/ or logs/ here"

    BUNDLE="results_${NAME}.tar.gz"
    echo "[send] Bundling: ${PATHS[*]}"
    tar czf "$BUNDLE" "${PATHS[@]}"
    echo "[send] $BUNDLE ($(du -h "$BUNDLE" | cut -f1)) — contents:"
    tar tzf "$BUNDLE" | grep -v '/$' | sed 's/^/  /'
    echo ""
    echo "[send] On your machine, run:  bash runpod_results.sh receive <code shown below>"
    echo "       (this waits until the transfer completes; keep the pod running)"
    runpodctl send "$BUNDLE"
    rm -f "$BUNDLE"
    ;;

receive)
    CODE="${1:-}"
    [ -n "$CODE" ] || die "usage: bash runpod_results.sh receive <code>"
    mkdir -p runs
    STAGE="$(mktemp -d runs/.incoming.XXXXXX)"
    trap 'rm -rf "$STAGE"' EXIT
    echo "[receive] Downloading..."
    (cd "$STAGE" && runpodctl receive "$CODE")

    BUNDLE=$(find "$STAGE" -maxdepth 1 -name 'results_*.tar.gz' | head -1)
    [ -n "$BUNDLE" ] || die "no results_*.tar.gz received (wrong code, or a file not made by 'send')"
    NAME="$(basename "$BUNDLE" .tar.gz)"; NAME="${NAME#results_}"
    DEST="runs/$NAME"
    if [ -e "$DEST" ]; then DEST="${DEST}_$(date +%H%M%S)"; fi  # never overwrite a previous pull
    mkdir -p "$DEST"
    tar xzf "$BUNDLE" -C "$DEST"
    echo "[receive] Saved to $DEST:"
    find "$DEST" -type f | sort | while read -r f; do
        printf '  %-60s %s\n' "${f#"$DEST"/}" "$(du -h "$f" | cut -f1)"
    done
    ;;

*) usage ;;
esac
