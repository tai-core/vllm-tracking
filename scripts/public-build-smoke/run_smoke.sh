#!/usr/bin/env bash
# Codex entrypoint: synthetic no-weight smoke inside the built container.
# Gate levels: IMPORT / COMPILE / CPU / GPU (see the python script).
# No weights, no network, no downloads.
set -uo pipefail
python "$(dirname "$0")/smoke_import_and_kernels.py"
