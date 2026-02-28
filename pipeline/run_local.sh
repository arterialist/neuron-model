#!/usr/bin/env bash
# Run the pipeline API server natively (without Docker).
# Use this to enable Apple Metal (MPS) and other local GPU acceleration.
#
# From project root:
#   ./pipeline/run_local.sh
#
# Or with custom output dir:
#   PIPELINE_OUTPUT_DIR=./my_experiments ./pipeline/run_local.sh
#
# Custom port (default 8000):
#   PIPELINE_PORT=8080 ./pipeline/run_local.sh
#   PORT=8080 ./pipeline/run_local.sh

set -e
cd "$(dirname "$0")/.."

# Ensure pipeline_db and experiments exist
mkdir -p pipeline_db experiments

# Run from project root so paths resolve correctly
exec python -m pipeline.api.main
