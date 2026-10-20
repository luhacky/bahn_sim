#!/usr/bin/env bash
# Set up the S-Bahn simulation on this Linux / macOS machine. Needs Python 3.8+.
cd "$(dirname "$0")"
exec python3 setup.py "$@"
