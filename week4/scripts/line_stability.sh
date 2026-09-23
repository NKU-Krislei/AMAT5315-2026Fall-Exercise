#!/bin/sh
# Draw evidence/line-stability.png from the week4 library.
cd "$(dirname "$0")/.."
cargo run --example line_stability --release
