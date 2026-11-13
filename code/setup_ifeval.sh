#!/bin/bash
# Builds $CUT_ROOT/ifeval_pkg/instruction_following_eval from the Google IFEval files that fetch_data.sh put in
# data/ifeval, with the two small edits in ifeval_patch.diff (regex sentence and word splitting instead of NLTK
# tokenizers), which is how the reported IFEval scores were computed. The checker still imports nltk, langdetect,
# absl-py and immutabledict, so install those first (pip install nltk langdetect absl-py immutabledict).
# step4_eval.py finds the package at $CUT_ROOT/ifeval_pkg.
# Usage: setup_ifeval.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
R="${CUT_ROOT:-$PWD}"
PKG="$R/ifeval_pkg/instruction_following_eval"
mkdir -p "$PKG"
cp "$R"/data/ifeval/*.py "$PKG/"
touch "$PKG/__init__.py"
patch -p1 -d "$PKG" -i "$HERE/ifeval_patch.diff"
echo "IFEval checker ready in $PKG"
