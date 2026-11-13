#!/bin/bash
# Fetches the public test sets the evaluation scripts read, into $CUT_ROOT/data (default: ./data).
# All are open-licensed: GSM8K (MIT), MATH-500 (MIT), IFEval (Apache-2.0), MMLU (MIT), BIG-Bench Hard (MIT).
# The files are data: they are only ever read, never executed (the IFEval checker is Python code from Google's
# repository; read setup_ifeval.sh before running it).
# Afterwards, from inside the data folder, check them against the hashes of the copies behind the reported numbers:
#   cd "$CUT_ROOT/data" && sha256sum -c ../DATA_SHA256.txt        (setup_workdir.py puts DATA_SHA256.txt in $CUT_ROOT)
# A different hash means the upstream file changed; the numbers may then differ from the published ones.
# Usage: fetch_data.sh
set -e
D="${CUT_ROOT:-$PWD}/data"
mkdir -p "$D/gsm8k" "$D/math500" "$D/ifeval" "$D/mmlu" "$D/bbh" "$D/bbh_hard"

get() {  # get URL DESTINATION
  if curl -sSfL --proto =https --retry 3 -o "$2" "$1"; then
    echo "ok $(stat -c %s "$2") $2"
  else
    echo "FAILED $1" >&2
    exit 1
  fi
}

GSM=https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data
get $GSM/test.jsonl "$D/gsm8k/test.jsonl"
get $GSM/train.jsonl "$D/gsm8k/train.jsonl"

get https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl "$D/math500/test.jsonl"

IFE=https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval
get $IFE/data/input_data.jsonl "$D/ifeval/input_data.jsonl"
for f in instructions.py instructions_registry.py instructions_util.py evaluation_lib.py; do
  get $IFE/$f "$D/ifeval/$f"
done

get https://people.eecs.berkeley.edu/~hendrycks/data.tar "$D/mmlu/data.tar"
tar -xf "$D/mmlu/data.tar" -C "$D/mmlu"
echo "ok mmlu extracted"

BBH=https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh
for t in boolean_expressions formal_fallacies logical_deduction_five_objects web_of_lies \
         tracking_shuffled_objects_three_objects date_understanding; do
  get $BBH/$t.json "$D/bbh/$t.json"
done

# the harder logic tasks of the long-budget test: raw files, HTTPS only, no redirects, 2 MB ceiling per file
for t in logical_deduction_seven_objects tracking_shuffled_objects_seven_objects multistep_arithmetic_two \
         dyck_languages; do
  if curl -sSf --proto =https --max-redirs 0 --max-filesize 2000000 --retry 3 -o "$D/bbh_hard/$t.json" "$BBH/$t.json"; then
    echo "ok $(stat -c %s "$D/bbh_hard/$t.json") $t.json"
  else
    echo "FAILED $t" >&2
    exit 1
  fi
done
du -sh "$D"
