#!/usr/bin/env bash
# Gen chain -> clean the generated files -> gate coverage -> retrain.  (v2 of chain_to_retrain.sh)
#
# NEW FILE ON PURPOSE, not an edit of chain_to_retrain.sh: bash reads a script incrementally by BYTE
# OFFSET, so editing a running script makes it resume mid-line. That already happened in this session --
# run_gen.sh was edited at 14:57 while the 14:25 testset run was live, and bash resumed inside `ln -f`,
# reporting `n -f ... syntax error near ;;`. Nothing was lost (the testset target had no step after the
# merge) but the lesson stands: a running .sh is immutable.
#
# WHAT v2 ADDS: a drop_empty_answers pass before each coverage gate.
# gen_mids_vllm.py emits a record with `answers: []` for every deliberate lowq_real drop (27 of them in
# the testset). train_mids4c.py:152-155 reads item['answers'] with no length check, and the train step
# flattens a batch's answers then re-slices by logits.size(1) -- so a single 0-answer record shifts
# every LATER item in its batch onto the wrong answers. That is silent supervision corruption, so those
# records are removed and the gate now fails if any survive.
#
# The hardlinks MUST be re-created after the drop: run_gen.sh links dataset/valset.json ->
# work/mids_qwen_testset.json, and the drop replaces the file via os.replace (a NEW inode), which would
# otherwise leave run_finetuning.sh reading the OLD uncleaned inode.
set -uo pipefail
cd /datasets/work/vLLM/temp/EVAL_SPACE
PY=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
GENPID=${1:?gen chain pid required}

echo "[chain2] waiting for the gen chain (pid $GENPID)"
while [ -d "/proc/$GENPID" ]; do sleep 60; done
echo "[chain2] gen chain finished $(date '+%F %T')"

# name : image manifest : merged file : shard base : hardlink name run_finetuning.sh reads ('-' = none)
for s in testset:work/testset_clean_images.json:dataset/testset.json:work/mids_test:- \
         valset:work/valset_images.json:dataset/valset.json:work/mids_val:work/mids_qwen_testset.json \
         trainset:work/trainset_final_images.json:dataset/trainset.json:work/mids_train:work/mids_qwen_train.json; do
  IFS=: read -r name imgs merged base link <<< "$s"
  [ -f "$merged" ] || { echo "[chain2] ABORT: $merged was never produced"; exit 1; }

  echo; echo "############ clean($name) ############"
  $PY drop_empty_answers.py "$merged" --report "runs/drop_empty_${name}.json" \
    || { echo "[chain2] ABORT: drop_empty_answers($name) failed"; exit 1; }
  if [ "$link" != "-" ]; then
    ln -f "$merged" "$link" || { echo "[chain2] ABORT: could not re-link $link"; exit 1; }
    a=$(stat -c %i "$merged"); b=$(stat -c %i "$link")
    [ "$a" = "$b" ] || { echo "[chain2] ABORT: $link inode $b != $merged inode $a"; exit 1; }
    echo "[chain2] $link -> same inode as $merged ($a)"
  fi

  echo "############ coverage($name) ############"
  $PY check_gen_coverage.py --images "$imgs" --merged "$merged" \
      --errors "${base}_*.json.errors.json" --min-cov 0.95 \
      --out "runs/cov_${name}.json" || { echo "[chain2] ABORT: coverage($name) failed"; exit 1; }
done

echo; echo "############ launching retrain $(date '+%F %T') ############"
exec bash retrain_all.sh
