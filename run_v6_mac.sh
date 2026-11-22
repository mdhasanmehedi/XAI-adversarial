#!/bin/bash
# Functional-validity (v6) re-run of the two replications: BiLSTM (main corpus) and GramBeddings.
# Run from the repository root (the folder that contains src/, runs/, data/ and results/):
#     caffeinate -i bash run_v6_mac.sh
# Optional: WORKERS=3 caffeinate -i bash run_v6_mac.sh      (default 2 jobs at a time)
#           ONLY=bilstm  or  ONLY=gram                     (run just one of the two)
# Every job writes its own file and is skipped on a re-run, so you can stop (Ctrl-C)
# and start again at any time. Results go to results_v6/, logs to results_v6/logs/.
# At the end the script creates results_v6.zip -- send that file.

cd "$(dirname "$0")"
WORKERS=${WORKERS:-2}
ONLY=${ONLY:-all}
mkdir -p results_v6/logs

# ---------- checks ----------
python3 -c "import tldextract" 2>/dev/null || { echo "STOP: run  python3 -m pip install --break-system-packages tldextract"; exit 1; }
python3 -c "import sys; sys.path.insert(0,'src'); from model import build_model" || { echo "STOP: the model import above failed -- send me a screenshot"; exit 1; }
grep -q "URL_VALIDITY_MODE" src/attacks/validity.py || { echo "STOP: src/attacks/validity.py is the old version -- unzip the v5 package again first"; exit 1; }
MISSING=0
for s in 0 1 2; do
  for f in runs/runs_bilstm/model_seed$s.pt data/processed/attack_eval_sample_seed${s}_bilstm.csv \
           runs/runs_gram/model_seed$s.pt data/processed/attack_eval_sample_seed${s}_gram.csv; do
    case "$f" in *bilstm*) [ "$ONLY" = gram ] && continue;; *gram*) [ "$ONLY" = bilstm ] && continue;; esac
    [ -f "$f" ] || { echo "MISSING: $f"; MISSING=1; }
  done
done
[ $MISSING = 1 ] && { echo "STOP: files above are missing -- send me a screenshot"; exit 1; }

# ---------- job list ----------
ENV="cd src && URL_VALIDITY_MODE=functional PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
JOBS=$(mktemp)
for corpus in bilstm gram; do
  [ "$ONLY" != all ] && [ "$ONLY" != "$corpus" ] && continue
  if [ $corpus = bilstm ]; then RUNS=runs_bilstm; else RUNS=runs_gram; fi
  for s in 0 1 2; do
    ES=../data/processed/attack_eval_sample_seed${s}_${corpus}.csv
    CK=../runs/$RUNS/model_seed$s.pt
    OD=../results_v6/$corpus/seed$s
    LG=../results_v6/logs/${corpus}_seed$s
    for m in random_substitution random_insertion random_deletion adjacent_swap segment_manipulation domain_path_tld_transform; do
      echo "$ENV python3 -m attacks.unguided --split_dir x --checkpoint $CK --outdir $OD --eval_sample $ES --max_budget 10 --budgets 1 2 3 5 8 10 --methods $m --seed 0 > ${LG}_unguided_$m.log 2>&1" >> $JOBS
    done
    echo "$ENV python3 -m attacks.unguided_protected --checkpoint $CK --eval_sample $ES --outdir $OD --n_urls 500 --max_budget 10 --seed 0 > ${LG}_unguided_protected.log 2>&1" >> $JOBS
    for prot in "" " --protect_prefix"; do
      if [ -z "$prot" ]; then tag=V0; else tag=V1; fi
      for md in adaptive one_shot; do
        for rk in occlusion ig saliency gradxinput random; do
          echo "$ENV python3 -m attacks.guided_variants --split_dir x --checkpoint $CK --outdir $OD --eval_sample $ES --rankers $rk --modes $md --max_budget 10 --n_urls 500 --ig_steps 20 --seed 0$prot > ${LG}_${tag}_${rk}_${md}.log 2>&1" >> $JOBS
        done
        if [ $corpus = bilstm ]; then
          echo "$ENV python3 -m attacks.hotflip --checkpoint $CK --eval_sample $ES --outdir $OD --modes $md --max_budget 10 --n_urls 500 --seed 0$prot > ${LG}_${tag}_hotflip_${md}.log 2>&1" >> $JOBS
        fi
      done
    done
  done
done

# one small script per job (macOS xargs cannot take the long command lines directly)
JOBSDIR=$(mktemp -d); i=0
while IFS= read -r line; do i=$((i+1)); printf '%s\n' "$line" > "$JOBSDIR/job_$(printf %03d $i).sh"; done < "$JOBS"
echo "$i jobs, $WORKERS at a time. Started $(date '+%H:%M'). This takes many hours; leave the window open."
ls "$JOBSDIR"/job_*.sh | xargs -P "$WORKERS" -I{} bash {}
rm -rf "$JOBSDIR" "$JOBS"

# ---------- finish: copy clean-model faithfulness (it does not depend on the validity check) ----------
for corpus in bilstm gram; do
  for s in 0 1 2; do
    [ -d results_v6/$corpus/seed$s ] || continue
    cp results/$corpus/seed$s/faithfulness_by_ranker*.csv results_v6/$corpus/seed$s/ 2>/dev/null
  done
done
echo
echo "Finished $(date '+%H:%M'). Jobs that failed (if any):"
grep -l "Traceback" results_v6/logs/*.log || echo "  none"
N=$(ls results_v6/*/seed*/variants*/*.csv results_v6/*/seed*/hotflip*/*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "Result files: $N"
rm -f results_v6.zip; zip -qr results_v6.zip results_v6 && echo "Created results_v6.zip -- send it to Claude."
