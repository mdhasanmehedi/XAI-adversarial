#!/bin/bash
# Functional-validity (v5) re-run of the main-corpus attacks.
# Run from the repository root (the folder that contains src/ and runs/):
#     pip install tldextract
#     caffeinate -i bash run_v5_mac.sh          # caffeinate keeps the Mac awake
# Optional: WORKERS=3 caffeinate -i bash run_v5_mac.sh   (default 2 parallel jobs)
#
# No internet is needed: tldextract uses the Public Suffix List bundled with the package.
# Every job writes its own output file when it finishes and is skipped on a re-run,
# so you can stop (Ctrl-C) and start the script again at any time without losing work.
# Results go to results_v5/ ; logs to results_v5/logs/ .

cd "$(dirname "$0")"
WORKERS=${WORKERS:-2}
N=${N:-500}
OUT=../results_v5           # relative to src/
mkdir -p results_v5/logs

python3 -c "import tldextract" 2>/dev/null || { echo "Please run: pip install tldextract"; exit 1; }
python3 -c "import sys; sys.path.insert(0,'src'); from model import build_model" 2>/dev/null || {
  echo "src/model.py has no build_model(): use the src/model.py from this repository."; exit 1; }

# the functional re-run attacks the same 10,000-URL samples as the syntactic run (run_seed_pipeline.sh)
mkdir -p data/eval_v5
for s in 0 1 2 3 4; do
  dst=data/eval_v5/attack_eval_sample_seed$s.csv; [ -f "$dst" ] && continue
  src=data/processed/attack_eval_sample_seed${s}_v1.csv
  [ "$s" = 0 ] && [ ! -f "$src" ] && src=data/processed/attack_eval_sample_v1.csv
  [ -f "$src" ] || { echo "missing $src: run src/run_seed_pipeline.sh $s first"; exit 1; }
  cp "$src" "$dst"
done

ENV="cd src && URL_VALIDITY_MODE=functional OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
JOBS=$(mktemp)
for s in 0 1 2 3 4; do
  ES=../data/eval_v5/attack_eval_sample_seed$s.csv
  OD=$OUT/tables/seed$s
  CK=../runs/runs_v1/model_seed$s.pt
  LG=$OUT/logs/seed$s
  for prot in "" " --protect_prefix"; do
    if [ -z "$prot" ]; then tag=V0; else tag=V1; fi
    for md in adaptive one_shot; do
      for rk in ig occlusion saliency gradxinput random; do
        echo "$ENV python3 -m attacks.guided_variants --split_dir x --checkpoint $CK --outdir $OD --eval_sample $ES --rankers $rk --modes $md --n_urls $N --ig_steps 20 --seed 0$prot > ${LG}_${tag}_${rk}_${md}.log 2>&1" >> $JOBS
      done
      echo "$ENV python3 -m attacks.hotflip --checkpoint $CK --eval_sample $ES --outdir $OD --modes $md --n_urls $N --seed 0$prot > ${LG}_${tag}_hotflip_${md}.log 2>&1" >> $JOBS
    done
  done
  echo "$ENV python3 -m attacks.unguided --split_dir x --checkpoint $CK --outdir $OD --eval_sample $ES --seed 0 > ${LG}_unguided.log 2>&1" >> $JOBS
  echo "$ENV python3 -m attacks.unguided_protected --checkpoint $CK --eval_sample $ES --outdir $OD --n_urls $N --seed 0 > ${LG}_unguided_protected.log 2>&1" >> $JOBS
done

echo "$(wc -l < $JOBS) jobs, $WORKERS in parallel. Progress: ls results_v5/tables/seed*/variants*/ | wc -l"
tr '\n' '\0' < $JOBS | xargs -0 -P $WORKERS -I{} bash -c "{}"
rm -f $JOBS

echo
echo "Finished. Jobs that failed (if any):"
grep -l "Traceback" results_v5/logs/*.log || echo "  none"
echo "Now run:  bash analyze_v5.sh   and send me results_v5 as a zip."
