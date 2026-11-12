#!/bin/sh
# v8b: v8 plus Open Development Cambodia law scans pseudo-labelled by Chakkhu at conf >= 0.97
# (scripts/odc_pseudo.py, registry "odc"), with a third of v8's mix replayed so nothing else drifts.
# On odc_test M2 reads 11.1% CER against those labels, v8 10.4%; Chakkhu at that confidence is 1.2% on KHOB.
#   nohup sh scripts/run_v8b.sh > experiments/finetune_v8b.log 2>&1 &
# Gate as v8 (run_v8.sh): REAL sets and the app pipeline against M2; then KHOB on device.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M2=${M2:-experiments/v7b_last}
python scripts/train.py --model_name experiments/v8_last --output_dir experiments/v8b \
  --datasets odc:30000 synth_doc:30000 thareah:5000 \
             synth_v7:20000 synth_screen_v7:20000 synth_hand_v7:10000 handwriting:10000 synth_screen:15000 \
             synth_en:25000 lazygreed:8000 darayut:4000 siemreap:4000 soyvitou:8000 sokheng:4000 hanuman:4000 \
             kiteaether:4000 mrrtmob:25000 \
  --augment --epochs 1 --lr 5e-6 --warmup_steps 300 \
  --max_eval_per_source 300 --max_eval_samples 0 --eval_steps 2000 --early_stopping_patience 4 --max_aspect 32 \
  ${RESUME:+--resume_from_checkpoint "$RESUME"}
last=$(ls -d experiments/v8b/checkpoint-* | sort -t- -k2 -n | tail -1)
mkdir -p experiments/v8b_last
cp experiments/v8b/final/*.json experiments/v8b/final/merges.txt experiments/v8b_last/
cp "$last/config.json" "$last/generation_config.json" "$last/model.safetensors" experiments/v8b_last/
for d in handwriting_test thareah_test odc_test; do
  echo "REAL $d v8b"; python scripts/evaluate_model.py experiments/v8b_last --datasets $d --full 2>&1 | grep -i "cer" | tail -2
done
cd ../eval
for fx in fixtures fixtures_screen fixtures_slices fixtures_noisy; do
  python scripts/bench.py --fixtures $fx --engine trocr-pipeline --trocr-model-dir ../trocr/experiments/v8b_last \
    --variant split_touching=1,split_wide_at=12,crop=1,robust_bg=1,drop_noise=1,drop_edge=1,local_bg=1,letters_all=1,max_gap=1,columns=1,big_gap=1 \
    --out results/gate8_v8b_$fx.csv
done
echo GATE_DONE
