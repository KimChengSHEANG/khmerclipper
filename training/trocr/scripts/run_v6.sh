#!/bin/sh
# v6: continue from v3 as shipped (step 16000) with handwriting-looking strips (synth.render_handwriting_sample:
# Fasthand / Freehand, elastic wobble, wavy baseline, uneven pen weight, blue / black ink on white / cream paper,
# sometimes ruled, scanned) next to the real handwriting set, over v4's replay mix minus synth_screen_v4.
#   nohup sh scripts/run_v6.sh > experiments/finetune_v6.log 2>&1 &
# Gate (training/eval/results/handwriting_v6.md): real handwriting_test CER better; frozen 200, screen 450,
# slices 320 and noisy 300 no more than 3% worse than v3 in the app pipeline; then on device before shipping.
set -e
V3=${V3:-experiments/v3_16k}
[ -d "$V3" ] || { echo "v3 checkpoint not found at $V3 (set V3=...)"; exit 1; }
for i in 0 1 2 3 4 5; do
  [ -n "$(ls cache/synth_hand/train/p$i/*.parquet 2>/dev/null)" ] && continue
  python scripts/render_synth.py --handwriting --max_chars 24 --out cache/synth_hand/train/p$i --count 20000 \
    --seed $((600 + i)) &
done
wait
python scripts/train.py --model_name "$V3" --output_dir experiments/v6 \
  --datasets synth_hand:80000 handwriting:20000 synth_screen:60000 synth_en:100000 lazygreed:30000 synth:40000 \
             darayut:15000 siemreap:15000 soyvitou:25000 sokheng:15000 hanuman:15000 kiteaether:15000 mrrtmob:50000 \
  --augment --epochs 1 --lr 5e-6 --warmup_steps 500 \
  --max_eval_per_source 300 --max_eval_samples 0 --eval_steps 2000 --early_stopping_patience 4 --max_aspect 32
last=$(ls -d experiments/v6/checkpoint-* | sort -t- -k2 -n | tail -1)
mkdir -p experiments/v6_last
cp experiments/v6/final/*.json experiments/v6/final/merges.txt experiments/v6_last/
cp "$last/config.json" "$last/generation_config.json" "$last/model.safetensors" experiments/v6_last/
# Real handwriting (never trained on: the test split), v3 vs both v6 candidates.
for m in "$V3" experiments/v6_last experiments/v6/final; do
  echo "HANDWRITING $m"; python scripts/evaluate_model.py "$m" --datasets handwriting_test --full 2>&1 | grep -i "cer" | tail -2
done
cd ../eval
for m in v3 v6_last v6_final; do
  dir=../trocr/$V3; [ $m = v6_last ] && dir=../trocr/experiments/v6_last; [ $m = v6_final ] && dir=../trocr/experiments/v6/final
  for fx in fixtures fixtures_screen fixtures_slices fixtures_noisy; do
    [ -s results/gate6_${m}_$fx.csv ] || python scripts/bench.py --fixtures $fx --engine trocr-pipeline --trocr-model-dir $dir \
      --variant split_touching=1,split_wide_at=12,crop=1,robust_bg=1,drop_noise=1,drop_edge=1,local_bg=1,letters_all=1,max_gap=1,columns=1,big_gap=1 \
      --out results/gate6_${m}_$fx.csv
  done
done
echo GATE_DONE
