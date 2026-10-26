#!/bin/sh
# v5: continue from v3 as shipped (step 16000) with screen strips set on busy panels (photos, gradients,
# textures; synth.busy_background). On the noisy eval set v3 invents words around text on such panels
# (background CER 0.31). Same replay as v4 minus synth_screen_v4, whose faux-bold strips hurt bold text.
#   nohup sh scripts/run_v5.sh > experiments/finetune_v5.log 2>&1 &
# Ship gate (training/eval/results/accuracy_v5.md): noisy_background errors -20% or better; frozen 200,
# screen 450, slices 320 and the other noisy kinds no more than 3% worse; English UI line accuracy and
# the verified real captures no worse. Then on device (CoreML) before shipping.
set -e
V3=${V3:-experiments/v3_16k}
[ -d "$V3" ] || { echo "v3 checkpoint not found at $V3 (set V3=...)"; exit 1; }
for i in 0 1 2 3 4 5; do
  [ -n "$(ls cache/synth_screen_bg/train/p$i/*.parquet 2>/dev/null)" ] && continue
  python scripts/render_synth.py --screen --out cache/synth_screen_bg/train/p$i --count 20000 --seed $((500 + i)) \
    --screen_busy 1.0 &
done
wait
python scripts/train.py --model_name "$V3" --output_dir experiments/v5 \
  --datasets synth_screen_bg:120000 synth_screen:60000 synth_en:100000 lazygreed:30000 synth:40000 \
             handwriting:10000 darayut:15000 siemreap:15000 soyvitou:25000 sokheng:15000 hanuman:15000 \
             kiteaether:15000 mrrtmob:50000 \
  --augment --epochs 1 --lr 5e-6 --warmup_steps 500 \
  --max_eval_per_source 300 --max_eval_samples 0 --eval_steps 2000 --early_stopping_patience 4 --max_aspect 32
# final/ is the best mid-train step; the last step won for v2 and v3, so gate both. v5_last = the
# last checkpoint's weights with final/'s tokenizer files (same layout as v3_16k).
last=$(ls -d experiments/v5/checkpoint-* | sort -t- -k2 -n | tail -1)
mkdir -p experiments/v5_last
cp experiments/v5/final/*.json experiments/v5/final/merges.txt experiments/v5_last/
cp "$last/config.json" "$last/generation_config.json" "$last/model.safetensors" experiments/v5_last/
# PyTorch app pipeline, v3 and both v5 candidates in the same run (paired with scripts/compare.py).
cd ../eval
for m in v3 v5_last v5_final; do
  dir=../trocr/$V3; [ $m = v5_last ] && dir=../trocr/experiments/v5_last; [ $m = v5_final ] && dir=../trocr/experiments/v5/final
  for fx in fixtures fixtures_screen fixtures_slices fixtures_noisy; do
    [ -s results/gate5_${m}_$fx.csv ] || python scripts/bench.py --fixtures $fx --engine trocr-pipeline \
      --trocr-model-dir $dir --variant split_touching=1,split_wide_at=12,crop=1,robust_bg=1,drop_noise=1,drop_edge=1 \
      --out results/gate5_${m}_$fx.csv
  done
done
echo GATE_DONE
