#!/bin/sh
# v7: more fonts and real scans. Adds the Khmer OS family (LGPL; synth.KHMER_OS_FONTS) to every synthetic render
# (general lines, screen strips, handwriting), and real scanned textbook lines from CADT-IDRI/khmer-sala-ocr-16k
# (CC BY-NC 4.0, scripts/cadt_lines.py). Starts from v3 as shipped. Waits for v6's run to finish (GPU).
#   nohup sh scripts/run_v7.sh > experiments/finetune_v7.log 2>&1 &
# Gate (training/eval/results/fonts_cadt_v7.md): real handwriting_test and cadt_test CER better than v3; app
# pipeline frozen / screen / slices / noisy no more than 3% worse, reported separately for the held-out Apple
# fonts (Khmer MN, Khmer Sangam MN) and the now-trained Khmer OS fonts; then on device before shipping.
set -e
while pgrep -f "[r]un_v6.sh" >/dev/null; do sleep 120; done
V3=${V3:-experiments/v3_16k}
python scripts/fetch_fonts.py
for i in 0 1 2 3; do
  [ -n "$(ls cache/synth_v7/train/p$i/*.parquet 2>/dev/null)" ] || \
    python scripts/render_synth.py --out cache/synth_v7/train/p$i --count 25000 --seed $((700 + i)) &
done
for i in 0 1 2 3; do
  [ -n "$(ls cache/synth_screen_v7/train/p$i/*.parquet 2>/dev/null)" ] || \
    python scripts/render_synth.py --screen --max_chars 64 --out cache/synth_screen_v7/train/p$i --count 25000 --seed $((710 + i)) &
done
for i in 0 1; do
  [ -n "$(ls cache/synth_hand_v7/train/p$i/*.parquet 2>/dev/null)" ] || \
    python scripts/render_synth.py --handwriting --max_chars 24 --out cache/synth_hand_v7/train/p$i --count 30000 --seed $((720 + i)) &
done
[ -n "$(ls cache/cadt/test/*.parquet 2>/dev/null)" ] || \
  python scripts/cadt_lines.py --split test --model "$V3" --out cache/cadt/test --max_cer 1.0
[ -n "$(ls cache/cadt/train/*.parquet 2>/dev/null)" ] || \
  python scripts/cadt_lines.py --split train --model "$V3" --out cache/cadt/train --max_cer 0.3 --limit 6000
wait
python scripts/train.py --model_name "$V3" --output_dir experiments/v7 \
  --datasets synth_v7:80000 synth_screen_v7:80000 synth_hand_v7:50000 synth_hand:30000 handwriting:20000 cadt:60000 \
             synth_screen:40000 synth_en:100000 lazygreed:30000 darayut:15000 siemreap:15000 soyvitou:25000 \
             sokheng:15000 hanuman:15000 kiteaether:15000 mrrtmob:50000 \
  --augment --epochs 1 --lr 5e-6 --warmup_steps 500 \
  --max_eval_per_source 300 --max_eval_samples 0 --eval_steps 2000 --early_stopping_patience 4 --max_aspect 32
last=$(ls -d experiments/v7/checkpoint-* | sort -t- -k2 -n | tail -1)
mkdir -p experiments/v7_last
cp experiments/v7/final/*.json experiments/v7/final/merges.txt experiments/v7_last/
cp "$last/config.json" "$last/generation_config.json" "$last/model.safetensors" experiments/v7_last/
for m in "$V3" experiments/v7_last experiments/v7/final; do
  for d in handwriting_test cadt_test; do
    echo "REAL $d $m"; python scripts/evaluate_model.py "$m" --datasets $d --full 2>&1 | grep -i "cer" | tail -2
  done
done
cd ../eval
for m in v3 v7_last v7_final; do
  dir=../trocr/$V3; [ $m = v7_last ] && dir=../trocr/experiments/v7_last; [ $m = v7_final ] && dir=../trocr/experiments/v7/final
  for fx in fixtures fixtures_screen fixtures_slices fixtures_noisy; do
    [ -s results/gate7_${m}_$fx.csv ] || python scripts/bench.py --fixtures $fx --engine trocr-pipeline --trocr-model-dir $dir \
      --variant split_touching=1,split_wide_at=12,crop=1,robust_bg=1,drop_noise=1,drop_edge=1,local_bg=1,letters_all=1,max_gap=1,columns=1,big_gap=1 \
      --out results/gate7_${m}_$fx.csv
  done
done
echo GATE_DONE
