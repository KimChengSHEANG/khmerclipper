#!/bin/sh
# v4: continue from v3 as shipped (checkpoint-8 step 16000) with screen strips aimed at v3's weakest on-device
# slices (training/eval/results/accuracy_v3.md, Follow-ups): text <= 13 px, bold, dark panels,
# and Latin ':' next to Khmer (read as U+17C8). Replays the v3b mix at lower weights so
# English UI accuracy and the other domains hold.
#   nohup sh scripts/run_v4.sh > experiments/finetune_v4.log 2>&1 &
# Ship gate (same as v3): paired errors on frozen 200, screen 450 and slices 320 no worse, and
# English UI line accuracy >= v3 (eval_english.py), before converting for the apps.
set -e
# checkpoint-8/final is the best mid-train step (8k), not the shipped 16k; v3_16k is
# checkpoint-16000's weights plus final/'s tokenizer files (same layout as v2_10k).
V3=${V3:-experiments/v3_16k}
[ -d "$V3" ] || { echo "v3 checkpoint not found at $V3 (set V3=...)"; exit 1; }
for i in 0 1 2 3 5 6 7 8; do
  python scripts/render_synth.py --screen --out cache/synth_screen_v4/train/p$i --count 20000 --seed $((400 + i)) \
    --screen_sizes 10 14 --screen_dark 0.5 --screen_bold 0.35 --colon_prefix 0.1 &
done
wait
exec python scripts/train.py --model_name "$V3" \
  --datasets synth_screen_v4:120000 synth_screen:60000 synth_en:100000 lazygreed:30000 synth:40000 \
             handwriting:10000 darayut:15000 siemreap:15000 soyvitou:25000 sokheng:15000 hanuman:15000 \
             kiteaether:15000 mrrtmob:50000 \
  --augment --epochs 1 --lr 5e-6 --warmup_steps 500 \
  --max_eval_per_source 300 --max_eval_samples 0 --eval_steps 2000 --early_stopping_patience 4 --max_aspect 32
