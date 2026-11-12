#!/bin/sh
# v8: v7b's mix plus what Chakkhu trains on and we didn't (training/eval/results/weight_space.md):
# document lines at scan resolution with neighbour slivers, Khmer Wikipedia text (synth_doc,
# render_synth.py --doc --wiki_km), real handwriting from Thareah/khmer-handwriting (MIT), and more
# khmer_english_ocr_image_line (CC BY 4.0). KHOB stays a test set. Starts from v7b_last (M2).
#   nohup sh scripts/run_v8.sh > experiments/finetune_v8.log 2>&1 &
# Gate: thareah_test / handwriting_test better than M2; app pipeline frozen / screen / slices / noisy no more
# than 3% worse; then KHOB and on device against M2 and Chakkhu.
# RESUME=experiments/v8/checkpoint-N continues a run. expandable_segments: after an eval, 2.7 GB sat reserved
# but fragmented and a backward pass ran out of memory at step 4001 (batch 32, 24 GB).
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M2=${M2:-experiments/v7b_last}
python scripts/fetch_fonts.py
[ -n "$(ls cache/thareah/test/*.parquet 2>/dev/null)" ] || python scripts/thareah_lines.py
for i in 0 1 2 3; do
  [ -n "$(ls cache/synth_doc/train/p$i/*.parquet 2>/dev/null)" ] || \
    python scripts/render_synth.py --doc --wiki_km 300000 --max_chars 64 --out cache/synth_doc/train/p$i \
      --count 30000 --seed $((800 + i)) > experiments/render_doc_p$i.log 2>&1 &
done
wait
python scripts/train.py --model_name "$M2" --output_dir experiments/v8 \
  --datasets synth_doc:120000 thareah:15000 \
             synth_v7:80000 synth_screen_v7:80000 synth_hand_v7:50000 synth_hand:30000 handwriting:20000 \
             synth_screen:60000 synth_en:100000 lazygreed:30000 darayut:15000 siemreap:15000 soyvitou:25000 \
             sokheng:15000 hanuman:15000 kiteaether:15000 mrrtmob:100000 \
  --augment --epochs 1 --lr 5e-6 --warmup_steps 500 \
  --max_eval_per_source 300 --max_eval_samples 0 --eval_steps 2000 --early_stopping_patience 4 --max_aspect 32 \
  ${RESUME:+--resume_from_checkpoint "$RESUME"}
last=$(ls -d experiments/v8/checkpoint-* | sort -t- -k2 -n | tail -1)
mkdir -p experiments/v8_last
cp experiments/v8/final/*.json experiments/v8/final/merges.txt experiments/v8_last/
cp "$last/config.json" "$last/generation_config.json" "$last/model.safetensors" experiments/v8_last/
for m in "$M2" experiments/v8_last; do
  for d in handwriting_test thareah_test; do
    echo "REAL $d $m"; python scripts/evaluate_model.py "$m" --datasets $d --full 2>&1 | grep -i "cer" | tail -2
  done
done
cd ../eval
for m in m2 v8_last; do
  dir=../trocr/$M2; [ $m = v8_last ] && dir=../trocr/experiments/v8_last
  for fx in fixtures fixtures_screen fixtures_slices fixtures_noisy; do
    [ -s results/gate8_${m}_$fx.csv ] || python scripts/bench.py --fixtures $fx --engine trocr-pipeline --trocr-model-dir $dir \
      --variant split_touching=1,split_wide_at=12,crop=1,robust_bg=1,drop_noise=1,drop_edge=1,local_bg=1,letters_all=1,max_gap=1,columns=1,big_gap=1 \
      --out results/gate8_${m}_$fx.csv
  done
done
echo GATE_DONE
