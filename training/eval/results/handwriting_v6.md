# v6 fine-tune: synthetic handwriting (2026-10-27): dropped

`training/trocr/scripts/run_v6.sh`: from shipped v3 (step 16000), lr 5e-6, 1 epoch, adding 120k `synth_hand` strips
(`synth.render_handwriting_sample`: Fasthand / Freehand, elastic wobble, wavy baseline, uneven pen weight, blue / black
ink on white / cream paper, sometimes ruled, scanned) with the real 4.2k handwriting set at 20k, over v4's replay mix.

| Model | real handwriting_test CER (307) | frozen 200 | screen 450 | slices 320 | noisy 300 |
|---|---:|---:|---:|---:|---:|
| v3 (shipped) | 0.0071 | 0.0070 | 0.0158 | 0.0108 | 0.0862 |
| v6 last | 0.0071 | 0.0077 | 0.0162 | 0.0105 | 0.0931 |
| v6 best mid-train | 0.0074 | 0.0080 | 0.0164 | 0.0105 | 0.0891 |

(PyTorch app pipeline, mean CER per clip.) Dropped: no gain on real handwriting, worse on frozen and noisy.

The real handwriting test split is saturated: v3 already reads it at 0.7% CER. It comes from the same 4.2k-line
source v3 trained on (same writers, short words), so it can't show a handwriting gain. The user's own handwriting
capture (real-001, faint pen on lined paper) is the kind of case this targets, and it needs real, varied
handwriting to measure, which no openly licensed source provides (see the v7 notes).
