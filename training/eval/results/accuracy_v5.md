# v5 fine-tune: busy backgrounds (2026-10-27): dropped

`training/trocr/scripts/run_v5.sh`: from shipped v3 (step 16000), lr 5e-6, 1 epoch (15,884 steps, 2.3 h), adding
120k `synth_screen_bg` strips (screen strips set on generated gradient / blob / fractal-noise / two-region panels,
`synth.busy_background`) over v4's replay mix minus `synth_screen_v4`. Target: `noisy_background`, CER 0.31 on v3.
Gated with the PyTorch app pipeline (split touching + wide split 12 + crop + robust background + noise and edge
fragment removal), paired character errors vs v3 in the same run.

| Set | v3 | v5 last (15884) | v5 best mid-train |
|---|---:|---:|---:|
| frozen 200 | 35 | 39 (+11.4%) | 34 (-2.9%) |
| screen 450 | 716 | 741 (+3.5%) | 729 (+1.8%) |
| slices 320 | 164 | 157 (-4.3%) | 173 (+5.5%) |
| noisy 300 | 3208 | 3212 (+0.1%) | 3192 (-0.5%) |
| - background | 1438 | 1464 (+1.8%) | 1459 (+1.5%) |
| - avatar | 236 | 199 (-15.7%) | 192 (-18.6%) |
| - underline | 134 | 161 (+20.1%) | 135 (+0.7%) |
| - chrome | 165 | 163 | 177 (+7.3%) |

Bar (background -20%, nothing else worse than +3%): both fail. Background did not move at all.

## Why: the background errors are segmentation, not the model

On-device v3 (CoreML), the 50 `noisy_background` clips split by whether `line_strips` (the app's line splitter)
finds the right number of lines:

| Strips | clips | mean CER |
|---|---:|---:|
| right count | 26 | 0.005 |
| wrong count | 24 | 0.651 |

Wrong counts go both ways: 2-3 lines on a gradient come out as one strip (the ink mask, measured against one
background colour, covers the whole panel), and one Latin word on a texture comes out as 3 strips, which the
model fills with invented words ("Bluetooth" -> four words). Given the right strips the model already reads text
on busy panels. The next lever is a background-aware ink mask in preprocessing (local / per-row background
estimate before line splitting), in eval, Swift and desktop together, as the noise-robust preprocessing work did.
