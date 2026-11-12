# Literature Search Report: Khmer OCR (related work for KhmerClipper / M3)

ARS `academic-paper` `lit-review` mode, integrative form. Discipline: computer science (document analysis / OCR).
Domain evidence profile: `cs_ml`, so arXiv preprints, proceedings and model/dataset cards are admissible and tagged
by evidence type. Citation style: IEEE. Companion section: [related_work.md](related_work.md).

## Search Strategy

- **Last searched:** 2026-11-12.
- **Databases:**
  - OpenAlex: the primary index; it covers Crossref, arXiv and most IEEE, ACM and Springer proceedings.
  - Semantic Scholar: used only for one batch lookup. Its keyword search refused anonymous requests (HTTP 429).
  - arXiv: for abstracts.
  - Hugging Face Hub API and GitHub API: for the dataset, model and benchmark cards.
- **Window:** 2015-2026, plus foundational works outside it (Jaderberg et al. 2014).
- **Language:** English. No Khmer-language venues were searched; that limitation is listed under the gaps.
- **Queries:** 16 broad searches on OpenAlex, 50 results each, `from_publication_date:2015-01-01`.

  | # | Query | Hits |
  |---|---|---|
  | 1 | Khmer optical character recognition | 222 |
  | 2 | Khmer text recognition | 3,262 |
  | 3 | Khmer character recognition | 2,356 |
  | 4 | Khmer handwritten | 370 |
  | 5 | Khmer script deep learning | 1,108 |
  | 6 | Khmer palm leaf manuscript | 305 |
  | 7 | Thai optical character recognition deep learning | 5,237 |
  | 8 | Lao optical character recognition | 1,163 |
  | 9 | Myanmar optical character recognition | 909 |
  | 10 | Indic text recognition deep learning | 6,071 |
  | 11 | TrOCR transformer optical character recognition | 951 |
  | 12 | transformer handwritten text line recognition | 6,695 |
  | 13 | synthetic data scene text recognition | 53,165 |
  | 14 | pseudo-labeling handwritten text recognition | 4,261 |
  | 15 | OCR low-resource languages benchmark | 8,757 |
  | 16 | quantization transformer OCR mobile | 730 |

- **Targeted follow-ups:** six searches, 15 results each, from 2014: Jaderberg synthetic data, integer quantization, Myanmar printed text, Thai printed text, abugida scripts, and KHOB.
- **Seeds from the repo:**
  - KHOB.
  - Chakkhu-OCR-Khmer.
  - TrOCR.
  - The four Hugging Face datasets the model trains on.
- **Citation chaining:** limited to the Hugging Face and GitHub cards, which name KHOB, Chakkhu and the Netra Lab model. No reference lists were mined, because few full texts were accessible (see Screening).

## Coverage Distribution Advisory

```
DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: time distribution
- Concentration: 2021-2026 = 33/42 (79%)
- Advisory: This is a coverage-distribution signal, not a defect. Khmer deep-learning OCR is recent work; pre-2015
  Khmer OCR (feature-based, e.g. Zernike moments) was screened out as outside the window.
- Search response: no expansion; the related-work section is about current recognizers.
```

The methodological (model papers 24/42, 57%), geographic and venue-tier dimensions don't trigger an advisory. Khmer work comes from Cambodian, Japanese, French, Belgian and Vietnamese groups across IEEE, Springer and arXiv venues.

## Screening Results

- **Initial hits:** 615 unique records across the 16 queries, after removing duplicate titles. The six follow-ups added 4 more candidates.
- **After title/abstract screening:** 51 candidates. Most exclusions were off-topic for "Khmer": genomics (the `khmer` software package), history, linguistics and speech.
- **Full-text assessment:** 44 DOIs resolved and their metadata retrieved through OpenAlex; 7 non-DOI sources were checked on Hugging Face and GitHub.
- **Final included:** 42 (35 DOI-bearing works and 7 dataset, model or benchmark cards).
- **Read scope:** abstracts only, for every source. Springer full texts sit behind a sign-in wall. Six Springer chapters have no abstract in any open index, so their entries rest on the title and metadata, plus a search summary for KhmerST and the "Towards Universal Khmer Text Recognition" preprint. These are marked `read: title+metadata` below, and the related-work text claims nothing about them beyond their stated subject.
- **Excluded at full assessment:**

  | Source | Reason |
  |---|---|
  | Annanurov & Noor 2021, compact CNN for Khmer handwriting | Recognizes isolated characters, not lines |
  | Hean et al. 2025, broken Khmer characters | No abstract obtainable; covered by [1] and [10] |
  | Born et al. 2022, Khmer palm-leaf language model | Post-correction only |
  | Keo et al. 2026, Khmer visual text generation | Image generation, not recognition |
  | Rang et al. 2024, scaling laws | Redundant with [25] |
  | Coquenet et al. 2023, DAN | Recognizes whole pages, outside line recognition |
  | Souibgui et al. 2022, few-shot HTR | Recognizes symbols in cipher alphabets |
  | Cheema et al. 2024, Urdu VLM OCR | Overlaps with [41] |
  | Gholami et al. 2021, quantization survey | [42] is enough |

## Annotated Bibliography

Numbering follows the IEEE reference list in `related_work.md`. Every entry has `Method weaknesses: not assessed (read scope: abstract_only)` or `(read scope: title+metadata)` per rule 5 of the protocol. No method-level weaknesses are inferred without the full text. "Stance" is the source's relation to KhmerClipper's approach: supports, opposes or neutral.

### Khmer recognition

**[1] Buoy, Iwamura, Srun, Kise (2023). Toward a low-resource non-Latin-complete baseline: An exploration of Khmer OCR.** IEEE Access 11.
- **Type:** journal article.
- **Method:** a convolutional Transformer over local 2-D feature maps. Arbitrarily long lines are handled by chunking and merging the image, not by resizing it.
- **Key findings:** frames Khmer as "non-Latin-complete": stacked characters, diacritics, ligatures, uneven character widths and no word boundaries. Proposes a low-cost recognizer for such scripts.
- **Relevance:** the closest prior academic baseline for Khmer line recognition.
- **Quality:** high (peer-reviewed journal).
- **Stance:** neutral (a different architecture solving the same problem).
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Use:** Related work A and B.

**[2] Valy, Verleysen, Chhun, Burie (2017). SleukRith Set.** HIP 2017.
- **Type:** dataset paper.
- **Contents:** 657 pages of Khmer palm-leaf manuscripts, annotated as isolated characters, words and lines.
- **Relevance:** the first public Khmer manuscript dataset.
- **Quality:** high. Stance: neutral.
- **Method weaknesses:** not assessed (abstract_only).
- **Use:** A.

**[3] Valy et al. (2018). Character and text recognition of Khmer historical palm-leaf manuscripts.** ICFHR 2018.
- CNN, LSTM and hybrid character classifiers.
- 1-D and 2-D RNN word recognition with glyph localization, motivated by Khmer's 2-D layout.
- Stance: neutral. Method weaknesses: not assessed (abstract_only). Use: A.

**[4] Valy, Verleysen, Chhun (2019). Glyph class map generation with encoder-decoder.** ICPRAM 2019.
- A two-stage model: predict a map of glyph classes, then decode it with attention.
- Evaluated on SleukRith.
- Stance: neutral. Method weaknesses: not assessed (abstract_only). Use: A.

**[5] Valy, Verleysen, Chhun (2020). Data augmentation and text recognition on Khmer historical manuscripts.** ICFHR 2020.
- Builds new word images by combining annotated glyphs into sub-syllables.
- Recognizer: Inception blocks, a BiLSTM and an attention decoder.
- An early synthetic-data strategy that respects Khmer structure.
- Stance: supports (structure-aware synthesis). Method weaknesses: not assessed (abstract_only). Use: A and C.

**[6] Nguyen, Tang, Vo, Huynh (2026). OCR of ancient Khmer palm-leaf manuscripts using TrOCR and segment-level data augmentation.** CCIS.
- Read: title+metadata.
- By its title, applies TrOCR to Khmer manuscripts with augmentation, the closest architectural match to M3.
- Stance: supports. Method weaknesses: not assessed (read scope: title+metadata). Use: A and B.

**[7] Buoy, Kor, Taing (2021). End-to-end Khmer OCR using Seq2Seq with attention.** arXiv preprint.
- A residual CNN with a GRU encoder and an attention decoder.
- Trained on rendered text lines in seven common fonts.
- Stance: supports (synthetic printed lines). Method weaknesses: not assessed (abstract_only). Use: A and C.

**[8] Buoy, Taing, Chenda, Kor (2022). Khmer printed character recognition using attention-based Seq2Seq.** HCMCOU J. Sci. 12(1).
- The journal version of [7], with complex data augmentation.
- Stance: supports. Method weaknesses: not assessed (abstract_only). Use: A.

**[9] Nom, Bakkali, Luqman, Coustaty, Ogier (2025). Evaluating the impact of Khmer font types on text recognition.** arXiv.
- Tests Tesseract on 19 Khmer fonts; Bayon, Dangrek and iSeth First do poorly.
- Evidence that font choice shapes accuracy, which supports rendering many fonts.
- Stance: supports. Method weaknesses: not assessed (abstract_only). Use: C.

**[10] Buoy, Chenda, Taing, Kong, Iwamura, Kise (2025). Addressing the attention drift problem for Khmer long textline recognition.** IJDAR 29(2).
- A partially autoregressive decoder keeps attention aligned on long lines; Semantic Scholar TLDR only.
- Relevant to our end-of-text bias and line splitting.
- Stance: neutral (an alternative fix). Method weaknesses: not assessed (read scope: TLDR). Use: B.

**[11] Buoy, Iwamura, Srun, Kise (2025). Language-aware non-autoregressive Khmer textline recognition.** LNCS.
- Read: title+metadata.
- Non-autoregressive decoding with language awareness: a speed and accuracy alternative to autoregressive decoders.
- Stance: neutral. Method weaknesses: not assessed (title+metadata). Use: B.

**[12] Kong, Buoy, Chenda, Taing, Iwamura, Kise (2026). Towards a joint Khmer text recognition and word segmentation.** arXiv.
- A CTC model that can output Khmer text with or without word boundaries.
- Motivated by RAG pipelines.
- Stance: neutral. Method weaknesses: not assessed (abstract_only). Use: B.

**[13] Kong, Buoy, Chenda, Taing, Iwamura, Kise (2026). Towards universal Khmer text recognition.** LNCS; arXiv:2603.00702.
- Read: title+metadata plus a search summary.
- One model for printed, handwritten and scene text, using modality-aware adaptive feature selection.
- Releases a "universal" Khmer benchmark; the release is gated.
- Stance: neutral (the most direct competitor in scope). Method weaknesses: not assessed (title+metadata). Use: B and D.

### Text recognition architectures

**[14] Shi, Bai, Yao (2017). CRNN.** IEEE TPAMI 39(11).
- CNN, RNN and CTC trained end to end; no segmentation, arbitrary line length.
- Foundational.
- Stance: neutral. Method weaknesses: not assessed (abstract_only). Use: B.

**[15] Baek et al. (2019). What is wrong with scene text recognition model comparisons?** ICCV 2019.
- Shows that inconsistent training and evaluation data confound comparisons.
- Proposes a four-stage STR framework.
- The methodological basis for our KHOB re-registration argument.
- Stance: supports. Method weaknesses: not assessed (abstract_only). Use: B and D.

**[16] Li et al. (2023). TrOCR.** AAAI 37(11).
- A pre-trained image Transformer encoder with a text Transformer decoder.
- Pre-trained on synthetic data, then fine-tuned.
- The base architecture of M3.
- Stance: supports. Method weaknesses: not assessed (abstract_only). Use: B.

**[17] Fujitake (2024). DTrOCR.** WACV 2024.
- A decoder-only Transformer OCR that reuses a pre-trained generative language model.
- Reports gains on printed, handwritten and scene text in English and Chinese.
- Stance: neutral (successor design). Method weaknesses: not assessed (abstract_only). Use: B.

**[18] Bautista, Atienza (2022). PARSeq.** ECCV 2022, LNCS.
- Permutation language modelling: one model decodes both autoregressively and non-autoregressively, with iterative refinement.
- Stance: neutral. Method weaknesses: not assessed (abstract_only, via arXiv). Use: B.

**[19] Kang, Riba, Rusiñol, Fornés, Villegas (2022). Pay attention to what you read: Non-recurrent handwritten text-line recognition.** Pattern Recognition 129.
- A Transformer without recurrence for handwritten lines.
- Stance: neutral. Method weaknesses: not assessed (abstract_only). Use: B.

**[20] Li, Chen, Tang, Shen (2024). HTR-VT.** Pattern Recognition 158.
- A data-efficient ViT encoder with CNN features, SAM optimization and span masking.
- Shows Transformers can work on small handwriting sets.
- Stance: neutral. Method weaknesses: not assessed (abstract_only). Use: B.

**[21] Garrido-Munoz, Ríos-Vila, Calvo-Zaragoza (2025). Handwritten text recognition: A survey.** IEEE TPAMI 48(4).
- A survey from line level to document level.
- Stance: neutral. Method weaknesses: not assessed (abstract_only). Use: B.

### Training data

**[22] Jaderberg, Simonyan, Vedaldi, Zisserman (2014). Synthetic data and artificial neural networks for natural scene text recognition.** arXiv.
- Shows synthetic text can replace real labels for training.
- Foundational, outside the window.
- Stance: supports. Method weaknesses: not assessed (abstract_only). Use: C.

**[23] Yim, Kim, Cho, Park (2021). SynthTIGER.** ICDAR 2021, LNCS.
- A synthetic text-image generator that rebalances the long tails of text length and character distributions.
- Stance: supports. Method weaknesses: not assessed (abstract_only, via arXiv). Use: C.

**[24] Baek, Matsui, Aizawa (2021). What if we only use real datasets for scene text recognition?** CVPR 2021.
- Shows STR can be trained on fewer real labels.
- Notes that synthetic data is hard to come by for handwriting and for non-English scripts.
- Stance: supports (mixing real data). Method weaknesses: not assessed (abstract_only). Use: C.

**[25] Jiang et al. (2023). Revisiting scene text recognition: A data perspective (Union14M).** ICCV 2023.
- The common benchmarks are saturated; 13 models average 66.53% on harder real data.
- Stance: supports (benchmarks hide failures). Method weaknesses: not assessed (abstract_only). Use: C and D.

**[26] Wolf, Fink (2024). Self-training for handwritten word recognition and retrieval.** IJDAR 27(3).
- Synthetic-trained initial models produce pseudo-labels for self-training.
- Closest to our ODC pseudo-labelling, but self-generated rather than labelled by another model.
- Stance: supports. Method weaknesses: not assessed (abstract_only). Use: C.

**[27] Pen, Sin (2026). Akhara: Style-conditioned latent diffusion for Khmer handwriting synthesis.** Zenodo (grey literature).
- Diffusion-generated, labelled Khmer handwriting.
- Evaluated by FID and by OCR readability.
- Stance: supports. Method weaknesses: not assessed (abstract_only). Use: C.

**[28] SoyVitou (2025). 62k-images-khmer-printed-dataset.** Hugging Face, MIT. Printed Khmer line images. Dataset card. Use: C.

**[29] Sokheng (2026). khmer-synthetic-ocr-v1-100k.** Hugging Face, CC BY 4.0. 100k synthetic Khmer lines. Dataset card. Use: C.

**[30] Chanrith123333 (2026). khmer_english_ocr_image_line.** Hugging Face, CC BY 4.0. Khmer and English line images. Dataset card. Use: C.

**[31] Thareah (2026). khmer-handwriting.** Hugging Face, MIT. Scanned handwriting pages split by page into train, validation and test. Dataset card. Use: C.

### Benchmarks and Khmer systems

**[32] EKYC Solutions (2022). Khmer OCR benchmark dataset (KHOB).** GitHub, MIT, created 2022-07-20.
- A standardized benchmark for Khmer OCR engines.
- The model cards in [36] and [37] report different subset sizes for it.
- Stance: neutral. Use: D.

**[33] Nom, Bakkali, Luqman, Coustaty, Ogier (2024). KhmerST.** ACCV 2024, LNCS.
- The first Khmer scene-text dataset: 1,544 images with line labels and polygon boxes.
- State-of-the-art models degrade on it. Source: the CVF/arXiv summary.
- Stance: neutral. Method weaknesses: not assessed (read scope: search summary). Use: D.

**[34] Nom et al. (2025). WildKhmerST.** LNCS.
- 29,601 annotated text lines from 10,000 real-world images.
- Stance: neutral. Method weaknesses: not assessed (abstract_only). Use: D.

**[35] Nom, Keo, Bakkali, Luqman, Coustaty, Ogier (2026). Cross-lingual learning for low-resource Khmer scene text detection and recognition.** LNCS.
- Read: title+metadata. Transfer from other languages to Khmer scene text.
- Stance: neutral. Method weaknesses: not assessed (title+metadata). Use: D.

**[36] Ing Muyleang (2026). Chakkhu-OCR-Khmer.** Hugging Face model card, Apache-2.0.
- A CNN, a 6-layer Transformer encoder, a CTC head and a cluster decoder; about 18.6M parameters, 39 MB as int8.
- Self-reports 4.50% CER on 4,018 KHOB lines it was not trained on.
- The baseline we compare against. Stance: opposes (competing system). Use: D.

**[37] Netra Lab / Darayut (2026). khmer-text-recognition.** Hugging Face model card, MIT.
- Reports KHOB results on a subset; self-reported.
- Stance: opposes (competing system). Use: D.

**[38] Thuon, Theang (2026). Do MLLMs really understand low-resource Khmer documents? A pilot study on Khmer document VQA.** LNCS.
- Read: title+metadata. Questions whether general multimodal LLMs handle Khmer documents.
- Stance: neutral. Method weaknesses: not assessed (title+metadata). Use: D.

### Neighbouring complex scripts

**[39] Mathew, Jain, Jawahar (2017). Benchmarking scene text recognition in Devanagari, Telugu and Malayalam.** ICDAR 2017.
- A CNN-RNN-CTC recognizer trained on synthetic words rendered from Unicode fonts.
- Introduces the IIIT-ILST benchmark.
- Stance: supports. Method weaknesses: not assessed (abstract_only). Use: E.

**[40] Aung, Thu, Oo (2024). myOCR: Myanmar OCR with post-OCR error correction.** iSAI-NLP 2024.
- CRNN-CTC on 25,790 synthetic images in 14 fonts.
- N-gram, SymSpell, NMT and LLM post-correction.
- Stance: neutral. Method weaknesses: not assessed (abstract_only). Use: E.

**[41] Jayatilleke, de Silva (2025). Zero-shot OCR accuracy of low-resourced languages: Sinhala and Tamil.** RANLP 2025.
- Six engines (Cloud Vision, Document AI, Surya, Tesseract and others), compared zero-shot.
- Printed OCR for unique scripts "remains an open problem".
- Stance: supports (generic engines fall short). Method weaknesses: not assessed (abstract_only). Use: E.

### Deployment

**[42] Jacob et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference.** CVPR 2018.
- Integer-only inference with quantization-aware training, improving the accuracy-latency trade-off on mobile CPUs.
- Background for the int8 ONNX decoder.
- Stance: supports. Method weaknesses: not assessed (abstract_only). Use: F.

## Literature Matrix

Themes: **A** Khmer recognition, **B** architectures, **C** training data, **D** benchmarks and evaluation, **E** neighbouring scripts, **F** deployment. Quality is the evidence role, not a score: High = peer-reviewed venue; Medium = preprint or unindexed venue; Grey = dataset or model card.

| Source | A | B | C | D | E | F | Method | Quality |
|---|---|---|---|---|---|---|---|---|
| [1] Buoy 2023 | main | x | | x | | x | Model | High |
| [2] Valy 2017 | main | | x | x | | | Dataset | High |
| [3] Valy 2018 | main | x | | | | | Model | High |
| [4] Valy 2019 | main | x | | | | | Model | High |
| [5] Valy 2020 | main | | x | | | | Model + augmentation | High |
| [6] Nguyen 2026 | main | x | x | | | | Model | High (title only) |
| [7] Buoy 2021 | main | x | x | | | | Model | Medium |
| [8] Buoy 2022 | main | x | x | | | | Model | Medium |
| [9] Nom 2025a | x | | main | | | | Evaluation | Medium |
| [10] Buoy 2025a | x | main | | | | | Model | High |
| [11] Buoy 2025b | x | main | | | | | Model | High (title only) |
| [12] Kong 2026a | x | main | | | | | Model | Medium |
| [13] Kong 2026b | x | main | | x | | | Model + benchmark | High (summary only) |
| [14] Shi 2017 | | main | | | | | Model | High |
| [15] Baek 2019 | | x | | main | | | Evaluation | High |
| [16] Li 2023 | | main | x | | | | Model | High |
| [17] Fujitake 2024 | | main | | | | | Model | High |
| [18] Bautista 2022 | | main | | | | | Model | High |
| [19] Kang 2022 | | main | | | | | Model | High |
| [20] Li 2024 | | main | x | | | | Model | High |
| [21] Garrido-Munoz 2025 | | main | | | | | Survey | High |
| [22] Jaderberg 2014 | | | main | | | | Model + data | Medium |
| [23] Yim 2021 | | | main | | | | Data tool | High |
| [24] Baek 2021 | | | main | x | | | Evaluation | High |
| [25] Jiang 2023 | | | x | main | | | Dataset + evaluation | High |
| [26] Wolf 2024 | | | main | | | | Self-training | High |
| [27] Pen 2026 | x | | main | | | | Generative model | Grey |
| [28]-[31] HF datasets | x | | main | | | | Datasets | Grey |
| [32] KHOB | x | | | main | | | Benchmark | Grey |
| [33] Nom 2024 | x | | | main | | | Dataset | High |
| [34] Nom 2025b | x | | | main | | | Dataset | High |
| [35] Nom 2026 | x | | x | main | | | Model | High (title only) |
| [36] Chakkhu | x | x | | main | | x | System | Grey |
| [37] Netra Lab | x | | | main | | | System | Grey |
| [38] Thuon 2026 | x | | | main | | | Evaluation | High (title only) |
| [39] Mathew 2017 | | | x | x | main | | Model + benchmark | High |
| [40] Aung 2024 | | | x | | main | | Model | High |
| [41] Jayatilleke 2025 | | | | x | main | | Evaluation | High |
| [42] Jacob 2018 | | | | | | main | Method | High |

## Identified Gaps

Each gap is bounded by this search: the queries above, OpenAlex plus arXiv, 2015-2026, abstracts only.

1. **No shared evaluation protocol for Khmer line OCR.**
   - KHOB is the de facto benchmark [32], but the model cards report different subset sizes [36], [37].
   - None of the screened abstracts describe how KHOB crops are cut from pages, or check them.
   - This mirrors the comparison problem [15] identified for English STR.
   - Nearest prior work: [13] proposes a new benchmark, but it is gated.
2. **Transformer encoder-decoders fine-tuned for modern Khmer documents.**
   - Khmer Transformer work uses custom convolutional Transformers [1], [10], CTC [12], or TrOCR on palm-leaf manuscripts [6].
   - The search found no study that fine-tunes a pre-trained TrOCR-style model on modern printed, screen and handwritten Khmer together. [13] targets the same breadth with a different architecture.
3. **Labels from another model, and data mixing.**
   - Self-training uses the model's own labels [26]. Khmer synthesis uses glyph composition [5], fonts [7], [9] or diffusion [27].
   - No screened abstract trains a Khmer recognizer on another system's confident outputs, or reports replaying a data mix to prevent forgetting.
4. **On-device Khmer OCR.**
   - Apart from Chakkhu's int8 export [36] and low-complexity designs [1], no screened Khmer work reports quantized or mobile inference.
5. **Inaccessible coverage.**
   - Khmer-language venues and local theses weren't searched.
   - The full texts of six 2025-2026 Springer chapters were unavailable, and their abstracts were not in OpenAlex.
   - These are limits of this search, not evidence that the work is absent.

## Recommended Sources by Paper Section

| Section | Key sources |
|---|---|
| Introduction | [1], [16], [32], [36], [41] |
| Related work | All of [1]-[42] (see `related_work.md`) |
| Method (data) | [5], [7], [9], [22]-[24], [26], [28]-[31] |
| Method (model, deployment) | [10], [16], [17], [42] |
| Evaluation | [15], [25], [32], [33], [36], [37] |
| Discussion | [12], [13], [38], [40] |

## Verification Log

- **DOIs:** all 35 DOI-bearing sources resolved through `api.openalex.org/works/doi:…` on 2026-11-12. Titles, authors, years, venues and pages come from those records.
- **arXiv:** [18], [19] and [23] were checked on arxiv.org. [13] was checked via arXiv:2603.00702.
- **Cards:**
  - [28]-[31], [36] and [37] were checked through the Hugging Face Hub API (id, creation date, licence).
  - [32] was checked through the GitHub API (EKYCSolutions/khmer-ocr-benchmark-dataset, MIT, created 2022-07-20).
- **Fabricated or unverified entries:** none. Nothing is cited from memory.
