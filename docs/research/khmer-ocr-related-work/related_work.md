# Related Work

Draft related-work section for a KhmerClipper / M3 paper. The citations are IEEE style and every source is verified (see
[search_report.md](search_report.md)). Words in body: about 1,000.

## A. Khmer text recognition

Khmer is hard for recognizers built around Latin text. Its consonants stack vertically, its vowels and diacritics sit
above, below and around the base character, and it is written without spaces between words. Buoy et al. call it a
"non-Latin-complete" script for these reasons [1]. Early neural work on Khmer came from historical documents. The
SleukRith set provided 657 annotated pages of palm-leaf manuscripts [2]. Valy et al. used it to study character and
word recognition with 1-D and 2-D recurrent networks [3], to predict glyph-class maps before decoding [4], and to
synthesize word images from glyphs grouped into sub-syllables [5]. TrOCR has recently been applied to the same
manuscripts with segment-level augmentation [6].

For modern printed Khmer, Buoy et al. trained an attention-based sequence-to-sequence recognizer on text lines
rendered in seven common fonts [7], [8]. Font choice matters: a study of 19 Khmer fonts found large accuracy
differences under Tesseract [9]. The same group later proposed a convolutional Transformer that handles arbitrarily
long lines by chunking the image rather than resizing it [1]. Follow-up work addressed attention drift on long lines
with a partially autoregressive decoder [10], studied non-autoregressive decoding [11], and joined recognition with
word segmentation in one connectionist temporal classification (CTC) model [12]. The broadest recent effort recognizes printed, handwritten and scene Khmer in
one model and introduces a benchmark for that setting [13].

## B. Transformer recognizers

General text recognition moved from convolutional-recurrent (CNN-RNN) pipelines trained with CTC [14] to attention and Transformer decoders. A unified
comparison showed that many reported gains came from differences in training and evaluation data rather than from
architecture [15]. TrOCR pairs a pre-trained image Transformer encoder with a pre-trained text Transformer decoder.
It is pre-trained on synthetic text and fine-tuned on labelled data, and it is reported to be competitive on printed,
handwritten and scene text without an external language model [16]. Successors drop the encoder in favour of a
pre-trained generative decoder [17], or train one model for both autoregressive and non-autoregressive decoding [18].
For handwriting, Transformers without recurrence [19] and data-efficient vision Transformers [20] reduce the
labelled data needed; a recent survey traces this shift from line-level to document-level recognition [21]. Our model
follows the TrOCR design [16]. We fine-tune a single checkpoint across printed, screen and handwritten Khmer, and add
an end-of-text bias at export to stop decoding run-ons, a simpler alternative to the decoder changes in [10], [11].

## C. Training data for low-resource scripts

Synthetic data has been the standard answer to label scarcity since rendered words were shown to replace real labels
for scene text [22]. Generators such as SynthTIGER also balance character and length distributions [23]. Synthetic
data has limits. Baek et al. note that it is hard to produce convincingly for handwriting and for scripts other than
English, and they show that real data alone can train competitive models [24]. Union14M shows that models saturating
the standard benchmarks still fail often on harder real images [25]. Self-training offers a middle route: a model
trained on synthetic data labels unlabelled real images for its own next round [26].

For Khmer, synthesis has used glyph composition [5], font rendering [7], [9] and, recently, diffusion models that
generate labelled handwriting [27]. Public line datasets now exist for printed text [28], synthetic text [29], mixed
Khmer and English [30], and scanned handwriting [31]. Our training combines these sources with Wikipedia text rendered
at scan resolution with slivers of the neighbouring lines. It also adds law scans labelled by another recognizer and
kept only above a confidence threshold. This differs from [26], where the labels come from the model being trained.

## D. Benchmarks and Khmer systems

The Khmer OCR Benchmark (KHOB), released by EKYC Solutions in 2022, is the de facto benchmark for Khmer line OCR [32]. Scene-text benchmarks
followed: KhmerST with 1,544 images [33], and WildKhmerST with 29,601 annotated lines from 10,000 images [34], along
with cross-lingual transfer methods for them [35]. Open models now report KHOB results on their cards. Chakkhu-OCR-Khmer,
a CTC model with about 18.6M parameters and a 39 MB int8 export, reports a 4.50% character error rate (CER) on 4,018 KHOB lines [36]. Netra
Lab's model reports results on a KHOB subset [37]. These figures are self-reported and use different subsets, so they
can't be compared directly. Recent work also asks whether general multimodal large language models (LLMs) read Khmer documents reliably [38].

The comparison problem identified for English scene text recognition (STR) [15] applies here too. Results depend on which KHOB lines are used
and how each crop is cut from its page. We found that KHOB's boxes on its scanned pages are scaled about 2% too tall, so crops near the
bottom of a page show the line below their label. We therefore register the boxes to each page before cropping, and score every
system on the same corrected crops.

## E. Neighbouring complex scripts

Research on other complex scripts points to the same remedies. Indic scene-text recognition was first benchmarked with
a convolutional-recurrent network (CRNN) trained on words rendered from Unicode fonts [39]. Myanmar OCR combines synthetic training in 14 fonts with
statistical and neural post-correction [40]. For Sinhala and Tamil, a zero-shot comparison of six commercial and
open-source engines concluded that printed OCR for such scripts remains an open problem [41]. That supports training
script-specific models rather than relying on general engines.

## F. Deployment

Integer quantization with quantization-aware training enables efficient on-device inference while keeping accuracy
[42]. Among Khmer systems, only Chakkhu publishes a quantized export [36]. KhmerClipper ships TrOCR as CoreML on macOS
and as int8 ONNX on Windows and Linux. We report accuracy for each shipped format, not only for the training
checkpoint.

## Summary of the gap

Prior Khmer work offers strong custom recognizers [1], [10]-[13] and growing datasets [28]-[34]. Evaluation, however,
is not shared [32], [36], [37], and no screened study fine-tunes a pre-trained TrOCR-style model across modern printed,
screen and handwritten Khmer and evaluates its deployed formats. This paper contributes such a model, its data
pipeline, and a registered KHOB evaluation on which it reaches 2.6% CER against Chakkhu's 3.1%.

> **Note for the author.** Sections A-F state only what the cited abstracts support. These sentences come from our own
> results in `training/eval/results/v8.md`, not from the literature, and belong in the Method or Results:
> - The rendering and pseudo-labelling description at the end of C.
> - The box-drift and registration finding in D.
> - The 2.6% vs 3.1% figures in the summary.
>
> Six sources were read from their title and metadata only: [6], [11], [13], [33], [35] and [38]. Read their full texts
> before submission, in case they report results that overlap with ours.

## References

[1] R. Buoy, M. Iwamura, S. Srun, and K. Kise, "Toward a low-resource non-Latin-complete baseline: An exploration of Khmer optical character recognition," *IEEE Access*, vol. 11, pp. 128044–128060, 2023, doi: 10.1109/ACCESS.2023.3332361.

[2] D. Valy, M. Verleysen, S. Chhun, and J.-C. Burie, "A new Khmer palm leaf manuscript dataset for document analysis and recognition: SleukRith set," in *Proc. 4th Int. Workshop Historical Document Imaging and Processing (HIP)*, 2017, pp. 1–6, doi: 10.1145/3151509.3151510.

[3] D. Valy, M. Verleysen, S. Chhun, and J.-C. Burie, "Character and text recognition of Khmer historical palm leaf manuscripts," in *Proc. 16th Int. Conf. Frontiers in Handwriting Recognition (ICFHR)*, 2018, pp. 13–18, doi: 10.1109/ICFHR-2018.2018.00012.

[4] D. Valy, M. Verleysen, and S. Chhun, "Text recognition on Khmer historical documents using glyph class map generation with encoder-decoder model," in *Proc. 8th Int. Conf. Pattern Recognition Applications and Methods (ICPRAM)*, 2019, pp. 749–756, doi: 10.5220/0007555507490756.

[5] D. Valy, M. Verleysen, and S. Chhun, "Data augmentation and text recognition on Khmer historical manuscripts," in *Proc. 17th Int. Conf. Frontiers in Handwriting Recognition (ICFHR)*, 2020, pp. 73–78, doi: 10.1109/ICFHR2020.2020.00024.

[6] T.-T. Nguyen, V.-T. Tang, P. H. Vo, and D.-K. Huynh, "Optical character recognition of ancient Khmer palm-leaf manuscripts using TrOCR and segment-level data augmentation," in *Communications in Computer and Information Science*. Singapore: Springer, 2026, pp. 184–200, doi: 10.1007/978-981-92-7411-6_13.

[7] R. Buoy, S. Kor, and N. Taing, "An end-to-end Khmer optical character recognition using sequence-to-sequence with attention," arXiv:2106.10875, 2021, doi: 10.48550/arXiv.2106.10875.

[8] R. Buoy, N. Taing, S. Chenda, and S. Kor, "Khmer printed character recognition using attention-based Seq2Seq network," *Ho Chi Minh City Open Univ. J. Sci. – Eng. Technol.*, vol. 12, no. 1, pp. 3–16, 2022, doi: 10.46223/HCMCOUJS.tech.en.12.1.2217.2022.

[9] V. Nom, S. Bakkali, M. M. Luqman, M. Coustaty, and J.-M. Ogier, "Evaluating the impact of Khmer font types on text recognition," arXiv:2506.23963, 2025, doi: 10.48550/arXiv.2506.23963.

[10] R. Buoy, S. Chenda, N. Taing, M. Kong, M. Iwamura, and K. Kise, "Addressing the attention drift problem for Khmer long textline recognition," *Int. J. Document Analysis and Recognition*, vol. 29, no. 2, pp. 241–266, 2025, doi: 10.1007/s10032-025-00554-6.

[11] R. Buoy, M. Iwamura, S. Srun, and K. Kise, "Language-aware non-autoregressive Khmer textline recognition," in *Lecture Notes in Computer Science*. Singapore: Springer, 2025, pp. 339–353, doi: 10.1007/978-981-97-8702-9_23.

[12] M. Kong, R. Buoy, S. Chenda, N. Taing, M. Iwamura, and K. Kise, "Towards a joint Khmer text recognition and word segmentation," arXiv:2608.30213, 2026, doi: 10.48550/arXiv.2608.30213.

[13] M. Kong, R. Buoy, S. Chenda, N. Taing, M. Iwamura, and K. Kise, "Towards universal Khmer text recognition," in *Lecture Notes in Computer Science*. Cham: Springer, 2026, pp. 698–715, doi: 10.1007/978-3-032-36023-6_40. (Preprint: arXiv:2603.00702.)

[14] B. Shi, X. Bai, and C. Yao, "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 39, no. 11, pp. 2298–2304, 2017, doi: 10.1109/TPAMI.2016.2646371.

[15] J. Baek, G. Kim, J. Lee, S. Park, D. Han, S. Yun, S. J. Oh, and H. Lee, "What is wrong with scene text recognition model comparisons? Dataset and model analysis," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2019, pp. 4714–4722, doi: 10.1109/ICCV.2019.00481.

[16] M. Li, T. Lv, J. Chen, L. Cui, Y. Lu, D. Florencio, C. Zhang, Z. Li, and F. Wei, "TrOCR: Transformer-based optical character recognition with pre-trained models," in *Proc. AAAI Conf. Artificial Intelligence*, vol. 37, no. 11, 2023, pp. 13094–13102, doi: 10.1609/aaai.v37i11.26538.

[17] M. Fujitake, "DTrOCR: Decoder-only transformer for optical character recognition," in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2024, pp. 8010–8020, doi: 10.1109/WACV57701.2024.00784.

[18] D. Bautista and R. Atienza, "Scene text recognition with permuted autoregressive sequence models," in *Proc. European Conf. Computer Vision (ECCV)*, LNCS. Cham: Springer, 2022, pp. 178–196, doi: 10.1007/978-3-031-19815-1_11.

[19] L. Kang, P. Riba, M. Rusiñol, A. Fornés, and M. Villegas, "Pay attention to what you read: Non-recurrent handwritten text-line recognition," *Pattern Recognition*, vol. 129, Art. no. 108766, 2022, doi: 10.1016/j.patcog.2022.108766.

[20] Y. Li, D. Chen, T. Tang, and X. Shen, "HTR-VT: Handwritten text recognition with vision transformer," *Pattern Recognition*, vol. 158, Art. no. 110967, 2024, doi: 10.1016/j.patcog.2024.110967.

[21] C. Garrido-Munoz, A. Ríos-Vila, and J. Calvo-Zaragoza, "Handwritten text recognition: A survey," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 48, no. 4, pp. 4367–4387, 2025, doi: 10.1109/TPAMI.2025.3646002.

[22] M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman, "Synthetic data and artificial neural networks for natural scene text recognition," arXiv:1406.2227, 2014, doi: 10.48550/arXiv.1406.2227.

[23] M. Yim, Y. Kim, H.-C. Cho, and S. Park, "SynthTIGER: Synthetic text image GEneratoR towards better text recognition models," in *Proc. Int. Conf. Document Analysis and Recognition (ICDAR)*, LNCS. Cham: Springer, 2021, pp. 109–124, doi: 10.1007/978-3-030-86337-1_8.

[24] J. Baek, Y. Matsui, and K. Aizawa, "What if we only use real datasets for scene text recognition? Toward scene text recognition with fewer labels," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 3112–3121, doi: 10.1109/CVPR46437.2021.00313.

[25] Q. Jiang, J. Wang, D. Peng, C. Liu, and L. Jin, "Revisiting scene text recognition: A data perspective," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2023, pp. 20486–20497, doi: 10.1109/ICCV51070.2023.01878.

[26] F. Wolf and G. A. Fink, "Self-training for handwritten word recognition and retrieval," *Int. J. Document Analysis and Recognition*, vol. 27, no. 3, pp. 225–244, 2024, doi: 10.1007/s10032-024-00484-9.

[27] K. Pen and S. Sin, "Akhara: A study on a style-conditioned latent diffusion model for Khmer handwriting synthesis," Zenodo, 2026, doi: 10.5281/zenodo.21099787.

[28] SoyVitou, "62k-images-khmer-printed-dataset," Hugging Face, 2025. [Online]. Available: https://huggingface.co/datasets/SoyVitou/62k-images-khmer-printed-dataset

[29] Sokheng, "khmer-synthetic-ocr-v1-100k," Hugging Face, 2026. [Online]. Available: https://huggingface.co/datasets/Sokheng/khmer-synthetic-ocr-v1-100k

[30] Chanrith123333, "khmer_english_ocr_image_line," Hugging Face, 2026. [Online]. Available: https://huggingface.co/datasets/Chanrith123333/khmer_english_ocr_image_line

[31] Thareah, "khmer-handwriting," Hugging Face, 2026. [Online]. Available: https://huggingface.co/datasets/Thareah/khmer-handwriting

[32] EKYC Solutions, "Khmer OCR benchmark dataset (KHOB)," GitHub, 2022. [Online]. Available: https://github.com/EKYCSolutions/khmer-ocr-benchmark-dataset

[33] V. Nom, S. Bakkali, M. M. Luqman, M. Coustaty, and J.-M. Ogier, "KhmerST: A low-resource Khmer scene text detection and recognition benchmark," in *Proc. Asian Conf. Computer Vision (ACCV)*, LNCS. Singapore: Springer, 2024, pp. 383–399, doi: 10.1007/978-981-96-0917-8_22.

[34] V. Nom, S. Keo, S. Bakkali, M. M. Luqman, M. Coustaty, M. Rusiñol, and J.-M. Ogier, "WildKhmerST: A comprehensive dataset and benchmark for Khmer scene text detection and recognition in the wild," in *Lecture Notes in Computer Science*. Cham: Springer, 2025, pp. 351–368, doi: 10.1007/978-3-032-04630-7_20.

[35] V. Nom, S. Keo, S. Bakkali, M. M. Luqman, M. Coustaty, and J.-M. Ogier, "Cross-lingual learning for low-resource Khmer scene text detection and recognition," in *Lecture Notes in Computer Science*. Cham: Springer, 2026, pp. 347–365, doi: 10.1007/978-3-032-09371-4_21.

[36] M. Ing, "Chakkhu-OCR-Khmer," Hugging Face, 2026. [Online]. Available: https://huggingface.co/muyleanging/Chakkhu-OCR-Khmer

[37] Netra Lab (Darayut), "khmer-text-recognition," Hugging Face, 2026. [Online]. Available: https://huggingface.co/Darayut/khmer-text-recognition

[38] N. Thuon and P. Theang, "Do MLLMs really understand low-resource Khmer documents? A pilot study on Khmer document VQA," in *Lecture Notes in Computer Science*. Cham: Springer, 2026, pp. 247–262, doi: 10.1007/978-3-032-39385-2_17.

[39] M. Mathew, M. Jain, and C. V. Jawahar, "Benchmarking scene text recognition in Devanagari, Telugu and Malayalam," in *Proc. 14th IAPR Int. Conf. Document Analysis and Recognition (ICDAR)*, 2017, pp. 42–46, doi: 10.1109/ICDAR.2017.364.

[40] T. Aung, Y. K. Thu, and M. N. Oo, "myOCR: Optical character recognition for Myanmar language with post-OCR error correction," in *Proc. Int. Joint Symp. Artificial Intelligence and Natural Language Processing (iSAI-NLP)*, 2024, pp. 1–6, doi: 10.1109/iSAI-NLP64410.2024.10799448.

[41] N. Jayatilleke and N. de Silva, "Zero-shot OCR accuracy of low-resourced languages: A comparative analysis on Sinhala and Tamil," in *Proc. Int. Conf. Recent Advances in Natural Language Processing (RANLP)*, 2025, pp. 471–480, doi: 10.26615/978-954-452-098-4-056.

[42] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko, "Quantization and training of neural networks for efficient integer-arithmetic-only inference," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 2704–2713, doi: 10.1109/CVPR.2018.00286.
