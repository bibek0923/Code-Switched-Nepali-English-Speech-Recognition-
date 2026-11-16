# Code-Switched Nepali-English Speech Recognition

Fine-tuning OpenAI Whisper for Nepali speech that mixes English into Nepali sentences, and
using the transcript to produce bilingual subtitles where Nepali words stay in Devanagari and
English words appear in Roman script.

This was our final year major project in Computer Engineering at National College of
Engineering (IOE, Tribhuvan University), submitted March 2026. The full report, including the
system design chapters and all the figures, is in `report/major_project.pdf`.

**Team:** Arudhi Bohora (NCE078BCT006), Bibek Prasad Paneru (NCE078BCT008),
Prabesh Khanal (NCE078BCT022), Siddhartha Thapa (NCE078BCT040)
**Supervisor:** Er. Pradip Adhikaree

## Why code-switching

Most Nepali ASR work assumes the speaker stays in Nepali. In practice almost nobody does.
Lectures, news panels and ordinary conversation drop English nouns and whole phrases inside
Nepali syntax ("यो *assignment* भोलि *submit* गर्नुपर्छ"), and a monolingual model handles that
badly in a specific way: it transcribes the English word phonetically in Devanagari, so
*happy* comes out as "ह्याप्पी". The transcript is readable but it is the wrong script for
that word, which breaks subtitle rendering and anything downstream that does lookup.

So the pipeline has two parts. Whisper produces a Devanagari transcript, then a token-level
language ID pass decides which words were actually English and romanises those back.

## Dataset

14.81 hours in total, from three places:

| Source | Clips | Duration | Notes |
|---|---|---|---|
| Nepali Speech-to-Text (Kaggle) | 330 | 2.31 h | long clips, avg ~25 s, Federal Parliament speeches |
| OpenSLR High-Quality Nepali TTS | 8,000 | 6.67 h | avg ~3 s, 19 female speakers, read news-style sentences |
| Our own recordings | 4,670 | ~5.84 h | avg ~4.5 s, short code-switched phrases, academic and casual |

We recorded the third set ourselves because there was nothing public with Nepali-English
code-switching in it. That set is the reason the project works at all, and also the reason the
numbers below are not comparable to published Nepali WER figures, which are measured on clean
monolingual read speech.

Preprocessing: dropped rows with a missing audio file or empty transcript, stripped punctuation
and danda, normalised whitespace, resampled everything to 16 kHz mono, and filtered out
transcripts longer than 448 label tokens (that removed 7 training and 11 eval samples). 90/10
train/eval split with a fixed seed.

## Model and training

`openai/whisper-medium`, with LoRA adapters on the attention query and value projections only.
Base weights frozen.

```
trainable params: 4,718,592 / 768,576,512  (0.6139%)
LoRA r=16, alpha=32, dropout=0.05, target_modules=["q_proj", "v_proj"]
```

Only `q_proj` and `v_proj` because that is the configuration the original LoRA paper reports as
the best accuracy-per-parameter trade-off, and on a single Colab GPU we did not have the budget
to ablate placement ourselves.

Training used `Seq2SeqTrainer`: learning rate 5e-5, 8 epochs, fp16, AdamW, evaluation and
checkpointing at the end of every epoch, `save_total_limit=2`, and the best checkpoint picked by
lowest eval WER. Audio augmentation was applied online with probability 0.5 — Gaussian noise,
time stretch 0.9-1.1x, pitch shift +/-2 semitones, gain +/-6 dB. Ranges kept deliberately
narrow, since anything more aggressive starts changing what was actually said and the transcript
no longer matches.

One note if you read the report alongside the notebook: Table 3.8 lists train and eval batch
size 8, which was what we used in the earlier step-based runs. The notebook here uses 32, which
is what the final 8-epoch run was done with.

## Results

| Epoch | Train loss | Val loss | WER % | CER % |
|---|---|---|---|---|
| 1 | 0.439962 | 0.403435 | 53.24 | 24.36 |
| 2 | 0.347663 | 0.329335 | 49.22 | 24.02 |
| 3 | 0.308911 | 0.298760 | 43.20 | 22.36 |
| 4 | 0.265398 | 0.278325 | 43.58 | 19.06 |
| 5 | 0.251387 | 0.269441 | 39.99 | 16.22 |
| 6 | 0.225057 | 0.259557 | 37.46 | 15.36 |
| 7 | 0.221369 | 0.255901 | 34.05 | 15.16 |
| 8 | 0.220494 | 0.255022 | 33.26 | 14.24 |

WER 53.24% to 33.26%, CER 24.36% to 14.24%. Both training and validation loss were still
falling at epoch 8, so we stopped on time budget rather than on any sign of overfitting.
WER goes slightly the wrong way at epoch 4 while CER keeps improving, which is what you would
expect when the model is getting characters right inside words it still gets wrong.

The report also quotes a BLEU of 0.5851 for the bilingual output. That came from a hand-checked
sample of a few sentence pairs, so treat it as indicative. The BLEU cell in the notebook runs
sacrebleu over the whole held-out split instead, which is the number worth quoting.

Where the errors actually are: language boundaries inside a clause, loanwords that are ambiguous
between the two languages, and proper nouns. The public half of our training data is also a poor
match for the target — parliamentary oratory and single-gender read speech standing in for
conversational bilingual speech — and that shows up in the eval set.

## What's in here

```
notebooks/whisper_nepali_asr_lora_finetune_final.ipynb   the full training and evaluation run
report/major_project.pdf                                 the submitted report
```

The notebook is the consolidated version of what we actually ran, with the dead branches and
abandoned dataset paths taken out. Outputs are stripped so the diff stays readable.

The web app (upload a video, get subtitles back, export SRT/VTT) is not in this repo yet.

## Running the notebook

It was written for Google Colab and expects the dataset on Drive:

```
/content/drive/MyDrive/merged_dataset/
├── merged_with_manual_remove_dirty.csv    columns: filename, transcript
└── <audio files referenced by filename>
```

Everything you would want to change is in the `config` class in the second cell, so you should
not have to touch anything further down. Then run top to bottom.

Hardware: whisper-medium with batch size 32 needs an A100. On a T4 you will have to drop the
batch size and turn on gradient accumulation, and the eval pass with `predict_with_generate=True`
is slow regardless, because it generates for the entire eval split at every epoch.

## Known gaps

- The language ID and transliteration stage is described in the report but is not in this
  notebook; the notebook covers the ASR model only.
- No speaker-held-out split. Our recordings come from four speakers, so some of the gain on the
  code-switched portion is probably speaker adaptation rather than better code-switching.
- CER and WER are computed on the normalised transcripts (punctuation stripped), so they are not
  directly comparable to papers that keep punctuation.
- The title says real-time. The system processes uploaded files in short chunks with low latency;
  it is not streaming ASR.

## Report

`report/major_project.pdf`. The parts worth jumping to are section 3.5 (architecture and
training), Table 3.7 (LoRA parameter counts), Table 4.1 (the table above), and section 5.2
(what we would do next).
