# SkeeterScore

**An exploratory Python NLP tool for examining sensational language in news-style text.**

SkeeterScore extracts linguistic signals associated with sensational framing and combines six of them into a transparent heuristic score. It uses spaCy for tokenization, lemmas, part-of-speech tags, syntax and named entities; VADER for sentiment; TextBlob for subjectivity; and the NRC Emotion Intensity Lexicon for affect intensity.

> **Interpretation:** This is a handcrafted *language-feature index*, **not** a trained misinformation detector, a factual-accuracy check, or a validated probability that an article is sensational. Higher scores reflect stronger signals under the implemented rules, not necessarily bias or inaccuracy.

## Run locally

Requires Python 3.10+ (recommended). From this repository directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Obtain **NRC-Emotion-Intensity-Lexicon-v1.txt** separately from its official distributor, subject to its license, and place it alongside `main.py`. It is excluded from this repository to avoid redistributing a third-party lexicon without confirming permissions.

```bash
python main.py
python main.py articles/example_neutral.txt
python main.py articles/example_neutral.txt articles/example_emotive.txt --json
```

With no arguments, the tool processes every `.txt` article in `articles/`. The code assumes the first nonempty line is the headline for experimental salience analysis. The two included texts are **synthetic illustrations**, not a benchmark. They are approximate fact-preserving rewrites of one library announcement.

## How the score works

Six features are each capped at 100 before weighting:

| Scored feature | Weight (original code) | Signal |
|---|---:|---|
| Sentiment strength | 20% | Absolute VADER compound sentiment, scaled by the non-neutral share |
| Subjectivity | 20% | TextBlob subjectivity |
| Emotional intensity | 20% | Mean NRC affect intensity per alphabetic token |
| Evaluative modifiers | 10% | Emotion-lexicon adjectives/adverbs in selected syntactic roles per word |
| Intensifiers | 10% | Matches from a specified intensifier vocabulary per word |
| Punctuation | 5% | `!` and `?` count per word |

The original weights total **85%**. This release divides their weighted sum by **0.85** so the index has a theoretical 0–100 range **without altering the relative importance of those six features**. This is a mathematical rescaling, **not a data-fitted calibration or evidence of validity**. The index is often much lower than 100 in practice.

### Additional experimental signals (not included in total)

The program separately computes urgency/immediacy, threat phrase matching, exaggeration markers, causal certainty with affect-rich outcomes, responsibility framing, prediction framing, and named-entity salience. These features are included in `--json` under `experimental_features_not_in_total`; they **do not** affect SkeeterScore. Some are early exploratory heuristics and may yield weak or incomplete detections.

## Preliminary validation: same story, different framing

To check that the index responds to writing style rather than simply article topic, I wrote two short versions of the same fictional library announcement: one relatively neutral and one using heightened emphasis, evaluative language, and exclamation marks. Both describe a council approving a library, construction beginning next spring, and planned reading/computer facilities.

| Example | Words | SkeeterScore | Subjectivity | Emotional intensity | Punctuation |
|---|---:|---:|---:|---:|---:|
| `example_neutral.txt` | 27 | **11.06** | 24.39 | 9.00 | 0.00 |
| `example_emotive.txt` | 27 | **19.71** | 49.17 | 11.57 | 11.11 |

**Observed difference: +8.65 points** for the more emphatic rewrite. These values are from one local run reported during development; they are not independently reproduced in this packaged repository because the licensed NRC lexicon is not included. Exact reproduction may depend on dependency and lexicon versions.

This is a **sanity check, not model validation**. It uses one hand-authored pair, includes some lexical differences, and does not measure agreement with human assessments. It is too small to establish accuracy, generalization, a meaningful cutoff, or performance on published journalism. A separate early extreme-style pair scored 10.24 versus 44.32, but changed the underlying factual claims; it was therefore not used as the controlled demonstration.

## Project structure

```text
SkeeterScore/
├── main.py
├── articles/
│   ├── example_neutral.txt
│   └── example_emotive.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## Why I built this

Sensationalism can arise from how events are presented, not only what happened. I wanted a measurable way to inspect this difference instead of relying on an unexplained black-box label. The project grew from simple sentiment and subjectivity baselines into feature engineering around affective language, grammatical modifiers, and exploratory framing signals. The most important design decision was keeping the composite score interpretable, while separating unvalidated experimental signals from the reported number.

## Limitations and next steps

- The rules and weights are **hand-selected**; no human-rated test set has yet been supplied to establish accuracy, inter-rater agreement, or thresholds.
- Strong emotions can appear in accurate, responsible reporting. The score does not establish intent, falsehood, ideological bias or article quality.
- Punctuation, lexicons, VADER and TextBlob may be sensitive to genre, context, negation and figurative language.
- The first nonempty line is assumed to be the headline. Real article page noise and entity recognition errors can distort salience.
- Experimental framing features rely on approximate syntax/lexical heuristics and need targeted evaluation before integration.
- Future work: annotated news samples, error analysis, ablation studies, calibrated weights, contextual phrase matching, and automated tests.

## Attribution and licenses

This repository uses third-party libraries installed through `requirements.txt` and requires the separately obtained **NRC Emotion Intensity Lexicon**. Follow the NRC resource's licensing and citation instructions when using or distributing the project. The original source code was supplied by the project author. Add a repository license only after choosing what rights you wish to grant.

## Reproducing the comparison

After installing requirements and obtaining the lexicon, run:

```bash
python main.py articles/example_neutral.txt articles/example_emotive.txt
```

There is no universal cutoff for sensational language in this implementation. Compare scores *relative to deliberately comparable texts* rather than treating individual values as definitive labels.
