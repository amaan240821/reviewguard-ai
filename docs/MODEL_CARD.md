# ReviewGuard AI model card

## Intended use

Research and educational screening of English hotel reviews. Scores support human analysis and must not determine punitive actions without other evidence. Neither a corpus deception probability nor the composite is a real-world authenticity guarantee.

## Data and training

Ott Deceptive Opinion Spam Corpus v1.4: 1,600 original reviews from 20 Chicago hotels; 1,596 unique texts after deduplication. Main split: 1,276 training / 320 test, stratified by label × polarity, seed 42. Positive class = deceptive. Corpus has no ratings. Synthetic demonstration rows are excluded from training and classification evaluation.

Four models compared on training-only 5-fold CV: TF-IDF LR, TF-IDF linear SVM, MiniLM LR, and stacked text/embedding/linguistic LR. Selected model: **TF-IDF + Linear SVM**. Fold-local vectorizers prevent vocabulary leakage. Calibration uses sigmoid calibration inside training folds. The binary threshold maximizes training OOF F1 and is frozen before holdout evaluation.

## Holdout classifier results

| Metric | Value |
|---|---:|
| accuracy | 0.9125 |
| precision | 0.8976 |
| recall | 0.9313 |
| f1 | 0.9141 |
| roc_auc | 0.9640 |
| pr_auc | 0.9628 |

## Hotel generalisation

Five hotel-held-out folds on the training pool, disjoint hotel groups within each outer fold. The holdout test is not reused.

| Metric | Value |
|---|---:|
| accuracy | 0.8746 |
| precision | 0.8659 |
| recall | 0.8875 |
| f1 | 0.8765 |
| roc_auc | 0.9434 |
| pr_auc | 0.9469 |

## Composite scoring

D duplication 30%, C classifier 25%, L language 20%, R rating mismatch 15%, E extreme sentiment 10%. Missing R is omitted and weights are renormalized. Design weights are not empirical constants. The full comparison, learned signed coefficients, and ablations are in reports/metrics.json. The composite is an interpretable screening score, not a calibrated probability.

## Pretrained components

MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; DistilBERT SST-2 revision `714eb0fa89d2f80546fda750413ed43d93601a13`. CPU inference, 256-token sentiment truncation, normalized MiniLM embeddings. VADER and fold-fitted LSA are explicit fallbacks. Model comparisons and the delivered evaluation used MiniLM and DistilBERT successfully.

## Limitations and ethics

The corpus is small and crowdsourced; labels reflect an experimental construction. Hotel-domain performance does not transfer automatically to products, other languages, modern LLM output, sarcasm, or natural coordinated manipulation. Strong sentiment or low specificity can be legitimate. Similarity can reflect common experiences. Ratings are absent, so R has no corpus validation. Injected-group pair recall is a synthetic workflow check, not real-world detection accuracy. The low-end hardware target was not separately benchmarked.

Use evidence review, appeal mechanisms, and domain-specific validation before operational deployment. Avoid labelling people dishonest based on this tool.

## Reproducibility

Python 3.11.17, pinned dependencies, seed 42, corpus SHA256 and exact split/model selection metadata in models/model_card.json; run `make train && make evaluate`. Full raw metrics, prediction tables, figures, and captions are shipped.
