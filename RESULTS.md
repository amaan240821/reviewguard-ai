# Measured results

All figures and metrics come from the delivered run.

| Protocol | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Holdout classifier | 0.9125 | 0.8976 | 0.9313 | 0.9141 | 0.9640 | 0.9628 |
| Holdout composite | 0.8781 | 0.8623 | 0.9000 | 0.8807 | 0.9418 | 0.9330 |
| Hotel-held-out classifier | 0.8746 | 0.8659 | 0.8875 | 0.8765 | 0.9434 | 0.9469 |

Synthetic group recovery: 23/23 within-group pairs. 54 automated tests passed. Hotel-held-out folds use the training pool only.

See README.md for setup and docs/REPORT_ASSETS.md for every figure path and caption.
