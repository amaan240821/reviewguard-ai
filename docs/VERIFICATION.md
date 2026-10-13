# Verification record

Final release tested with Python 3.11.17 on macOS ARM64, CPU-only. Dependencies are pinned in requirements.txt.

## Automated checks

`python -m pytest`: **54 passed**, 5 third-party PyMuPDF deprecation warnings; no failed tests. The release run took 65.01 seconds while other local checks were active. Full output: test-results.txt.

Checks cover preprocessing, schema/encoding/empty-input errors, label/polarity-stratified disjoint splits, weight sums and renormalization, all risk band boundaries, linguistic features, safe HTML highlighting, identical/unrelated similarity, graph components, detailed/promotional/contradictory review behaviour, three-review clustering, injected-group recovery, JSON serialization, CSV export, PDF opening/content, real metric consistency, and ten evaluation figures. Missing classifier and embedding fallback paths are tested. Streamlit AppTest covers all seven pages, example analysis, live reweighting, demo loading/search, graph loading, and threshold changes.

## Executed commands

- Data conversion from the author's v1.4 archive; 1,600 raw rows, 1,596 unique texts.
- `python scripts/train.py`: all four candidates compared, selected classifier calibrated, artifact and OOF predictions saved.
- `python scripts/evaluate.py`: holdout metrics, hotel-held-out folds, ablations, learned signal coefficients, synthetic recovery, tables and figures generated.
- `python scripts/cli.py` for standalone JSON review analysis and CSV batch output.
- `python -m streamlit run app/Home.py`: final configured server served successfully at localhost:8501.

## Browser checks

- Overview renders with the custom theme and live counts; navigation to single review and back works.
- One-click three-review demonstration produces 3 reviews, 1 cluster and 2 threshold-crossing pairs with MiniLM / DistilBERT.
- Detailed review with 4 stars returns Low suspicion (13.6/100), counter-evidence and a contribution waterfall.
- Review PDF downloaded from the browser, reopened with PyMuPDF, checked for score/content, and visually inspected. An example is included at docs/example-review-report.pdf.
- A CSV was uploaded through the actual file chooser. Text and rating columns were automatically detected, the preview displayed, and the batch run completed to 100% with the expected cluster.
- The Cluster Explorer opened from batch results and displayed the graph, three-review group, two connections, and review-detail controls.
- Layout was visually checked at the available 939-pixel browser width. Metric labels wrap; the gauge number avoids overflowing a narrow column. Reduced-motion CSS is provided. The final overview screenshot is in docs/screenshots/overview.png.
- Streamlit's file watcher was disabled after it was observed to inspect unrelated lazy transformer vision modules. The final server starts without those watcher import errors.

## Practical limits

The 5,000-row cap is validated but a full 5,000-row performance benchmark was not run. The 8 GB / modest-laptop target and Windows/Linux installations were not separately benchmarked. Offline fallback logic is tested; fallback outputs can differ from pretrained outputs. The dataset has no ratings, so rating contradictions are tested with synthetic examples and are not a corpus-level validation claim. Synthetic recovery is not real-world coordination accuracy. Optional CPU fine-tuning is not implemented.
