# Decisions and tradeoffs

- The delivered architecture follows the supplied Python / Streamlit specification. A separate web hosting rewrite would change the required stack, so the runnable result is a local Streamlit application.
- The prompt's section 5 headings conflict with its scoring table. The authoritative scoring table is followed: D = duplication (30%), C = classifier (25%), L = language (20%), R = rating (15%), E = extreme sentiment (10%).
- Scores are continuous. Bands are <=30, <=60, <=80, >80, avoiding gaps between integer boundaries. Evaluation uses a separate CV-selected binary threshold.
- The 1,600-row author-provided archive has four duplicate texts. They are removed before splitting, leaving 1,596 rows (800 deceptive, 796 truthful). The original CSV and licence are included.
- The downloader tries authenticated Kaggle when configured, then the original author's public archive. Manual Kaggle instructions are printed on failure.
- Main evaluation is an 80/20 label × polarity stratified holdout. Model selection uses five training-only folds; calibration is nested inside training. Additional GroupKFold evaluation holds out hotels within the training pool, so the final test is not repurposed for model selection.
- The corpus has no ratings. Missing ratings are not imputed; R is omitted and available weights are renormalized. Synthetic demonstration ratings are explicitly flagged.
- Pretrained MiniLM and DistilBERT are CPU-only and cache under models/hf_cache. Project-local HF_HOME and HF_XET_CACHE prevent reliance on writable home caches. VADER and TF-IDF/LSA remain explicit fallbacks.
- Model comparison includes TF-IDF logistic regression, linear SVM, embedding logistic regression, and a feature union of text, embedding, and linguistic features. Fallback embeddings, when necessary, fit only inside training folds. Transformer fine-tuning is optional in the prompt and is intentionally not implemented.
- Linguistic features are interpretable heuristics, not learned deception rules. Their weights are documented in linguistic.py. The optional lemma map is conservative; no external language resource downloads occur.
- Extreme sentiment rises above 0.80 absolute polarity and is downweighted past 30 words. It is an indicator, never independent proof.
- Lexical character 5-gram similarity supplements semantic cosine. D ignores evidence below 0.60, saturates near 0.95, and includes a small neighbour-count term. Cluster edges use max(semantic, lexical), and both values remain visible in review evidence.
- The full demo has 180 real rows and 13 synthetic rows across three known groups. Recovery is pair recall for connected components; no natural coordination accuracy is claimed.
- Signals can conflict: a designed composite is not expected to beat a supervised classifier on the corpus's binary label. The app displays both and includes leave-one-signal-out and learned-logistic comparisons.
- Scores re-compute when advanced weights change. Duplicate comparisons use training data plus prior session reviews in single mode; current text is not inserted into the session pool until analysis finishes.
- Model terms show signed TF-IDF × coefficient contributions from one ensemble estimator; they do not pretend to explain the full calibration function causally.
- Data is processed locally. Session caches live in the local process; text-derived disk caches contain embeddings and sentiment, not raw uploaded review files. Do not share the local server with untrusted users.
- Google fonts are cosmetic; system fonts work offline. CSS respects reduced-motion preferences. Browser/Python smoke checks and API tests are recorded in VERIFICATION.md.
- Python 3.11.17 was installed in an isolated workspace environment. Final training, evaluation, and tests run on Python 3.11. Dependencies are pinned to that tested environment; the initial prototype also ran on Python 3.12.

- Streamlit source watching is disabled in the delivered runtime. Inspecting Transformers lazy modules caused optional vision/audio dependency imports unrelated to this NLP application; restarting after code edits avoids that overhead. The server binds to localhost.
