# Contributing to ReviewGuard AI

Use Python 3.11 and install the pinned dependencies in a virtual environment as described in README.md. The trained classifier, demo data, and measured reports are included; pretrained model caches download on first use and stay untracked.

## Work together

1. Clone `https://github.com/amaan240821/reviewguard-ai.git` and enter its directory.
2. Before starting work, run `git switch main` followed by `git pull --ff-only`.
3. Create a branch for your change, for example `git switch -c improve-review-evidence`.
4. Make the change and run `make test`. If training or scoring changes, also run `make train` and `make evaluate`, and inspect the generated metrics and figures.
5. Review your changes with `git diff`, then stage the intended files and commit them.
6. Push your branch with `git push -u origin HEAD` and open a pull request on GitHub.
7. Ask the other contributor to review the pull request before merging. This is a collaboration convention; branch protection is not automatically configured.

Keep Streamlit presentation code in `app/` and backend logic in `reviewguard/`. Include meaningful tests for behavioural changes. Never claim fabricated model results or treat a suspicion score as proof of deception. Preserve the corpus attribution and its included licence when redistributing data or derived examples.

Do not commit virtual environments, Hugging Face downloads, local caches, or credentials. Store real API credentials outside tracked files. Public repository contents are visible to everyone; write access is limited to the owner and accepted collaborators.
