# 50 Startups Multiple Linear Regression Explorer

An interactive CRISP-DM dashboard using actual scikit-learn models. The static application is in `dashboard/`; the existing notebook, analysis script, and figures are preserved.

## Run locally

Python is only needed to serve files, not to run predictions:

```sh
python3 -m http.server 8000 --directory dashboard
```

Open http://localhost:8000. Do not open `index.html` as a `file://` URL: browsers restrict JSON fetching there. Chart.js is bundled locally; no CDN, npm, or Python backend is required.

## Reproduce the models

Python 3.9 or newer:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/train_models.py
.venv/bin/python -m unittest discover -s tests -v
```

The bundled CSV is the exact dataset from [Avik Jain's 100 Days of ML](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/refs/heads/master/datasets/50_Startups.csv). To download it again, run `scripts/train_models.py --download` with the virtual-environment Python. The export records the source URL, CSV SHA-256, library version, split row IDs, fold scores, selected columns, original-unit coefficients, and all test predictions. Training fails on unexpected columns, missing/nonfinite data, invalid spending, wrong state labels, or wrong record count.

## What is implemented

- Five selection methods: Pearson correlation, mutual information, RFE, RFECV, and native LASSO.
- Fifteen fixed-k models (k=1…5 for Pearson, MI, and RFE), plus two automatic models.
- Native RFECV and LASSO use Auto k. They do not pretend to support arbitrary feature counts.
- Algorithm/count changes update metrics, selected columns, equation, coefficients, four charts, prediction, and residuals.
- Compare all five methods at the current fixed k, with automatic counts for RFECV and LASSO.
- Live predictions for all three states, validation of spending inputs, and extrapolation warnings.
- Full dataset view, CSV residual downloads, and model JSON downloads.
- Responsive layout, keyboard-accessible controls, explicit button states, loading errors, and CRISP-DM explanations.

## Statistical protocol

1. Split 50 records into 40 training and 10 test records with `train_test_split(test_size=0.2, random_state=42)`.
2. Encode State with `OneHotEncoder(drop='first', handle_unknown='ignore')`. The known domain categories are declared as California, Florida, and New York; California is omitted. Thus four original predictors yield five encoded columns.
3. Rank Pearson and MI using training data only. MI marks the two indicators as discrete. Undefined/constant Pearson correlations become zero.
4. RFE/RFECV use a `StandardScaler → LinearRegression` estimator and standardized coefficient importance. RFECV fits its scaler inside each internal fold and chooses its count with five-fold CV.
5. LassoCV tunes alpha on standardized predictors, selects nonzero coefficients, and may produce any count (including zero). If zero, an intercept-only final regression is supported. LassoCV's scaler is fitted on each outer training fold before its internal alpha CV, so internal tuning shares scaling statistics; outer validation remains isolated. This limitation is also disclosed in the dashboard.
6. Refit all selected subsets using ordinary `LinearRegression` in original units.
7. Evaluate each configuration using the same five shuffled training folds (`random_state=42`). Repeat encoding, scaling, feature selection, and refitting within each fold. Automatic methods tune inside the outer training fold. Export mean fold RMSE and fold standard deviation. R² is averaged across folds, not computed from pooled predictions.
8. Recommend the lowest mean CV RMSE, tie-breaking by smaller final k and then stable model ID. The current export recommends RFE with R&D Spend, Administration, and Marketing Spend: mean CV RMSE **$9,104.68**. Test metrics never select the winner. Searching configurations can still bias the winning CV estimate downward; use the holdout as a separate check.

The RFECV curve is **internal tuning** RMSE; the metric card reports **outer evaluation** RMSE. Automatic methods appear as single points in cross-method curves, because their k can vary between folds. Training adjusted R² uses n=40 and the selected predictor count; it is not mislabeled as holdout inference. Coefficients are associative, not causal. Post-selection p-values are intentionally omitted. No unsupported claims of an “overfitting cliff” or optimal two-feature solution are hard-coded.

## Browser checks

With the static server running, open http://localhost:8000/qa.html. The test harness exercises all 17 configurations, all three state encodings, prediction parity, comparison, invalid input, extrapolation warnings, dataset display, and mobile overflow. It reports PASS/FAIL at the top. This QA page is optional for deployment.

## Deploy to GitHub Pages

The folder `dashboard/` is the deployable artifact. Upload it using a GitHub Pages Actions artifact (exclude `qa.html` if desired), or copy its contents to the root of a Pages branch. All application paths are relative, so repository subpaths work. Enable Pages for that branch or workflow in the repository settings. The precomputed data and vendored Chart.js must be included. No secrets or server runtime are needed.

A ready-to-use workflow is included at `.github/workflows/pages.yml`. It deploys the bundled dashboard automatically on pushes to `main`, and also supports manual dispatch. It uses the official GitHub Pages Actions and verifies the required assets and model JSON before uploading `dashboard/`; it does not retrain models. Enable **Settings → Pages → Build and deployment → Source → GitHub Actions** in the repository first.

## Design and attribution

The [reference explorer](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/) informed the purple hero, model controls, metrics, paired charts, coefficients, predictor, and residual table. This application implements the five algorithms requested in the supplied prompt rather than reusing the reference site's model results. The pasted prompt ends mid-sentence in Section D and no screenshot was available; subsequent sections follow the behaviors explicitly described earlier in the prompt.

Chart.js 4.4.1 is distributed under the MIT license; see `dashboard/vendor/LICENSE.chartjs.md`.
