"""CRISP-DM: reproducible feature selection, untouched holdout, static export."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE, RFECV, mutual_info_regression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dashboard/data'
URL = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/refs/heads/master/datasets/50_Startups.csv'
NUM = ['R&D Spend', 'Administration', 'Marketing Spend']
STATES = ['California', 'Florida', 'New York']
NAMES = NUM + ['State_Florida', 'State_New York']
METHODS = ['pearson', 'mi', 'rfe', 'rfecv', 'lasso']


def encode(train, other):
    encoder = OneHotEncoder(categories=[STATES], drop='first', handle_unknown='ignore', sparse_output=False)
    a = np.column_stack([train[NUM].values, encoder.fit_transform(train[['State']])])
    b = np.column_stack([other[NUM].values, encoder.transform(other[['State']])])
    return a, b


def select(x, y, method, k):
    info = {}
    if method in ('pearson', 'mi'):
        scores = np.array([abs(pearsonr(x[:, j], y).statistic) if np.ptp(x[:, j]) else 0 for j in range(5)]) if method == 'pearson' else mutual_info_regression(x, y, discrete_features=[False]*3+[True]*2, random_state=42)
        scores = np.nan_to_num(scores)
        chosen = np.sort(np.argsort(-scores, kind='stable')[:k])
        info['scores'] = scores.tolist()
    elif method in ('rfe', 'rfecv'):
        estimator = make_pipeline(StandardScaler(), LinearRegression())
        kwargs = dict(estimator=estimator, step=1, importance_getter='named_steps.linearregression.coef_')
        selector = RFE(n_features_to_select=k, **kwargs) if method == 'rfe' else RFECV(min_features_to_select=1, cv=KFold(5, shuffle=True, random_state=42), scoring='neg_root_mean_squared_error', **kwargs)
        selector.fit(x, y)
        chosen = np.flatnonzero(selector.support_)
        info['ranking'] = selector.ranking_.tolist()
        if method == 'rfecv':
            info['internal_curve'] = (-selector.cv_results_['mean_test_score']).tolist()
    else:
        z = StandardScaler().fit_transform(x)
        selector = LassoCV(cv=KFold(5, shuffle=True, random_state=42), random_state=42, max_iter=100000, tol=1e-7).fit(z, y)
        chosen = np.flatnonzero(selector.coef_ != 0)
        info.update(alpha=float(selector.alpha_), scores=selector.coef_.tolist())
    return chosen, info


def fit_predict(x, y, other, chosen):
    if len(chosen) == 0:
        return np.full(len(other), y.mean()), float(y.mean()), []
    model = LinearRegression().fit(x[:, chosen], y)
    return model.predict(other[:, chosen]), float(model.intercept_), model.coef_.tolist()


def metrics(y, pred):
    return dict(rmse=float(np.sqrt(mean_squared_error(y, pred))), mae=float(mean_absolute_error(y, pred)), r2=float(r2_score(y, pred)))


def main(download=False):
    DATA.mkdir(exist_ok=True, parents=True)
    path = DATA / '50_Startups.csv'
    if download or not path.exists():
        path.write_bytes(urlopen(URL, timeout=30).read())
    df = pd.read_csv(path)
    assert list(df.columns) == NUM + ['State', 'Profit']
    assert len(df) == 50 and not df.isna().any().any()
    assert sorted(df.State.unique()) == STATES
    assert np.isfinite(df[NUM + ['Profit']].values).all()
    assert (df[NUM] >= 0).all().all()
    train, test = train_test_split(df, test_size=.2, random_state=42)
    x, xt = encode(train, test)
    y, yt = train.Profit.to_numpy(), test.Profit.to_numpy()
    folds = list(KFold(5, shuffle=True, random_state=42).split(train))
    models = []
    for method in METHODS:
        for k in ([1, 2, 3, 4, 5] if method in METHODS[:3] else [None]):
            cv, fold_features = [], []
            for a, b in folds:
                xa, xb = encode(train.iloc[a], train.iloc[b])
                chosen, _ = select(xa, y[a], method, k)
                pred, _, _ = fit_predict(xa, y[a], xb, chosen)
                cv.append(metrics(y[b], pred))
                fold_features.append([NAMES[i] for i in chosen])
            chosen, info = select(x, y, method, k)
            pred, intercept, coef = fit_predict(x, y, xt, chosen)
            fitted, _, _ = fit_predict(x, y, x, chosen)
            train_metrics = metrics(y, fitted)
            train_metrics['adjusted_r2'] = 1-(1-train_metrics['r2'])*(len(y)-1)/(len(y)-len(chosen)-1)
            entry = dict(id=f'{method}-{k or "auto"}', method=method, requested_k=k, k=len(chosen), features=[NAMES[i] for i in chosen], indices=chosen.tolist(), intercept=intercept, coefficients=coef, selection=info, test=metrics(yt, pred), train=train_metrics,
                         cv={key:float(np.mean([v[key] for v in cv])) for key in cv[0]}, cv_rmse_sd=float(np.std([v['rmse'] for v in cv])), folds=cv, fold_features=fold_features,
                         residuals=[dict(row=int(idx)+1, actual=float(actual), predicted=float(p), residual=float(actual-p)) for idx, actual, p in zip(test.index, yt, pred)])
            models.append(entry)
            print(entry['id'], entry['features'], round(entry['cv']['rmse'], 2), flush=True)
    best = min(models, key=lambda m:(m['cv']['rmse'], m['k'], m['id']))
    out = dict(source=URL, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), sklearn_version=sklearn.__version__, seed=42, train_rows=(train.index+1).tolist(), test_rows=(test.index+1).tolist(), original_features=NUM+['State'], encoded_features=NAMES, categories=STATES, baseline='California', best_id=best['id'], models=models, records=df.to_dict(orient='records'), ranges={c:[float(train[c].min()), float(train[c].max())] for c in NUM})
    (DATA/'models.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Refresh the bundled source CSV')
    main(parser.parse_args().download)
