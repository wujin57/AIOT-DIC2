"""Verify exported inference, held-out evaluation, and selection boundaries."""
import json
import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_models import encode, metrics, select

ROOT = Path(__file__).resolve().parents[1]

class ExportTests(unittest.TestCase):
    problem = "startups"
    filename = "models.json"
    csv = "50_Startups.csv"
    target = "Profit"
    expected_models = 17
    discrete = None
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT/'dashboard/data'/cls.filename).read_text())
        cls.df = pd.read_csv(ROOT/'dashboard/data'/cls.csv)
        cls.train, cls.test = train_test_split(cls.df, test_size=.2, random_state=42)
        cls.x, cls.xt = encode(cls.train, cls.test, cls.problem)

    def test_split_and_recommendation(self):
        self.assertEqual(self.data['train_rows'], (self.train.index+1).tolist())
        self.assertEqual(self.data['test_rows'], (self.test.index+1).tolist())
        self.assertFalse(set(self.data['train_rows']) & set(self.data['test_rows']))
        self.assertEqual(len(self.data['models']), self.expected_models)
        best = min(self.data['models'], key=lambda m:(m['cv']['rmse'], m['k'], m['id']))
        self.assertEqual(best['id'], self.data['best_id'])

    def test_every_export_matches_independent_ols(self):
        for m in self.data['models']:
            with self.subTest(model=m['id']):
                idx = m['indices']
                ref = LinearRegression().fit(self.x[:, idx], self.train[self.target])
                predicted = ref.predict(self.xt[:, idx])
                exported = m['intercept'] + self.xt[:, idx] @ np.array(m['coefficients'])
                np.testing.assert_allclose(exported, predicted, atol=1e-8)
                np.testing.assert_allclose([r['predicted'] for r in m['residuals']], predicted)
                for key, value in metrics(self.test[self.target], predicted).items():
                    self.assertAlmostEqual(value, m['test'][key], places=7)
                self.assertEqual(len(m['folds']), 5)
                self.assertAlmostEqual(m['cv']['rmse'], np.mean([f['rmse'] for f in m['folds']]))
                chosen, _ = select(self.x, self.train[self.target].to_numpy(), m['method'], m['requested_k'], self.discrete)
                self.assertEqual(chosen.tolist(), idx)

    def test_pearson_constants_safe(self):
        x = np.column_stack([np.arange(20), np.ones((20,4))])
        chosen, info = select(x, np.arange(20), 'pearson', 1)
        self.assertEqual(chosen.tolist(), [0])
        self.assertTrue(np.isfinite(info['scores']).all())

class BostonExportTests(ExportTests):
    problem = "boston"
    filename = "boston-models.json"
    csv = "BostonHousing.csv"
    target = "medv"
    expected_models = 41
    discrete = [False, False, False, True, False, False, False, False, True, False, False, False, False]

if __name__ == '__main__':
    unittest.main()
