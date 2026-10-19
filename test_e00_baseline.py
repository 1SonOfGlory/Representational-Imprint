"""Local checks requiring NumPy only; model-dependent checks run inside the notebook."""
import unittest
from e00_baseline import make_items, summarize, TEMPLATES


class BaselineTests(unittest.TestCase):
    def test_dataset_identity(self):
        items = make_items()
        self.assertEqual(len(items), 164)
        self.assertEqual(len({r['item_id'] for r in items}), len(items))
        self.assertTrue(all(r['split'] == 'development' for r in items))
        self.assertEqual(sum(r['kind'] == 'control' for r in items), len(TEMPLATES))

    def test_signed_cancellation_does_not_hide_absolute_effect(self):
        items = make_items()
        for row in items:
            row['score'] = 2.0 if row['template_id'] in ('t0', 't1') else -2.0
        result = summarize(items)
        self.assertEqual(result['mean_signed_log_ratio'], 0)
        self.assertEqual(result['mean_absolute_item_log_ratio'], 2)
        self.assertEqual(result['signed_cluster_bootstrap_95_percent_interval'], [0, 0])
        self.assertEqual(result['absolute_cluster_bootstrap_95_percent_interval'], [2, 2])

    def test_controls_excluded_and_bootstrap_repeatable(self):
        items = make_items()
        for n, row in enumerate(items):
            row['score'] = 10000 if row['kind'] == 'control' else n / 100
        result = summarize(items)
        self.assertAlmostEqual(result['mean_signed_log_ratio'], .795)
        self.assertEqual(result, summarize(items))


if __name__ == '__main__':
    unittest.main()
