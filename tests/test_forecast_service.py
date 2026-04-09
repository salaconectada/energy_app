import unittest

from backend.services.forecast import run_forecast
from backend.services.preprocessing import load_dataset


class ForecastServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("Data demo/consumo_demo_multivar.csv", "rb") as f:
            cls.df = load_dataset(f.read(), "consumo_demo_multivar.csv")

    def test_sarimax_returns_expected_length(self):
        result = run_forecast(self.df, model="SARIMAX", periods=6, exog_sel=[])
        self.assertEqual(len(result.pred), 6)
        self.assertIn("total_historico_gwh", result.kpis)
        self.assertIsNotNone(result.metrics)

    def test_random_forest_returns_expected_length(self):
        result = run_forecast(self.df, model="Random-Forest", periods=6, exog_sel=[])
        self.assertEqual(len(result.pred), 6)
        self.assertIsNone(result.metrics)


if __name__ == "__main__":
    unittest.main()
