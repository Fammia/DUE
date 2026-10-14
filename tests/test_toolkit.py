"""Small, CPU-only tests; no datasets, training, or experiment logs are needed."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Profile-Pollution-Toolkit" / "scripts"))
import config
from curve_recorder import CurveRecorder

try:
    import torch
    from weight_ops import fuse_with_switches, get_based_weight, production_get_weight
except ImportError:
    torch = None


class ToolkitLayoutTests(unittest.TestCase):
    def test_paths_and_grid(self):
        self.assertEqual(Path(config.PROJECT_ROOT), ROOT)
        self.assertEqual(Path(config.TOOLKIT_ROOT), ROOT / "Profile-Pollution-Toolkit")
        self.assertEqual(len(config.all_stems()), 37)
        self.assertEqual(config.job_count(), 148)
        self.assertTrue(Path(config.targets_path()).is_file())
        for model, stem, setting, _, _ in config.iter_jobs():
            command = config.score_command(model, stem, setting)
            self.assertIn("main_profile_pollution.py", command)
            self.assertNotIn("Lab3", command)
            self.assertNotIn("Lab5", command)
            self.assertNotIn("Lab7", command)

    def test_curve_recording(self):
        with tempfile.TemporaryDirectory(prefix="due-curve-test-") as folder:
            path = Path(folder) / "curves.csv"
            recorder = CurveRecorder(str(path), columns=("epoch", "w_final_clean"))
            recorder.add(1, {"w_final_clean": 0.25})
            recorder.add(2, {"w_final_clean": 0.5})
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[-1]["w_final_clean"]), 0.5)
            self.assertFalse(Path(str(path) + ".tmp").exists())


@unittest.skipIf(torch is None, "Install the training dependencies to test DUE weights")
class WeightTests(unittest.TestCase):
    def test_right_rank_and_ties(self):
        actual = get_based_weight(-torch.tensor([1.0, 3.0, 1.0, 2.0]))
        torch.testing.assert_close(actual, torch.tensor([0.875, 0.125, 0.875, 0.375]))
        self.assertEqual(get_based_weight(torch.tensor([])).numel(), 0)

    def test_production_formula(self):
        losses = torch.tensor([3.0, 1.0, 2.0])
        counts = torch.tensor([0.0, 10.0, 100.0])
        proposal = torch.tensor([0.4, 2.0, 1.2])
        midpoint, half_width = 1.2, 0.8
        distance = (proposal - midpoint).abs() / half_width
        depth = ((midpoint - proposal) / half_width).clamp(min=0)
        scale = counts.median().clamp(min=1) * (1 + distance * (1 + depth))
        expected = midpoint + (counts / (counts + scale)).sqrt() * (proposal - midpoint)
        torch.testing.assert_close(production_get_weight(losses, 2.0, counts, 0.2), expected)
        torch.testing.assert_close(production_get_weight(losses, 2.0, None, 0.2), proposal)
        torch.testing.assert_close(production_get_weight(losses, 2.0, counts, 1.0), torch.full_like(losses, 2.0))

    def test_directional_shrinkage(self):
        actual = production_get_weight(torch.tensor([1.0, 2.0, 3.0]), 2.0, torch.full((3,), 10.0), 0.2)
        self.assertGreater(actual[0] - 1.2, 1.2 - actual[2])
        self.assertTrue(torch.all((actual >= 0.4) & (actual <= 2.0)))

    def test_factor_fusion(self):
        base = torch.tensor([0.25, 0.75])
        users, items = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
        ui, ii = torch.tensor([1, 0]), torch.tensor([0, 1])
        result = fuse_with_switches(base, users, items, ui, ii, use_bw=True, use_uf=True, use_if=True)
        torch.testing.assert_close(result[0], torch.tensor([1.5, 3.0]))


if __name__ == "__main__":
    unittest.main()
