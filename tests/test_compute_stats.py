from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from compute_stats import (
    clustered_correlation_ci,
    clustered_mean_ci,
    correlation,
    crossed_cluster_weights,
    is_successful,
    load_run_data,
    render_tables,
    sample_to_record,
)


def _record(
    voice: str,
    target: str,
    value: float,
    *,
    src: str = "en",
    tgt: str = "zh",
) -> dict[str, object]:
    return {
        "model": "test",
        "voice_id": voice,
        "target_id": target,
        "src": src,
        "tgt": tgt,
        "direction": f"{src}->{tgt}",
        "asr": value,
        "lid": 1.0 - value,
        "lid_correct": True,
        "sim": value,
        "leak": value,
        "placeholder": False,
    }


def _metric(name: str, value: float | None, **details: object) -> dict:
    return {
        "name": name,
        "status": "ok" if value is not None else "error",
        "value": value,
        "details": details,
    }


class ComputeStatsTests(unittest.TestCase):
    def test_crossed_weights_are_products_of_reference_and_text_multiplicities(self) -> None:
        records = [
            _record("v1", "t1", 1.0),
            _record("v1", "t2", 2.0),
            _record("v2", "t1", 3.0),
            _record("v2", "t2", 4.0),
        ]

        weights = crossed_cluster_weights(records, num_bootstraps=50, seed=7)

        self.assertEqual(weights.shape, (50, 4))
        for row in weights:
            matrix = row.reshape(2, 2)
            self.assertEqual(matrix[0, 0] * matrix[1, 1], matrix[0, 1] * matrix[1, 0])

    def test_crossed_weights_can_use_the_pre_filter_cluster_universe(self) -> None:
        universe = [
            _record("v1", "t1", 1.0),
            _record("v1", "t2", 2.0),
            _record("v2", "t1", 3.0),
            _record("v2", "t2", 4.0),
        ]

        weights = crossed_cluster_weights(
            [universe[0]],
            cluster_records=universe,
            num_bootstraps=100,
            seed=7,
        )

        # A success-only bootstrap must still draw v2 and t2. Consequently,
        # the sole successful cell is absent from some valid cluster draws.
        self.assertEqual(weights.shape, (100, 1))
        self.assertTrue(np.any(weights[:, 0] == 0))
        self.assertTrue(np.any(weights[:, 0] > 1))

    def test_clustered_mean_is_deterministic_and_input_order_invariant(self) -> None:
        records = [
            _record(f"v{voice}", f"t{text}", voice * 10.0 + text)
            for voice in range(3)
            for text in range(4)
        ]

        first = clustered_mean_ci(records, "leak", num_bootstraps=100, seed=19, label="grid")
        unrelated = clustered_mean_ci(records, "asr", num_bootstraps=40, seed=88, label="other")
        second = clustered_mean_ci(
            list(reversed(records)),
            "leak",
            num_bootstraps=100,
            seed=19,
            label="grid",
        )

        self.assertIsNotNone(unrelated)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], np.mean([float(row["leak"]) for row in records]))

    def test_clustered_mean_handles_incomplete_grid_and_true_zero(self) -> None:
        records = [_record("v1", "t1", 0.0), _record("v2", "t2", 0.0)]

        interval = clustered_mean_ci(records, "leak", num_bootstraps=100, seed=3, label="zero")

        self.assertEqual(interval, (0.0, 0.0, 0.0))

    def test_success_predicate_uses_correct_lid_and_strict_threshold(self) -> None:
        success = _record("v", "t", 0.099)
        boundary = _record("v", "t", 0.100)
        mismatch = _record("v", "t", 0.001)
        mismatch["lid_correct"] = False
        missing = _record("v", "t", 0.001)
        missing["asr"] = None

        self.assertTrue(is_successful(success))
        self.assertFalse(is_successful(boundary))
        self.assertFalse(is_successful(mismatch))
        self.assertFalse(is_successful(missing))

    def test_legacy_wrong_language_posterior_becomes_zero_target_lid(self) -> None:
        sample = {
            "voice": {"id": "v", "language": "ru"},
            "target": {"id": "t", "language": "en"},
            "synthesis_metadata": {},
            "metrics": [
                _metric("asr_error", 0.02),
                _metric(
                    "target_language_id",
                    0.98,
                    target_language="en",
                    detected_language="ru",
                    matches_target=False,
                ),
                _metric("speaker_similarity", None),
                _metric("normalized_leakage_delta", -0.03),
            ],
        }

        record = sample_to_record(sample, "model")

        self.assertEqual(record["lid"], 0.0)
        self.assertFalse(record["lid_correct"])
        self.assertIsNone(record["sim"])
        self.assertEqual(record["leak"], -0.03)

    def test_wrong_language_target_probability_is_also_zeroed(self) -> None:
        sample = {
            "voice": {"id": "v", "language": "ru"},
            "target": {"id": "t", "language": "en"},
            "synthesis_metadata": {},
            "metrics": [
                _metric(
                    "target_language_id",
                    0.98,
                    target_language_probability=0.25,
                    matches_target=False,
                ),
            ],
        }

        record = sample_to_record(sample, "model")

        self.assertEqual(record["lid"], 0.0)
        self.assertFalse(record["lid_correct"])

    def test_metric_specific_missingness_does_not_drop_leakage_record(self) -> None:
        sample = {
            "voice": {"id": "v", "language": "ru"},
            "target": {"id": "t", "language": "en"},
            "synthesis_metadata": {},
            "metrics": [
                _metric("asr_error", 0.02),
                _metric("target_language_id", 0.95, matches_target=True),
                _metric("speaker_similarity", None),
                _metric("normalized_leakage_delta", -0.04),
            ],
        }

        record = sample_to_record(sample, "model")

        self.assertEqual(record["leak"], -0.04)
        self.assertTrue(is_successful(record))

    def test_correlations_support_ties_and_degenerate_inputs(self) -> None:
        records = [_record(f"v{i}", f"t{i}", float(i)) for i in range(5)]
        for record in records:
            record["lid"] = -float(record["leak"])

        self.assertAlmostEqual(correlation(records, "leak", "asr", "pearson"), 1.0)
        self.assertAlmostEqual(correlation(records, "leak", "lid", "spearman"), -1.0)
        interval = clustered_correlation_ci(
            records,
            "leak",
            "asr",
            "pearson",
            num_bootstraps=50,
            seed=4,
            label="perfect",
        )
        self.assertAlmostEqual(interval[0], 1.0)

        constant = [dict(record, lid=1.0) for record in records]
        self.assertIsNone(correlation(constant, "leak", "lid", "spearman"))
        self.assertIsNone(correlation(records[:2], "leak", "asr", "pearson"))

    def test_loader_excludes_spark_unsupported_russian_and_placeholders(self) -> None:
        def sample(target_language: str, placeholder: bool) -> dict:
            return {
                "voice": {"id": "v", "language": "en"},
                "target": {"id": f"t-{target_language}-{placeholder}", "language": target_language},
                "synthesis_metadata": {"synthetic_placeholder": placeholder},
                "metrics": [],
            }

        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "results_spark_tts"
            result_dir.mkdir()
            (result_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "samples": [
                            sample("en", False),
                            sample("zh", True),
                            sample("ru", False),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            data = load_run_data(Path(tmp))

        self.assertEqual(len(data["spark_tts"]), 1)
        self.assertEqual(data["spark_tts"][0]["tgt"], "en")

    def test_render_reports_zero_success_as_missing_not_zero_metric(self) -> None:
        record = _record("v", "t", 0.2)
        record["lid_correct"] = False

        rendered = render_tables(
            {"f5tts": [record]},
            num_bootstraps=5,
            success_asr_threshold=0.1,
        )

        self.assertIn(
            "| F5-TTS | 1 | 1 | 0.200 [0.200, 0.200] | 0 | 0 (0.0%) | - | - |",
            rendered,
        )

    def test_render_uses_metric_specific_correlation_counts(self) -> None:
        records = [_record(f"v{i}", f"t{i}", float(i) / 10.0) for i in range(5)]
        records[0]["asr"] = None
        records[1]["lid"] = None

        rendered = render_tables({"f5tts": records}, num_bootstraps=20)

        row = next(
            line
            for line in rendered.splitlines()
            if line.startswith("| F5-TTS | 4 |") and line.count("|") == 8
        )
        self.assertIn(" | 4 | ", row)

    def test_aggregate_tables_do_not_drop_asr_or_lid_when_speaker_sim_is_missing(self) -> None:
        records = [_record(f"v{i}", f"t{i}", float(i) / 10.0) for i in range(5)]
        records[0]["sim"] = None

        rendered = render_tables({"f5tts": records}, num_bootstraps=20)

        self.assertIn("| F5-TTS | 5/5/4 |", rendered)


if __name__ == "__main__":
    unittest.main()
