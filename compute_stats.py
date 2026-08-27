from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


BOOTSTRAP_SEED = 20260628
BOOTSTRAP_RESAMPLES = 1000
SUCCESS_ASR_THRESHOLD = 0.10

MODELS = {
    "f5tts": "F5-TTS",
    "cosyvoice": "CosyVoice",
    "qwen0_6b": "Qwen3-TTS 0.6B",
    "qwen1_7b": "Qwen3-TTS 1.7B",
    "spark_tts": "Spark-TTS",
    "xtts": "XTTS v2",
}

DIRECTIONS = ("en->ru", "en->zh", "ru->en", "ru->zh", "zh->en", "zh->ru")

Record = dict[str, object]
Interval = tuple[float, float, float]


def _metric_by_name(sample: dict, name: str) -> dict | None:
    for metric in sample.get("metrics", []):
        if metric.get("name") == name:
            return metric
    return None


def _ok_metric_value(metric: dict | None) -> float | None:
    if metric is None or metric.get("status") != "ok" or metric.get("value") is None:
        return None
    try:
        value = float(metric["value"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _lid_fields(metric: dict | None) -> tuple[float | None, bool | None]:
    raw_value = _ok_metric_value(metric)
    if raw_value is None or metric is None:
        return None, None

    details = metric.get("details") or {}
    matches_target = details.get("matches_target")
    if matches_target is None:
        detected = details.get("detected_language")
        target = details.get("target_language")
        if detected is not None and target is not None:
            matches_target = detected == target

    target_probability = details.get("target_language_probability")
    if target_probability is not None:
        try:
            score = float(target_probability)
        except (TypeError, ValueError):
            return None, None
        if not math.isfinite(score):
            return None, None
        if matches_target is False:
            score = 0.0
    elif matches_target is not None:
        # Legacy manifests store the detected-language posterior even on a
        # wrong-language decision. Convert it to the conservative target score.
        score = raw_value if bool(matches_target) else 0.0
    else:
        # New manifests already store zero for a wrong-language decision.
        score = raw_value
        matches_target = score > 0.0

    return score, bool(matches_target) if matches_target is not None else None


def sample_to_record(sample: dict, model_key: str) -> Record:
    voice = sample["voice"]
    target = sample["target"]
    source_language = str(voice["language"])
    target_language = str(target["language"])
    lid, lid_correct = _lid_fields(_metric_by_name(sample, "target_language_id"))
    return {
        "model": model_key,
        "voice_id": str(voice["id"]),
        "target_id": str(target["id"]),
        "src": source_language,
        "tgt": target_language,
        "direction": f"{source_language}->{target_language}",
        "asr": _ok_metric_value(_metric_by_name(sample, "asr_error")),
        "lid": lid,
        "lid_correct": lid_correct,
        "sim": _ok_metric_value(_metric_by_name(sample, "speaker_similarity")),
        "leak": _ok_metric_value(_metric_by_name(sample, "normalized_leakage_delta")),
        "placeholder": bool(sample.get("synthesis_metadata", {}).get("synthetic_placeholder")),
    }


def load_run_data(run_root: Path) -> dict[str, list[Record]]:
    all_data: dict[str, list[Record]] = {}
    for model_key in MODELS:
        manifest_path = run_root / f"results_{model_key}" / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        all_data[model_key] = [
            sample_to_record(sample, model_key)
            for sample in manifest.get("samples", [])
            if not sample.get("synthesis_metadata", {}).get("synthetic_placeholder", False)
            # Spark-TTS documents English and Chinese synthesis only. Historical
            # manifests contain attempted target-Russian WAVs from before the
            # adapter enforced that support boundary; exclude them a priori.
            and not (model_key == "spark_tts" and sample["target"]["language"] == "ru")
        ]
    return all_data


def _derived_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _sort_records(records: Iterable[Record]) -> list[Record]:
    return sorted(
        records,
        key=lambda row: (
            str(row["model"]),
            str(row["src"]),
            str(row["voice_id"]),
            str(row["tgt"]),
            str(row["target_id"]),
        ),
    )


def _stratified_cluster_component(
    records: Sequence[Record],
    *,
    cluster_records: Sequence[Record] | None,
    cluster_key: str,
    language_key: str,
    num_bootstraps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    component = np.zeros((num_bootstraps, len(records)), dtype=np.int64)
    cluster_universe = records if cluster_records is None else cluster_records
    languages = sorted({str(record[language_key]) for record in cluster_universe})
    for language in languages:
        row_indices = [
            index for index, record in enumerate(records) if str(record[language_key]) == language
        ]
        cluster_ids = sorted(
            {
                str(record[cluster_key])
                for record in cluster_universe
                if str(record[language_key]) == language
            }
        )
        cluster_lookup = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
        counts = rng.multinomial(
            len(cluster_ids),
            np.full(len(cluster_ids), 1.0 / len(cluster_ids)),
            size=num_bootstraps,
        )
        for row_index in row_indices:
            record_cluster = str(records[row_index][cluster_key])
            if record_cluster not in cluster_lookup:
                raise ValueError(
                    f"analysis cluster {record_cluster!r} is absent from the bootstrap universe"
                )
            component[:, row_index] = counts[:, cluster_lookup[record_cluster]]
    return component


def crossed_cluster_weights(
    records: Sequence[Record],
    *,
    cluster_records: Sequence[Record] | None = None,
    num_bootstraps: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Generate language-stratified crossed (pigeonhole) bootstrap weights.

    Reference utterances and target texts are independently resampled with
    replacement within their language. Each observation receives the product
    of its reference and text multiplicities, preserving fixed direction sizes
    while accounting for both repeated factors.
    """

    # Column order follows the caller's record order. Public estimators sort
    # records before calling this helper so estimates are input-order invariant.
    ordered = list(records)
    if not ordered:
        return np.empty((num_bootstraps, 0), dtype=np.int64)
    cluster_universe = list(ordered if cluster_records is None else cluster_records)
    if not cluster_universe:
        raise ValueError("bootstrap cluster universe must not be empty")
    rng = np.random.default_rng(seed)
    reference_weights = _stratified_cluster_component(
        ordered,
        cluster_records=cluster_universe,
        cluster_key="voice_id",
        language_key="src",
        num_bootstraps=num_bootstraps,
        rng=rng,
    )
    text_weights = _stratified_cluster_component(
        ordered,
        cluster_records=cluster_universe,
        cluster_key="target_id",
        language_key="tgt",
        num_bootstraps=num_bootstraps,
        rng=rng,
    )
    return reference_weights * text_weights


def clustered_mean_ci(
    records: Sequence[Record],
    value_key: str,
    *,
    cluster_records: Sequence[Record] | None = None,
    num_bootstraps: int = BOOTSTRAP_RESAMPLES,
    ci: float = 95.0,
    seed: int = BOOTSTRAP_SEED,
    label: str = "",
) -> Interval | None:
    usable = _sort_records(record for record in records if record.get(value_key) is not None)
    if not usable:
        return None
    values = np.asarray([float(record[value_key]) for record in usable], dtype=float)
    point = float(values.mean())
    if len(usable) == 1:
        return point, point, point

    bootstrap_seed = _derived_seed(seed, label)
    replicates: list[float] = []
    for batch_index in range(100):
        remaining = num_bootstraps - len(replicates)
        if remaining <= 0:
            break
        batch_seed = (
            bootstrap_seed
            if batch_index == 0
            else _derived_seed(bootstrap_seed, f"retry|{batch_index}")
        )
        weights = crossed_cluster_weights(
            usable,
            cluster_records=cluster_records,
            num_bootstraps=remaining,
            seed=batch_seed,
        ).astype(float)
        denominators = weights.sum(axis=1)
        valid = denominators > 0
        replicates.extend(((weights[valid] @ values) / denominators[valid]).tolist())
    if len(replicates) < num_bootstraps:
        raise RuntimeError(
            f"only {len(replicates)} valid clustered mean resamples out of {num_bootstraps}"
        )
    alpha = (100.0 - ci) / 2.0
    lower, upper = np.percentile(replicates, [alpha, 100.0 - alpha])
    return point, float(lower), float(upper)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation(records: Sequence[Record], x_key: str, y_key: str, method: str) -> float | None:
    usable = [
        record
        for record in records
        if record.get(x_key) is not None and record.get(y_key) is not None
    ]
    if len(usable) < 3:
        return None
    x = np.asarray([float(record[x_key]) for record in usable], dtype=float)
    y = np.asarray([float(record[y_key]) for record in usable], dtype=float)
    if method == "spearman":
        x = _rankdata(x)
        y = _rankdata(y)
    elif method != "pearson":
        raise ValueError(f"unsupported correlation method: {method}")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def clustered_correlation_ci(
    records: Sequence[Record],
    x_key: str,
    y_key: str,
    method: str,
    *,
    cluster_records: Sequence[Record] | None = None,
    num_bootstraps: int = BOOTSTRAP_RESAMPLES,
    ci: float = 95.0,
    seed: int = BOOTSTRAP_SEED,
    label: str = "",
) -> Interval | None:
    usable = _sort_records(
        record
        for record in records
        if record.get(x_key) is not None and record.get(y_key) is not None
    )
    point = correlation(usable, x_key, y_key, method)
    if point is None:
        return None

    bootstrap_seed = _derived_seed(seed, label)
    replicates: list[float] = []
    for batch_index in range(100):
        remaining = num_bootstraps - len(replicates)
        if remaining <= 0:
            break
        batch_seed = (
            bootstrap_seed
            if batch_index == 0
            else _derived_seed(bootstrap_seed, f"retry|{batch_index}")
        )
        weights = crossed_cluster_weights(
            usable,
            cluster_records=cluster_records,
            num_bootstraps=remaining,
            seed=batch_seed,
        )
        for row_weights in weights:
            indices = np.repeat(np.arange(len(usable)), row_weights)
            if len(indices) < 3:
                continue
            replicate = correlation([usable[index] for index in indices], x_key, y_key, method)
            if replicate is not None and math.isfinite(replicate):
                replicates.append(replicate)
    if len(replicates) < num_bootstraps:
        raise RuntimeError(
            f"only {len(replicates)} valid clustered correlation resamples out of "
            f"{num_bootstraps}"
        )
    alpha = (100.0 - ci) / 2.0
    lower, upper = np.percentile(replicates, [alpha, 100.0 - alpha])
    return point, float(lower), float(upper)


def is_successful(record: Record, asr_threshold: float = SUCCESS_ASR_THRESHOLD) -> bool:
    return (
        record.get("asr") is not None
        and record.get("lid") is not None
        and record.get("lid_correct") is True
        and float(record["asr"]) < asr_threshold
    )


def _is_success_eligible(record: Record) -> bool:
    return (
        record.get("asr") is not None
        and record.get("lid") is not None
        and record.get("lid_correct") is not None
    )


def _format_ci(interval: Interval | None, *, percent: bool = False) -> str:
    if interval is None:
        return "-"
    point, lower, upper = interval
    if percent:
        return f"{point * 100:.1f}% [{lower * 100:.1f}, {upper * 100:.1f}]"
    return f"{point:.3f} [{lower:.3f}, {upper:.3f}]"


def _filtered(records: Iterable[Record], predicate: Callable[[Record], bool]) -> list[Record]:
    return [record for record in records if predicate(record)]


def _complete(records: Iterable[Record], *keys: str) -> list[Record]:
    return [record for record in records if all(record.get(key) is not None for key in keys)]


def _metric_summary(
    records: Sequence[Record],
    *,
    num_bootstraps: int,
    seed: int,
    label: str,
) -> dict[str, Interval | None]:
    return {
        key: clustered_mean_ci(
            records,
            key,
            cluster_records=records,
            num_bootstraps=num_bootstraps,
            seed=seed,
            label=label,
        )
        for key in ("asr", "lid", "sim", "leak")
    }


def render_tables(
    all_data: dict[str, list[Record]],
    *,
    num_bootstraps: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    success_asr_threshold: float = SUCCESS_ASR_THRESHOLD,
) -> str:
    lines: list[str] = []

    def add(*items: str) -> None:
        lines.extend(items)

    add(
        "<!-- Generated by compute_stats.py. -->",
        f"<!-- Stratified crossed reference/text bootstrap: {num_bootstraps} resamples; seed: {seed}. -->",
        "",
        "### Table 1: Common Target-Language Subset (en, zh targets)",
        "*The subset is defined from documented target-language support, not observed model quality. It does not assert documented support for every source-prompt language or cross-lingual direction. All confidence intervals use a crossed reference/text cluster bootstrap.*",
        "",
        "| Model | n (ASR/LID/spk) | ASR Error ↓ (95% CI) | Target LID score ↑ (95% CI) | Speaker Sim ↑ (95% CI) |",
        "|---|---:|---:|---:|---:|",
    )
    for model_key, model_name in MODELS.items():
        records = _filtered(
            all_data.get(model_key, []), lambda row: row["tgt"] in {"en", "zh"}
        )
        if not records:
            continue
        stats = _metric_summary(
            records,
            num_bootstraps=num_bootstraps,
            seed=seed,
            label=f"common|{model_key}",
        )
        counts = "/".join(str(len(_complete(records, key))) for key in ("asr", "lid", "sim"))
        add(
            f"| {model_name} | {counts} | {_format_ci(stats['asr'], percent=True)} | "
            f"{_format_ci(stats['lid'], percent=True)} | {_format_ci(stats['sim'])} |"
        )

    add(
        "",
        "### Table 2: Target-Language Aggregates",
        "*Aggregated across source languages. F5-TTS target-Russian rows are out-of-support stress tests for the exact English/Mandarin checkpoint; Spark-TTS has no target-Russian rows.*",
        "",
        "| Model | Target | n (ASR/LID/spk) | ASR Error ↓ (95% CI) | Target LID score ↑ (95% CI) | Speaker Sim ↑ (95% CI) |",
        "|---|---|---:|---:|---:|---:|",
    )
    for target in ("en", "ru", "zh"):
        for model_key, model_name in MODELS.items():
            records = _filtered(
                all_data.get(model_key, []), lambda row, t=target: row["tgt"] == t
            )
            if not records:
                add(f"| {model_name} | {target} | 0/0/0 | - | - | - |")
                continue
            stats = _metric_summary(
                records,
                num_bootstraps=num_bootstraps,
                seed=seed,
                label=f"target|{model_key}|{target}",
            )
            counts = "/".join(
                str(len(_complete(records, key))) for key in ("asr", "lid", "sim")
            )
            add(
                f"| {model_name} | {target} | {counts} | {_format_ci(stats['asr'], percent=True)} | "
                f"{_format_ci(stats['lid'], percent=True)} | {_format_ci(stats['sim'])} |"
            )

    add(
        "",
        "### Table 3: Source-Language Aggregates (Speaker Similarity)",
        "*Aggregated by reference language within each artifact. Target-language coverage differs for Spark-TTS, so these rows are not cross-model comparisons.*",
        "",
        "| Model | Source | n | Speaker Sim ↑ (95% CI) |",
        "|---|---|---:|---:|",
    )
    for source in ("en", "ru", "zh"):
        for model_key, model_name in MODELS.items():
            scope = _filtered(
                all_data.get(model_key, []), lambda row, s=source: row["src"] == s
            )
            records = _complete(scope, "sim")
            interval = clustered_mean_ci(
                records,
                "sim",
                cluster_records=scope,
                num_bootstraps=num_bootstraps,
                seed=seed,
                label=f"source|{model_key}|{source}",
            )
            add(f"| {model_name} | {source} | {len(records)} | {_format_ci(interval)} |")

    add(
        "",
        "### Table 4: Per-Direction Results",
        "*F5-TTS target-Russian rows are out-of-support stress tests for the exact checkpoint.*",
        "",
        "| Model | Direction | n (ASR/LID/spk) | ASR Error ↓ (95% CI) | Target LID score ↑ (95% CI) | Speaker Sim ↑ (95% CI) |",
        "|---|---|---:|---:|---:|---:|",
    )
    for model_key, model_name in MODELS.items():
        for direction in DIRECTIONS:
            records = _filtered(
                all_data.get(model_key, []),
                lambda row, d=direction: row["direction"] == d,
            )
            if not records:
                add(f"| {model_name} | {direction} | 0/0/0 | - | - | - |")
                continue
            stats = _metric_summary(
                records,
                num_bootstraps=num_bootstraps,
                seed=seed,
                label=f"direction|{model_key}|{direction}",
            )
            counts = "/".join(
                str(len(_complete(records, key))) for key in ("asr", "lid", "sim")
            )
            add(
                f"| {model_name} | {direction} | {counts} | {_format_ci(stats['asr'], percent=True)} | "
                f"{_format_ci(stats['lid'], percent=True)} | {_format_ci(stats['sim'])} |"
            )

    add(
        "",
        "### Table 5: Language-Centroid Leakage Proxy by Direction",
        "*The value is cosine similarity to the source centroid minus similarity to the target centroid; lower is better. F5-TTS target-Russian rows are out-of-support stress tests.*",
        "",
        "| Model | Direction | n | Leakage Delta ↓ (95% CI) |",
        "|---|---|---:|---:|",
    )
    for model_key, model_name in MODELS.items():
        for direction in DIRECTIONS:
            scope = _filtered(
                all_data.get(model_key, []),
                lambda row, d=direction: row["direction"] == d,
            )
            records = _complete(scope, "leak")
            interval = clustered_mean_ci(
                records,
                "leak",
                cluster_records=scope,
                num_bootstraps=num_bootstraps,
                seed=seed,
                label=f"leak|{model_key}|{direction}",
            )
            add(f"| {model_name} | {direction} | {len(records)} | {_format_ci(interval)} |")

    threshold_percent = success_asr_threshold * 100.0
    add(
        "",
        "### Table 6: Leakage after the Automatic ASR/LID Screen",
        f"*A pass requires correct target-language identification and ASR error < {threshold_percent:g}%. This screen does not verify decoder termination; see the generation-cap sensitivity in the paper and README. F5-TTS target-Russian rows are out-of-support stress tests. Confidence intervals remain clustered by reference and target text.*",
        "",
        "| Model | Eligible n | All leakage n | All-sample Leakage Delta ↓ (95% CI) | ASR/LID-pass n | Pass + leakage n (%) | Screened Leakage Delta ↓ (95% CI) | Direction-mean range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    )
    for model_key, model_name in MODELS.items():
        model_records = all_data.get(model_key, [])
        all_leakage = _complete(model_records, "leak")
        all_interval = clustered_mean_ci(
            all_leakage,
            "leak",
            cluster_records=model_records,
            num_bootstraps=num_bootstraps,
            seed=seed,
            label=f"all-success-comparison|{model_key}",
        )
        eligible = _filtered(model_records, _is_success_eligible)
        successful = _filtered(eligible, lambda row: is_successful(row, success_asr_threshold))
        successful_leakage = _complete(successful, "leak")
        interval = clustered_mean_ci(
            successful_leakage,
            "leak",
            cluster_records=eligible,
            num_bootstraps=num_bootstraps,
            seed=seed,
            label=f"success|{model_key}",
        )
        direction_means = []
        for direction in DIRECTIONS:
            values = [
                float(row["leak"])
                for row in successful_leakage
                if row["direction"] == direction
            ]
            if values:
                direction_means.append(float(np.mean(values)))
        direction_range = (
            f"[{min(direction_means):.3f}, {max(direction_means):.3f}]"
            if direction_means
            else "-"
        )
        success_rate = 100.0 * len(successful_leakage) / len(eligible) if eligible else 0.0
        add(
            f"| {model_name} | {len(eligible)} | {len(all_leakage)} | {_format_ci(all_interval)} | "
            f"{len(successful)} | "
            f"{len(successful_leakage)} ({success_rate:.1f}%) | {_format_ci(interval)} | {direction_range} |"
        )

    add(
        "",
        "### Table 7: Leakage after the Automatic ASR/LID Screen, by Direction",
        f"*A pass requires correct target LID and ASR error < {threshold_percent:g}%. It does not verify decoder termination. F5-TTS target-Russian rows are out-of-support stress tests.*",
        "",
        "| Model | Direction | Eligible n | Pass + leakage n | Screened Leakage Delta ↓ (95% CI) |",
        "|---|---|---:|---:|---:|",
    )
    for model_key, model_name in MODELS.items():
        for direction in DIRECTIONS:
            eligible = _filtered(
                _filtered(
                    all_data.get(model_key, []),
                    lambda row, d=direction: row["direction"] == d,
                ),
                _is_success_eligible,
            )
            successful = _complete(
                _filtered(eligible, lambda row: is_successful(row, success_asr_threshold)),
                "leak",
            )
            interval = clustered_mean_ci(
                successful,
                "leak",
                cluster_records=eligible,
                num_bootstraps=num_bootstraps,
                seed=seed,
                label=f"success-direction|{model_key}|{direction}",
            )
            add(
                f"| {model_name} | {direction} | {len(eligible)} | {len(successful)} | "
                f"{_format_ci(interval)} |"
            )

    add(
        "",
        "### Table 8: Leakage Correlation with LID and ASR",
        "*Correlations use metric-specific complete cases and are pooled across directions, so they can reflect direction composition and do not establish incremental validity. F5-TTS includes out-of-support target-Russian stress rows. The target-LID score is zero for wrong-language detections by construction. CIs use the same crossed reference/text bootstrap.*",
        "",
        "| Model | n(Δ,LID) | Pearson r(Δ,LID) | Spearman ρ(Δ,LID) | n(Δ,ASR) | Pearson r(Δ,ASR) | Spearman ρ(Δ,ASR) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    )
    for model_key, model_name in MODELS.items():
        model_records = all_data.get(model_key, [])
        lid_records = _complete(model_records, "leak", "lid")
        asr_records = _complete(model_records, "leak", "asr")
        lid_intervals = [
            clustered_correlation_ci(
                lid_records,
                "leak",
                "lid",
                method,
                cluster_records=model_records,
                num_bootstraps=num_bootstraps,
                seed=seed,
                label=f"correlation|{model_key}",
            )
            for method in ("pearson", "spearman")
        ]
        asr_intervals = [
            clustered_correlation_ci(
                asr_records,
                "leak",
                "asr",
                method,
                cluster_records=model_records,
                num_bootstraps=num_bootstraps,
                seed=seed,
                label=f"correlation|{model_key}",
            )
            for method in ("pearson", "spearman")
        ]
        add(
            f"| {model_name} | {len(lid_records)} | "
            + " | ".join(_format_ci(interval) for interval in lid_intervals)
            + f" | {len(asr_records)} | "
            + " | ".join(_format_ci(interval) for interval in asr_intervals)
            + " |"
        )

    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper tables with crossed reference/text cluster confidence intervals."
    )
    parser.add_argument("--run-root", type=Path, default=Path("overnight_runs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--success-asr-threshold", type=float, default=SUCCESS_ASR_THRESHOLD)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_resamples < 1:
        raise SystemExit("--bootstrap-resamples must be positive")
    if not 0.0 <= args.success_asr_threshold <= 1.0:
        raise SystemExit("--success-asr-threshold must be between 0 and 1")

    all_data = load_run_data(args.run_root)
    if not all_data:
        raise SystemExit(f"no result manifests found under {args.run_root}")
    rendered = render_tables(
        all_data,
        num_bootstraps=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        success_asr_threshold=args.success_asr_threshold,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
