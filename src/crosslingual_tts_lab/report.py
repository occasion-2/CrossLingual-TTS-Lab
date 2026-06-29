from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def write_reports(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = manifest_path.with_name("report.md")
    report_path.write_text(render_markdown_report(manifest), encoding="utf-8")
    return report_path


def render_markdown_report(manifest: dict) -> str:
    benchmark = manifest["benchmark"]
    summary = manifest["summary"]
    
    has_placeholders = False
    for sample in manifest.get("samples", []):
        metadata = sample.get("synthesis_metadata", {})
        if metadata.get("synthetic_placeholder", False) or metadata.get("synthesis_failed", False):
            has_placeholders = True
            break
        for metric in sample.get("metrics", []):
            if metric.get("status") in {"missing_backend", "synthesis_failed"} or "proxy" in metric.get("name", ""):
                has_placeholders = True
                break

    lines = [
        f"# {benchmark['name']}",
        "",
    ]
    if benchmark.get("description"):
        lines += [benchmark["description"], ""]

    device = manifest.get("device_profile", {})
    lines += [
        "## Summary",
        "",
        f"- Models: {summary['models']}",
        f"- Voices: {summary['voices']}",
        f"- Targets: {summary['targets']}",
        f"- Jobs: {summary['jobs']}",
        f"- Cross-lingual jobs: {summary['cross_lingual_jobs']}",
        f"- Device: {device.get('device', 'unknown')}",
        f"- Recommended ASR model: {device.get('recommended_whisper_model', 'unknown')}",
        f"- Recommended compute: {device.get('recommended_compute_type', 'unknown')}",
        "",
        "## Direction Overview",
        "",
        f"| Direction | Jobs | {'Placeholder metrics' if has_placeholders else 'Metrics'} |",
        "| --- | ---: | --- |",
    ]

    by_direction = defaultdict(list)
    for sample in manifest["samples"]:
        by_direction[sample["direction"]].append(sample)

    for direction, samples in sorted(by_direction.items()):
        metric_cells = []
        metric_names = sorted({metric["name"] for sample in samples for metric in sample["metrics"]})
        for name in metric_names:
            values = [
                float(metric["value"])
                for sample in samples
                for metric in sample["metrics"]
                if metric["name"] == name and isinstance(metric["value"], (int, float))
            ]
            if values:
                metric_cells.append(f"{name}: {mean(values):.3f}")
        lines.append(f"| {direction} | {len(samples)} | {'; '.join(metric_cells)} |")

    lines += [
        "",
        "## Samples",
        "",
        "| Job | Model | Voice | Target | Audio | Metric status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for sample in manifest["samples"]:
        statuses = sorted({metric["status"] for metric in sample["metrics"]})
        lines.append(
            "| {job_id} | {model} | {voice} | {target} | {audio} | {status} |".format(
                job_id=sample["job_id"],
                model=sample["model"]["id"],
                voice=f"{sample['voice']['id']} ({sample['voice']['language']})",
                target=f"{sample['target']['id']} ({sample['target']['language']})",
                audio=sample["audio_path"],
                status=", ".join(statuses),
            )
        )

    lines += ["", "## Notes", ""]
    if has_placeholders:
        lines += [
            "Some metric values are placeholders or were skipped after synthesis failures. Treat affected rows as pipeline checks, not scientific measurements.",
            "Replace placeholder metrics with real backends and inspect synthesis_failed rows before drawing conclusions.",
        ]
    else:
        lines += [
            "Metric values were computed using real evaluation backends.",
        ]
    lines.append("")
    return "\n".join(lines)
