import json
import glob
import numpy as np
import os

models = {
    "f5tts": "F5-TTS",
    "cosyvoice": "CosyVoice",
    "qwen0_6b": "Qwen3-TTS 0.6B",
    "qwen1_7b": "Qwen3-TTS 1.7B",
    "spark_tts": "Spark-TTS",
    "xtts": "XTTS v2",
}

def bootstrap_ci(data, num_bootstraps=1000, ci=95):
    if not data:
        return 0.0, 0.0, 0.0
    data = np.array(data)
    if len(data) == 1:
        return data[0], data[0], data[0]
    bootstrapped_means = np.random.choice(data, size=(num_bootstraps, len(data)), replace=True).mean(axis=1)
    lower = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    return data.mean(), lower, upper

def format_ci(mean, lower, upper, is_pct=False):
    if is_pct:
        return f"{mean*100:.1f}% [{lower*100:.1f}–{upper*100:.1f}]"
    return f"{mean:.3f} [{lower:.3f}–{upper:.3f}]"

all_data = {}

for key, name in models.items():
    manifest_path = f"overnight_runs/results_{key}/manifest.json"
    if not os.path.exists(manifest_path):
        continue
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    samples = manifest.get("samples", [])
    model_samples = []
    
    for s in samples:
        src = s["voice"]["language"]
        tgt = s["target"]["language"]
        direction = f"{src}->{tgt}"
        
        # Manually filter Spark-TTS unsupported ru targets
        if key == "spark_tts" and tgt == "ru":
            continue
            
        asr_val, lid_val, sim_val = None, None, None
        for m in s.get("metrics", []):
            if m["name"] == "asr_error" and m["value"] is not None:
                asr_val = m["value"]
            elif m["name"] == "target_language_id" and m["value"] is not None:
                lid_val = m["value"]
            elif m["name"] == "speaker_similarity" and m["value"] is not None:
                sim_val = m["value"]
        
        if asr_val is not None and lid_val is not None and sim_val is not None:
            model_samples.append({
                "src": src,
                "tgt": tgt,
                "direction": direction,
                "asr": asr_val,
                "lid": lid_val,
                "sim": sim_val
            })
    all_data[key] = model_samples

def get_stats(samples, filter_func=lambda x: True):
    filtered = [s for s in samples if filter_func(s)]
    if not filtered:
        return None, 0
    n = len(filtered)
    asr_mean, asr_l, asr_u = bootstrap_ci([s["asr"] for s in filtered])
    lid_mean, lid_l, lid_u = bootstrap_ci([s["lid"] for s in filtered])
    sim_mean, sim_l, sim_u = bootstrap_ci([s["sim"] for s in filtered])
    return {
        "n": n,
        "asr": (asr_mean, asr_l, asr_u),
        "lid": (lid_mean, lid_l, lid_u),
        "sim": (sim_mean, sim_l, sim_u)
    }, n

print("### Table 1: Common Subset Only (en, zh targets)")
print("*Excludes directions unsupported or highly degraded by certain models (e.g., Russian targets for Spark-TTS and F5-TTS).*")
print()
print("| Model | n | ASR Error ↓ (95% CI) | Target LID ↑ (95% CI) | Speaker Sim ↑ (95% CI) |")
print("|---|---|---|---|---|")
for key, name in models.items():
    if key not in all_data: continue
    stats, n = get_stats(all_data[key], lambda x: x["tgt"] in ["en", "zh"])
    if not stats: continue
    asr_str = format_ci(*stats["asr"], True)
    lid_str = format_ci(*stats["lid"], True)
    sim_str = format_ci(*stats["sim"], False)
    print(f"| {name} | {n} | {asr_str} | {lid_str} | {sim_str} |")


print("\n### Table 2: Target-Language Aggregates")
print("*Aggregated by target language across all sources.*")
print()
print("| Model | Target | n | ASR Error ↓ (95% CI) | Target LID ↑ (95% CI) | Speaker Sim ↑ (95% CI) |")
print("|---|---|---|---|---|---|")
for tgt in ["en", "ru", "zh"]:
    for key, name in models.items():
        if key not in all_data: continue
        stats, n = get_stats(all_data[key], lambda x: x["tgt"] == tgt)
        if not stats:
            print(f"| {name} | {tgt} | 0 | - | - | - |")
            continue
        asr_str = format_ci(*stats["asr"], True)
        lid_str = format_ci(*stats["lid"], True)
        sim_str = format_ci(*stats["sim"], False)
        print(f"| {name} | {tgt} | {n} | {asr_str} | {lid_str} | {sim_str} |")


print("\n### Table 3: Source-Language Aggregates (Speaker Similarity)")
print("*Aggregated by source language to show how well each model retains speaker identity across origin languages.*")
print()
print("| Model | Source | n | Speaker Sim ↑ (95% CI) |")
print("|---|---|---|---|")
for src in ["en", "ru", "zh"]:
    for key, name in models.items():
        if key not in all_data: continue
        stats, n = get_stats(all_data[key], lambda x: x["src"] == src)
        if not stats:
            print(f"| {name} | {src} | 0 | - |")
            continue
        sim_str = format_ci(*stats["sim"], False)
        print(f"| {name} | {src} | {n} | {sim_str} |")


print("\n### Table 4: Per-Direction Breakdowns")
print("*Provides full visibility into specific language pairs, exposing asymmetric performance.*")
print()
print("| Model | Direction | n | ASR Error ↓ (95% CI) | Target LID ↑ (95% CI) | Speaker Sim ↑ (95% CI) |")
print("|---|---|---|---|---|---|")
for key, name in models.items():
    if key not in all_data: continue
    for d in ["en->ru", "en->zh", "ru->en", "ru->zh", "zh->en", "zh->ru"]:
        stats, n = get_stats(all_data[key], lambda x: x["direction"] == d)
        if not stats:
            print(f"| {name} | {d} | 0 | - | - | - |")
            continue
        asr_str = format_ci(*stats["asr"], True)
        lid_str = format_ci(*stats["lid"], True)
        sim_str = format_ci(*stats["sim"], False)
        print(f"| {name} | {d} | {n} | {asr_str} | {lid_str} | {sim_str} |")

