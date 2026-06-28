import re
import glob

models = {
    "f5tts": "F5-TTS",
    "cosyvoice": "CosyVoice",
    "qwen0_6b": "Qwen3-TTS 0.6B",
    "qwen1_7b": "Qwen3-TTS 1.7B",
    "spark_tts": "Spark-TTS",
    "xtts": "XTTS v2",
}

all_directions = ["en->ru", "en->zh", "ru->en", "ru->zh", "zh->en", "zh->ru"]
common_subset_directions = ["en->zh", "ru->en", "ru->zh", "zh->en"]

def parse_report(filepath):
    metrics_by_dir = {}
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return metrics_by_dir

    in_table = False
    for line in lines:
        if line.startswith("| Direction |"):
            in_table = True
        elif in_table and line.startswith("##"):
            break
        elif in_table and line.startswith("|") and "asr_error" in line:
            parts = line.split("|")
            direction = parts[1].strip()
            metrics = parts[3].strip()
            m = re.search(r"asr_error: ([\d.]+); speaker_similarity: ([\d.]+); target_language_id: ([\d.]+)", metrics)
            if m:
                metrics_by_dir[direction] = {
                    "asr": float(m.group(1)),
                    "sim": float(m.group(2)),
                    "lid": float(m.group(3)),
                }
    return metrics_by_dir

model_data = {}
for key, name in models.items():
    model_data[key] = parse_report(f"overnight_runs/results_{key}/report.md")

print("### Table 1: Common Subset Only (en, zh targets)")
print()
print("| Model | ASR Error ↓ | Target LID ↑ | Speaker Sim ↑ |")
print("|---|---|---|---|")
for key, name in models.items():
    data = model_data[key]
    subset_data = [data[d] for d in common_subset_directions if d in data]
    # For spark_tts, en->ru and zh->ru are naturally missing.
    # We just average the subset.
    if len(subset_data) == 0:
        print(f"| {name} | - | - | - |")
        continue
    avg_asr = sum(x["asr"] for x in subset_data) / len(subset_data)
    avg_lid = sum(x["lid"] for x in subset_data) / len(subset_data)
    avg_sim = sum(x["sim"] for x in subset_data) / len(subset_data)
    print(f"| {name} | {avg_asr*100:.1f}% | {avg_lid*100:.1f}% | {avg_sim:.3f} |")

print("\n### Table 2: Per-Direction Breakdowns")
print()
print("| Model | Direction | ASR Error ↓ | Target LID ↑ | Speaker Sim ↑ |")
print("|---|---|---|---|---|")
for key, name in models.items():
    data = model_data[key]
    for d in all_directions:
        if d in data:
            # Note: For Spark-TTS we already know ru targets are skipped but let's check if they exist
            # If they exist and have bad numbers, we might still print them or skip?
            # Actually, spark_tts generated empty/placeholder numbers.
            # But earlier we removed placeholder generation and marked it as missing. So we shouldn't have metrics for it.
            # Wait, the overnight runs were BEFORE the fix! So spark_tts has ru metrics that are just bad.
            # F5 has bad ru targets too.
            # Let's print them anyway to show they fail.
            asr = data[d]["asr"]
            lid = data[d]["lid"]
            sim = data[d]["sim"]
            print(f"| {name} | {d} | {asr*100:.1f}% | {lid*100:.1f}% | {sim:.3f} |")
        else:
            print(f"| {name} | {d} | - | - | - |")

