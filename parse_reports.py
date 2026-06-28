import re
import glob

models = {
    "f5tts": "F5-TTS (385M)",
    "cosyvoice": "CosyVoice (300M)",
    "qwen0_6b": "Qwen3-TTS 0.6B",
    "qwen1_7b": "Qwen3-TTS 1.7B",
    "spark_tts": "Spark-TTS 0.5B",
    "xtts": "XTTS v2 (400M)",
}

for key, name in models.items():
    report_file = f"overnight_runs/results_{key}/report.md"
    try:
        with open(report_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"| {name} | Missing report |")
        continue

    metrics_list = []
    in_table = False
    for line in lines:
        if line.startswith("| Direction |"):
            in_table = True
        elif in_table and line.startswith("##"):
            break
        elif in_table and line.startswith("|") and "asr_error" in line:
            parts = line.split("|")
            direction = parts[1].strip()
            # If model is spark_tts and target is ru, we skip?
            # User said "pass - for unsupported languages".
            # Let's just collect everything first.
            metrics = parts[3].strip()
            # format: asr_error: 1.175; speaker_similarity: 0.331; target_language_id: 0.649
            m = re.search(r"asr_error: ([\d.]+); speaker_similarity: ([\d.]+); target_language_id: ([\d.]+)", metrics)
            if m:
                asr, sim, lid = float(m.group(1)), float(m.group(2)), float(m.group(3))
                # For spark_tts, en->ru and zh->ru are unsupported.
                # If they are unsupported, do we skip them?
                if key == "spark_tts" and "->ru" in direction:
                    continue
                metrics_list.append((asr, lid, sim))
    
    if metrics_list:
        avg_asr = sum(x[0] for x in metrics_list) / len(metrics_list)
        avg_lid = sum(x[1] for x in metrics_list) / len(metrics_list)
        avg_sim = sum(x[2] for x in metrics_list) / len(metrics_list)
        unsupported = "ru" if key == "spark_tts" else "-"
        print(f"| {name} | {avg_asr*100:.1f}% | {avg_lid*100:.1f}% | {avg_sim:.3f} | {unsupported} |")

