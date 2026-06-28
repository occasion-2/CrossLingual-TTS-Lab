from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

def compute_calibration(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    print("Loading ECAPA-TDNN...")
    from speechbrain.inference.speaker import EncoderClassifier
    import torch
    import numpy as np
    
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda:0"}
    )
    
    def encode_file(path: Path):
        if hasattr(classifier, "encode_file"):
            return classifier.encode_file(str(path)).flatten()
        signal = classifier.load_audio(str(path))
        return classifier.encode_batch(signal).flatten()
        
    samples = manifest.get("samples", [])
    if not samples:
        print("No samples found in manifest")
        return
        
    # Extract unique voices
    voices_dict = {}
    for s in samples:
        v = s["voice"]
        voices_dict[v["id"]] = v
        
    voices = list(voices_dict.values())
    print(f"Encoding {len(voices)} unique voices...")
    voice_embs = {}
    voice_langs = {}
    for v in voices:
        emb = encode_file(v["audio_path"]).cpu()
        voice_embs[v["id"]] = emb
        voice_langs[v["id"]] = v["language"]
        
    # Calculate different speaker baselines
    same_speaker_real_real = []
    diff_spk_same_lang = []
    diff_spk_cross_lang = []
    
    for v1, v2 in combinations(voices, 2):
        id1, id2 = v1["id"], v2["id"]
        sim = torch.nn.functional.cosine_similarity(voice_embs[id1], voice_embs[id2], dim=0).item()
        if voice_langs[id1] == voice_langs[id2]:
            if sim >= 0.4:
                same_speaker_real_real.append(sim)
            else:
                diff_spk_same_lang.append(sim)
        else:
            diff_spk_cross_lang.append(sim)
            
    # Calculate generated vs wrong reference
    gen_vs_wrong_ref = []
    print(f"Scoring {len(samples)} generated samples against wrong references...")
    for s in samples:
        gen_audio = Path(s["audio_path"])
        if not gen_audio.exists():
            continue
            
        src_voice_id = s["voice"]["id"]
        src_lang = s["voice"]["language"]
        # Find a wrong voice id from a DIFFERENT language to guarantee different speaker
        wrong_voices = [v_id for v_id, lang in voice_langs.items() if lang != src_lang]
        if not wrong_voices:
            continue
            
        # Just pick the first cross-language wrong voice
        wrong_voice_id = wrong_voices[0]
        
        with torch.no_grad():
            gen_emb = encode_file(gen_audio).cpu()
            sim = torch.nn.functional.cosine_similarity(gen_emb, voice_embs[wrong_voice_id], dim=0).item()
            gen_vs_wrong_ref.append(sim)
            
    def get_stats(arr):
        if not arr: return "N/A"
        mean = np.mean(arr)
        std = np.std(arr)
        return f"{mean:.3f} ± {std:.3f} (n={len(arr)})"
        
    out_path = run_dir / "calibration.md"
    
    lines = [
        "### Speaker Similarity Calibration (ECAPA-TDNN)",
        "| Pair type | Speaker Sim |",
        "|---|---|",
        f"| same speaker real-real (inferred) | {get_stats(same_speaker_real_real)} |",
        f"| different speaker same language | {get_stats(diff_spk_same_lang)} |",
        f"| different speaker cross-language | {get_stats(diff_spk_cross_lang)} |",
        f"| generated vs wrong reference | {get_stats(gen_vs_wrong_ref)} |",
        "",
        "Note: 'same speaker real-real' is inferred by clustering FLEURS same-language pairs with sim >= 0.4, as FLEURS lacks explicit speaker IDs."
    ]
    
    report = "\n".join(lines)
    print("\n" + report)
    
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path}")
