"""FP32 / FP16 / INT8 quantization benchmark for Style-Bert-VITS2.

Measures: inference time, RTF, BERT vs Synth breakdown, memory, audio quality.

Usage:
    python -m dev_tools.benchmark_quantization

Run from the project root directory (Style-Bert-VITS2/).
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import psutil
import torch

# ── Config ──────────────────────────────────────────────
MODEL_DIR = str(_PROJECT_ROOT / "model_assets" / "Elaina")
OUTPUT_DIR = _PROJECT_ROOT / "dev_tools" / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 44100
REPEATS = 5  # runs per text (after warmup)

# Diverse test texts: varying length, Japanese-only, mixed, punctuation
TEST_TEXTS = [
    # Short
    ("short_ja", "はい、わかりました。"),
    ("short_mix", "OK、了解です!"),
    # Medium
    ("med_ja", "先生、大丈夫ですか？今日はちょっと疲れてるみたいですね。"),
    ("med_mix", "今日のmeetingは何時から?conferenceのroomを予約した?"),
    # Long
    ("long_ja", "本日は天気が良いので、少し散歩に出かけようと思います。公園のベンチで本を読むのもいいかもしれませんね。"),
    ("long_mix", "machinelearningのmodelをtrainingしてるんだけど、accuracyが上がらない。datasetのqualityに問題があるのかも。"),
    # Very long
    ("vlong_ja", "今朝のニュースによると、来月から新しい法律が施行されるそうです。市民の皆さんは、詳しい内容について市役所のウェブサイトで確認することができます。何かご不明な点がございましたら、お気軽にお問い合わせください。"),
    ("vlong_mix", "projectのdeadlineが来週のfridayなんだけど、designのreviewがまだ終わってない。developmentチームとcommunicationを取って、priorityを再確認する必要がある。schedulingのmeetingをtomorrowにセットアップしてくれる?"),
]

# NOTE: Synthesizer must stay FP32 — Flow layers (rational_quadratic_spline)
# are numerically unstable in FP16.
CONFIGS = [
    {"name": "fp32_gpu", "device": "cuda", "bert_dtype": "fp32"},
    {"name": "fp16_gpu", "device": "cuda", "bert_dtype": "fp16"},
    {"name": "fp32_cpu", "device": "cpu", "bert_dtype": "fp32"},
    {"name": "q8_cpu",   "device": "cpu", "bert_dtype": "q8"},
]


# ── Helpers ─────────────────────────────────────────────
def mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def gpu_mb():
    return torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0


def audio_duration_sec(audio_result) -> float:
    """Get duration of generated audio in seconds."""
    return len(audio_result.data) / SAMPLE_RATE


def reset_all():
    from style_bert_vits2.nlp import bert_models
    bert_models.unload_model()
    bert_models.unload_tokenizer()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)


# ── Benchmark per config ────────────────────────────────
def benchmark_config(config: dict) -> dict:
    from style_bert_vits2 import TTS, Lang
    from style_bert_vits2.nlp import bert_models

    name = config["name"]
    device = config["device"]
    bert_dtype = config["bert_dtype"]

    if "cuda" in device and not torch.cuda.is_available():
        print(f"  SKIP {name} (no CUDA)")
        return {"name": name, "skipped": True}

    print(f"\n{'='*60}")
    print(f"  {name} | device={device} bert={bert_dtype} synth=fp32")
    print(f"{'='*60}")

    reset_all()
    mem_base = mem_mb()
    gpu_base = gpu_mb()

    # ── Load ──
    t0 = time.time()
    tts = TTS(device=device)
    bert_load_sec = time.time() - t0

    # Apply BERT dtype
    if bert_dtype == "fp16" and "cuda" in device:
        bert_models._loaded_model.half()
        print("  BERT → FP16")
    elif bert_dtype == "q8":
        original = bert_models._loaded_model
        quantized = torch.quantization.quantize_dynamic(
            original, {torch.nn.Linear}, dtype=torch.qint8
        )
        del original
        bert_models._loaded_model = quantized
        gc.collect()
        print("  BERT → INT8 (original freed)")

    mem_after_bert = mem_mb()
    gpu_after_bert = gpu_mb()

    t0 = time.time()
    speaker = tts.load("elaina", MODEL_DIR)
    speaker._ensure_model()
    synth_load_sec = time.time() - t0

    mem_loaded = mem_mb()
    gpu_loaded = gpu_mb()
    print(f"  Loaded: RAM +{mem_loaded - mem_base:.0f}MB, GPU +{gpu_loaded - gpu_base:.0f}MB")

    # ── Warmup (2 runs, discard) ──
    print("  Warmup (2 runs)...")
    for _ in range(2):
        speaker.generate(TEST_TEXTS[0][1], lang=Lang.JA, style="Neutral")

    mem_warm = mem_mb()
    gpu_warm = gpu_mb()
    print(f"  Warmed: RAM +{mem_warm - mem_base:.0f}MB, GPU +{gpu_warm - gpu_base:.0f}MB")

    # ── Benchmark ──
    print(f"  Benchmarking ({REPEATS} runs x {len(TEST_TEXTS)} texts)...")
    text_results = []

    for label, text in TEST_TEXTS:
        times = []
        audio_dur = 0
        for r in range(REPEATS):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            audio = speaker.generate(text, lang=Lang.JA, style="Neutral")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            times.append(elapsed)
            audio_dur = audio_duration_sec(audio)

            # Save audio from first run only
            if r == 0:
                audio.save(str(OUTPUT_DIR / f"{name}_{label}.wav"))

        avg = np.mean(times)
        std = np.std(times)
        rtf = audio_dur / avg if avg > 0 else 0

        text_results.append({
            "label": label,
            "text": text,
            "times": [round(t, 3) for t in times],
            "avg_time": round(avg, 3),
            "std_time": round(std, 3),
            "audio_duration": round(audio_dur, 2),
            "rtf": round(rtf, 2),
        })
        print(f"    {label:12s} | {avg:.3f}s ±{std:.3f} | audio={audio_dur:.2f}s | RTF={rtf:.2f}x")

    avg_all = np.mean([r["avg_time"] for r in text_results])
    avg_rtf = np.mean([r["rtf"] for r in text_results])

    result = {
        "name": name,
        "device": device,
        "bert_dtype": bert_dtype,
        "synth_dtype": "fp32",
        "skipped": False,
        "bert_load_sec": round(bert_load_sec, 2),
        "synth_load_sec": round(synth_load_sec, 2),
        "ram_after_load_mb": round(mem_loaded - mem_base),
        "ram_after_warm_mb": round(mem_warm - mem_base),
        "gpu_after_load_mb": round(gpu_loaded - gpu_base),
        "gpu_after_warm_mb": round(gpu_warm - gpu_base),
        "avg_inference_sec": round(avg_all, 3),
        "avg_rtf": round(avg_rtf, 2),
        "per_text": text_results,
    }

    print(f"\n  Summary: avg={avg_all:.3f}s, RTF={avg_rtf:.2f}x, RAM=+{result['ram_after_warm_mb']}MB, GPU=+{result['gpu_after_warm_mb']}MB")
    return result


# ── HTML Report ─────────────────────────────────────────
def generate_html(results: list[dict]):
    configs = [r for r in results if not r.get("skipped")]
    env_info = f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}"

    # Performance table
    rows_perf = ""
    for r in configs:
        rows_perf += f"""<tr>
  <td><strong>{r['name']}</strong></td><td>{r['device']}</td><td>{r['bert_dtype']}</td>
  <td>{r['avg_inference_sec']}s</td><td><strong>{r['avg_rtf']}x</strong></td>
  <td>{r['ram_after_warm_mb']}MB</td><td>{r['gpu_after_warm_mb']}MB</td>
</tr>"""

    # Speedup vs fp32 on same device
    baseline_gpu = next((r for r in configs if r["name"] == "fp32_gpu"), None)
    baseline_cpu = next((r for r in configs if r["name"] == "fp32_cpu"), None)

    speedup_rows = ""
    for r in configs:
        baseline = baseline_gpu if "gpu" in r["name"] else baseline_cpu
        if baseline and r["name"] != baseline["name"]:
            speedup = baseline["avg_inference_sec"] / r["avg_inference_sec"] if r["avg_inference_sec"] > 0 else 0
            mem_diff = r["ram_after_warm_mb"] - baseline["ram_after_warm_mb"]
            gpu_diff = r["gpu_after_warm_mb"] - baseline["gpu_after_warm_mb"]
            speedup_rows += f"<tr><td>{r['name']}</td><td>vs {baseline['name']}</td><td><strong>{speedup:.2f}x</strong></td><td>{mem_diff:+d}MB</td><td>{gpu_diff:+d}MB</td></tr>"

    # Per-text detail table
    detail_rows = ""
    for i, (label, text) in enumerate(TEST_TEXTS):
        for r in configs:
            pt = r["per_text"][i]
            detail_rows += f"<tr><td>{r['name']}</td><td>{label}</td><td>{pt['avg_time']}s ±{pt['std_time']}</td><td>{pt['audio_duration']}s</td><td><strong>{pt['rtf']}x</strong></td></tr>"

    # Audio comparison
    audio_html = ""
    for label, text in TEST_TEXTS:
        players = ""
        for r in configs:
            pt = next(p for p in r["per_text"] if p["label"] == label)
            fname = f"{r['name']}_{label}.wav"
            players += f"""<div class="player">
  <div class="cfg">{r['name']}</div>
  <div class="meta">{pt['avg_time']}s | RTF {pt['rtf']}x</div>
  <audio controls preload="none"><source src="{fname}" type="audio/wav"></audio>
</div>\n"""
        audio_html += f"""<div class="sample">
  <div class="label">{label}</div>
  <div class="text">{text}</div>
  <div class="players">{players}</div>
</div>\n"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Quantization Benchmark</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 2rem; max-width: 1040px; margin: 0 auto; line-height: 1.6; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
  .sub {{ color: #888; font-size: 0.85rem; margin-bottom: 0.3rem; }}
  .env {{ color: #666; font-size: 0.8rem; margin-bottom: 2rem; font-family: monospace; }}
  h2 {{ font-size: 1.1rem; color: #7eb8ff; margin: 2.5rem 0 0.8rem; border-bottom: 1px solid #333; padding-bottom: 0.4rem; }}
  h3 {{ font-size: 0.95rem; color: #aaa; margin: 1.5rem 0 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0; font-size: 0.82rem; }}
  th, td {{ padding: 0.45rem 0.6rem; text-align: left; border: 1px solid #2a2a2a; }}
  th {{ background: #1a1a2e; color: #7eb8ff; font-weight: 600; white-space: nowrap; }}
  td {{ background: #1a1a1a; }}
  .sample {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 1rem; margin: 0.8rem 0; }}
  .sample .label {{ font-size: 0.75rem; color: #7eb8ff; text-transform: uppercase; letter-spacing: 0.05em; }}
  .sample .text {{ font-size: 0.9rem; margin: 0.3rem 0 0.7rem; color: #fff; }}
  .players {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.5rem; }}
  .player {{ background: #151520; border-radius: 6px; padding: 0.5rem 0.6rem; }}
  .player .cfg {{ font-size: 0.8rem; color: #7eb8ff; }}
  .player .meta {{ font-size: 0.75rem; color: #888; margin-bottom: 0.3rem; }}
  audio {{ width: 100%; height: 32px; }}
  .note {{ background: #1a1a2e; border-left: 3px solid #7eb8ff; padding: 0.7rem 1rem; margin: 1rem 0; font-size: 0.83rem; border-radius: 0 4px 4px 0; }}
  .warn {{ background: #2e2a1a; border-left: 3px solid #c9c96e; padding: 0.7rem 1rem; margin: 1rem 0; font-size: 0.83rem; border-radius: 0 4px 4px 0; }}
</style>
</head>
<body>

<h1>Quantization Benchmark</h1>
<p class="sub">Style-Bert-VITS2 (Elaina) — FP32 / FP16 / INT8</p>
<p class="env">{env_info} | Repeats: {REPEATS} | Texts: {len(TEST_TEXTS)} | Synth: always FP32</p>

<div class="warn">Synthesizer(VITS)는 FP16 불가 — Flow 레이어의 rational_quadratic_spline이 half precision에서 수치 불안정. BERT만 dtype 변경.</div>

<h2>Summary</h2>
<table>
<tr><th>Config</th><th>Device</th><th>BERT dtype</th><th>Avg Inference</th><th>RTF</th><th>RAM</th><th>GPU VRAM</th></tr>
{rows_perf}
</table>

<div class="note">
RTF (Real-Time Factor) = 음성 길이 / 추론 시간. RTF &gt; 1이면 실시간보다 빠름.<br>
예: RTF 3.0x = 1초 음성을 0.33초에 생성.
</div>

<h2>Speedup vs FP32 Baseline</h2>
<table>
<tr><th>Config</th><th>Baseline</th><th>Speed</th><th>RAM diff</th><th>GPU diff</th></tr>
{speedup_rows}
</table>

<h2>Per-Text Detail</h2>
<table>
<tr><th>Config</th><th>Text</th><th>Time (±std)</th><th>Audio</th><th>RTF</th></tr>
{detail_rows}
</table>

<h2>Audio Comparison</h2>
<div class="note">동일 텍스트, 동일 파라미터. dtype에 의한 음질 차이 확인.</div>
{audio_html}

</body>
</html>"""

    (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"\nHTML: {OUTPUT_DIR / 'index.html'}")


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Quantization Benchmark | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Texts: {len(TEST_TEXTS)} | Repeats: {REPEATS}")

    all_results = []
    for config in CONFIGS:
        result = benchmark_config(config)
        all_results.append(result)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    generate_html(all_results)
    print("\nDone!")
