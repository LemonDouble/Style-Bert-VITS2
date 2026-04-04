"""Benchmark: PyTorch CPU vs ONNX FP32 vs ONNX INT8.

Compares inference speed and audio quality for Style-Bert-VITS2.

Usage:
    python -m dev_tools.benchmark_onnx

Run from the project root directory (Style-Bert-VITS2/).
"""

import gc
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

SAMPLE_RATE = 44100
REPEATS = 5
MODEL_DIR = str(_PROJECT_ROOT / "model_assets" / "Elaina")
ONNX_DIR = _PROJECT_ROOT / "onnx_models"
OUTPUT_DIR = _PROJECT_ROOT / "dev_tools" / "benchmark_results" / "onnx"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_TEXTS = [
    ("short", "はい、わかりま��た。"),
    ("med", "先生、大丈夫ですか？今日はちょっと疲れてるみたいですね。"),
    ("long", "本日は天気が良いので、少し散歩に出かけようと思います。公園のベンチで本を読むのもいいかもしれませんね。"),
]


def mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def benchmark_pytorch_cpu():
    """Baseline: PyTorch FP32 CPU."""
    import torch
    from style_bert_vits2 import TTS, Lang

    print("\n" + "=" * 60)
    print("  PyTorch FP32 CPU")
    print("=" * 60)

    gc.collect()
    mem_base = mem_mb()

    t0 = time.time()
    tts = TTS(device="cpu")
    speaker = tts.load("elaina", MODEL_DIR)
    speaker._ensure_model()
    load_time = time.time() - t0
    print(f"  Load: {load_time:.1f}s | RAM: +{mem_mb() - mem_base:.0f}MB")

    # Warmup
    speaker.generate("テスト", lang=Lang.JA, style="Neutral")
    speaker.generate("テスト", lang=Lang.JA, style="Neutral")

    results = []
    for label, text in TEST_TEXTS:
        times = []
        for r in range(REPEATS):
            t0 = time.time()
            audio = speaker.generate(text, lang=Lang.JA, style="Neutral")
            times.append(time.time() - t0)
            if r == 0:
                audio.save(str(OUTPUT_DIR / f"pytorch_{label}.wav"))

        avg = np.mean(times)
        std = np.std(times)
        dur = len(audio.data) / SAMPLE_RATE
        rtf = dur / avg
        results.append({"label": label, "avg": avg, "std": std, "rtf": rtf, "dur": dur})
        print(f"  {label:6s}: {avg:.3f}s ±{std:.3f}s | RTF={rtf:.1f}x | dur={dur:.2f}s")

    mem_peak = mem_mb()
    print(f"  Peak RAM: +{mem_peak - mem_base:.0f}MB")

    # Cleanup
    del tts, speaker
    from style_bert_vits2.nlp import bert_models
    bert_models.unload_model()
    bert_models.unload_tokenizer()
    gc.collect()

    return {"name": "pytorch_fp32_cpu", "load_time": load_time, "ram_mb": mem_peak - mem_base, "texts": results}


def benchmark_onnx(bert_path: str, synth_path: str, config_name: str):
    """Benchmark ONNX inference."""
    import onnxruntime as ort

    from style_bert_vits2.constants import Languages
    from style_bert_vits2.models.hyper_parameters import HyperParameters
    from style_bert_vits2.models.infer_onnx import infer_onnx
    from style_bert_vits2.nlp import bert_models

    print("\n" + "=" * 60)
    print(f"  ONNX {config_name}")
    print("=" * 60)

    gc.collect()
    mem_base = mem_mb()

    t0 = time.time()
    # Load tokenizer (must use HF repo for correct BertJapaneseTokenizer)
    from style_bert_vits2.constants import BERT_JP_REPO
    bert_models.load_tokenizer(pretrained_model_name_or_path=BERT_JP_REPO)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = os.cpu_count()

    bert_session = ort.InferenceSession(
        bert_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    synth_session = ort.InferenceSession(
        synth_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    # Load config + style vectors
    hps = HyperParameters.load_from_json(Path(MODEL_DIR) / "config.json")
    style_vectors = np.load(Path(MODEL_DIR) / "style_vectors.npy")
    style2id = hps.data.style2id if hasattr(hps.data, "style2id") else {"Neutral": 0}
    mean_vec = style_vectors[0]
    neutral_vec = style_vectors[style2id["Neutral"]]
    style_vec = mean_vec + (neutral_vec - mean_vec) * 1.0

    load_time = time.time() - t0
    print(f"  Load: {load_time:.1f}s | RAM: +{mem_mb() - mem_base:.0f}MB")

    # Warmup
    for _ in range(2):
        infer_onnx(
            text="テスト",
            style_vec=style_vec,
            sdp_ratio=0.2,
            noise_scale=0.6,
            noise_scale_w=0.8,
            length_scale=1.0,
            sid=0,
            language=Languages.JP,
            hps=hps,
            bert_session=bert_session,
            synth_session=synth_session,
        )

    results = []
    for label, text in TEST_TEXTS:
        times = []
        for r in range(REPEATS):
            t0 = time.time()
            audio = infer_onnx(
                text=text,
                style_vec=style_vec,
                sdp_ratio=0.2,
                noise_scale=0.6,
                noise_scale_w=0.8,
                length_scale=1.0,
                sid=0,
                language=Languages.JP,
                hps=hps,
                bert_session=bert_session,
                synth_session=synth_session,
            )
            times.append(time.time() - t0)
            if r == 0:
                # Save as wav
                audio_norm = audio / np.abs(audio).max()
                audio_16 = (audio_norm * 32767).astype(np.int16)
                import wave
                wav_path = str(OUTPUT_DIR / f"{config_name}_{label}.wav")
                with wave.open(wav_path, "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_16.tobytes())

        dur = len(audio) / SAMPLE_RATE
        avg = np.mean(times)
        std = np.std(times)
        rtf = dur / avg
        results.append({"label": label, "avg": avg, "std": std, "rtf": rtf, "dur": dur})
        print(f"  {label:6s}: {avg:.3f}s ±{std:.3f}s | RTF={rtf:.1f}x | dur={dur:.2f}s")

    mem_peak = mem_mb()
    print(f"  Peak RAM: +{mem_peak - mem_base:.0f}MB")

    # Cleanup
    del bert_session, synth_session
    bert_models.unload_tokenizer()
    gc.collect()

    return {"name": config_name, "load_time": load_time, "ram_mb": mem_peak - mem_base, "texts": results}


def print_summary(all_results):
    """Print comparison table."""
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    # Header
    labels = [t["label"] for t in all_results[0]["texts"]]
    header = f"{'Config':<22s} {'RAM':>6s} {'Load':>6s}"
    for l in labels:
        header += f" | {l:>8s}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        line = f"{r['name']:<22s} {r['ram_mb']:>5.0f}M {r['load_time']:>5.1f}s"
        for t in r["texts"]:
            line += f" | {t['avg']:>7.3f}s"
        print(line)

    # Speedup vs baseline
    if len(all_results) > 1:
        base = all_results[0]
        print("\nSpeedup vs PyTorch:")
        for r in all_results[1:]:
            speedups = []
            for bt, rt in zip(base["texts"], r["texts"]):
                speedups.append(bt["avg"] / rt["avg"])
            avg_speedup = np.mean(speedups)
            print(f"  {r['name']}: {avg_speedup:.2f}x avg ({', '.join(f'{s:.2f}x' for s in speedups)})")


def main():
    import json

    all_results = []

    # 1. PyTorch FP32 CPU
    all_results.append(benchmark_pytorch_cpu())

    # 2. ONNX FP32
    all_results.append(benchmark_onnx(
        str(ONNX_DIR / "bert.onnx"),
        str(ONNX_DIR / "synthesizer.onnx"),
        "onnx_fp32",
    ))

    # 3. ONNX FP32 BERT + INT8 Synth
    all_results.append(benchmark_onnx(
        str(ONNX_DIR / "bert.onnx"),
        str(ONNX_DIR / "synthesizer_q8.onnx"),
        "onnx_synth_q8",
    ))

    # 4. ONNX INT8 BERT + INT8 Synth (if BERT Q8 works)
    bert_q8_path = ONNX_DIR / "bert_q8.onnx"
    if bert_q8_path.exists():
        try:
            all_results.append(benchmark_onnx(
                str(bert_q8_path),
                str(ONNX_DIR / "synthesizer_q8.onnx"),
                "onnx_all_q8",
            ))
        except Exception as e:
            print(f"  BERT Q8 failed: {e}")
            print("  Skipping onnx_all_q8")

    print_summary(all_results)

    # Save results
    with open(OUTPUT_DIR / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
