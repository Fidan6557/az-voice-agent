"""Whisper-i FLEURS-az test dəstində işlədir, xam çıxışları diskə yazır və hesablayır.

İstifadə (Colab-da, GPU ilə):
    python -m eval.run_eval --model openai/whisper-large-v3 --n 200 --seed 42

Nəticə results/ qovluğuna yazılır. Eyni parametrlərlə yenidən işlətsən, model
təkrar işləmir (keşdən oxunur), --force ilə məcbur edə bilərsən.
"""
import argparse
import json
import time
from pathlib import Path

from eval.score import load_meta, load_rows, report

RESULTS_DIR = Path("results")


def make_tag(model: str, n: int, seed: int) -> str:
    return f"{model.split('/')[-1]}_fleurs-az_n{n}_seed{seed}"


def load_fleurs(n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("google/fleurs", "az_az", split="test")
    # Təsadüfi seçmə: ilk N nümunə təmsilçi deyil (əvvəlki ölçmədə hamısı kişi səsi idi)
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def gender_of(ds, sample) -> str:
    g = sample["gender"]
    return ds.features["gender"].int2str(g) if isinstance(g, int) else str(g)


def transcribe_whisper(ds, model_name: str, batch_size: int):
    import torch
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        dtype=torch.float16,
        device="cuda",
        chunk_length_s=30,
    )

    audios, seconds = [], []
    for s in ds:
        array, sr = s["audio"]["array"], s["audio"]["sampling_rate"]
        seconds.append(len(array) / sr)  # pipeline lüğətləri dəyişir, ona görə müddəti əvvəl hesablayırıq
        audios.append({"raw": array, "sampling_rate": sr})

    start = time.time()
    outputs = asr(
        audios,
        batch_size=batch_size,
        generate_kwargs={"language": "azerbaijani", "task": "transcribe"},
    )
    return [o["text"] for o in outputs], seconds, time.time() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{make_tag(args.model, args.n, args.seed)}.jsonl"

    if path.exists() and not args.force:
        print(f"Keşdə tapıldı, model işləmir: {path}\n")
    else:
        ds = load_fleurs(args.n, args.seed)
        hyps, seconds, elapsed = transcribe_whisper(ds, args.model, args.batch_size)

        with open(path, "w", encoding="utf-8") as f:
            for s, hyp, sec in zip(ds, hyps, seconds):
                row = {
                    "id": s["id"],
                    "gender": gender_of(ds, s),
                    "audio_sec": sec,
                    "ref": s["transcription"],
                    "hyp": hyp,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        meta = {
            "model": args.model,
            "n": len(hyps),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "elapsed_sec": elapsed,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Yazıldı: {path}\n")

    report(load_rows(path), load_meta(path))


if __name__ == "__main__":
    main()