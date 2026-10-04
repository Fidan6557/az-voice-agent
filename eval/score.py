"""Saxlanmış nəticələri hesablayır. Model işləmir, yalnız hesablama aparılır.

İstifadə:
    python -m eval.score results/whisper-large-v3_fleurs-az_n200_seed42.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from eval.metrics import per_sample, score


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_meta(path):
    meta_path = Path(path).with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return None


def report(rows, meta=None, worst=5):
    refs = [r["ref"] for r in rows]
    hyps = [r["hyp"] for r in rows]

    total = score(refs, hyps)
    print(f"Nümunə sayı: {total['n']}")
    print(f"WER: {total['wer']:.1%}")
    print(f"CER: {total['cer']:.1%}")

    if meta:
        audio = sum(r["audio_sec"] for r in rows)
        print(f"Audio: {audio:.0f} san, işləmə: {meta['elapsed_sec']:.0f} san, "
              f"RTF: {meta['elapsed_sec'] / audio:.2f} (batch={meta['batch_size']})")

    groups = defaultdict(list)
    for r in rows:
        groups[r.get("gender", "?")].append(r)
    print("\nCins üzrə:")
    for g, items in sorted(groups.items()):
        s = score([i["ref"] for i in items], [i["hyp"] for i in items])
        print(f"  {g}: n={s['n']}, WER={s['wer']:.1%}, CER={s['cer']:.1%}")

    print(f"\nƏn pis {worst} cümlə:")
    for wer, ref, hyp in sorted(per_sample(refs, hyps), reverse=True)[:worst]:
        print(f"  WER={wer:.0%}\n  REF: {ref}\n  HYP: {hyp}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("İstifadə: python -m eval.score <nəticə.jsonl>")
    report(load_rows(sys.argv[1]), load_meta(sys.argv[1]))