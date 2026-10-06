"""Böyük Azərbaycan nitq datasını yükləmədən (streaming) yoxlayır.

Məqsəd: fine-tune-dan əvvəl datanın formatına, keyfiyyətinə və test dəstimizlə
üst-üstə düşüb-düşmədiyinə baxmaq. Bütün dataseti yükləmir, yalnız təsadüfi nümunə götürür.

İstifadə (Colab-da):
    python -m training.asr.audit_data --n 300
    python -m training.asr.audit_data --n 300 --asr-check 100   # GPU lazımdır
"""
import argparse
import collections
import io
import json
import statistics
from pathlib import Path

from eval.normalize import normalize

DATASET = "LocalDoc/azerbaijani_asr"
AZ_LETTERS = set("abcçdeəfgğhxıijkqlmnoöprsştuüvyz")  # Azərbaycan əlifbası (32 hərf)
OUT_DIR = Path("audit")


def sample_rows(n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True)
    # Streaming-də shuffle təxminidir (bufer + fayl sırası), amma ilk N sətri götürməkdən yaxşıdır
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    rows = []
    for row in ds:
        rows.append(row)
        if len(rows) >= n:
            break
    return rows


def find_columns(row):
    """Sütun adlarını dataset kartından bilmirik, ona görə sətirdən tapırıq."""
    text_col = next(
        (c for c in ("transcription", "sentence", "text", "transcript") if isinstance(row.get(c), str)),
        None,
    )
    audio_col = next(
        (c for c, v in row.items() if isinstance(v, dict) and ("array" in v or "bytes" in v)),
        None,
    )
    if not text_col or not audio_col:
        raise SystemExit(f"Sütunlar tanınmadı. Mövcud sütunlar: {list(row)}")
    return text_col, audio_col


def get_audio(value):
    if value.get("array") is not None:
        return value["array"], value["sampling_rate"]
    import soundfile as sf

    array, sr = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, sr


def to_items(rows, text_col, audio_col):
    items = []
    for r in rows:
        array, sr = get_audio(r[audio_col])
        items.append({"id": r.get("id"), "sr": sr, "dur": len(array) / sr, "text": r[text_col], "array": array})
    return items


def report(items):
    print(f"\nNümunə sayı: {len(items)}")
    print("Nümunə mətnlər:")
    for i in items[:5]:
        print(f"  {i['dur']:.1f} san | {i['text']}")

    print("\nSampling rate:", dict(collections.Counter(i["sr"] for i in items)))
    durs = [i["dur"] for i in items]
    q = statistics.quantiles(durs, n=20)
    print(
        f"Müddət (san): min={min(durs):.1f} p5={q[0]:.1f} median={statistics.median(durs):.1f} "
        f"p95={q[-1]:.1f} max={max(durs):.1f} | orta={statistics.mean(durs):.1f}"
    )
    print(f"  1 saniyədən qısa: {sum(d < 1 for d in durs)} | 30 saniyədən uzun: {sum(d > 30 for d in durs)}")

    odd = collections.Counter()
    for i in items:
        for ch in normalize(i["text"]):
            if ch != " " and ch not in AZ_LETTERS and not ch.isdigit():
                odd[ch] += 1
    print("\nAzərbaycan əlifbasından kənar simvollar (normallaşmadan sonra):", odd.most_common(15) or "yoxdur")

    raw = [i["text"] for i in items]
    print(
        f"Rəqəm olan: {sum(any(c.isdigit() for c in t) for t in raw)} | "
        f"böyük hərf olan: {sum(any(c.isupper() for c in t) for t in raw)} | "
        f"durğu işarəsi olan: {sum(any(c in '.,?!;:' for c in t) for t in raw)}"
    )

    cps = [len(normalize(i["text"])) / i["dur"] for i in items if i["dur"] > 0]
    print(
        f"Hərf/saniyə: median={statistics.median(cps):.1f} | <4: {sum(c < 4 for c in cps)} | "
        f">25: {sum(c > 25 for c in cps)}  (şübhəli səs-mətn uyğunsuzluğu, evristikadır)"
    )

    norm = [normalize(t) for t in raw]
    print(f"Nümunə daxilində təkrar transkript: {len(norm) - len(set(norm))}")


def leakage_with_fleurs(items):
    """FLEURS test cümlələri ilə tam uyğunluq: eyni cümlədə öyrədib test etsək, nəticə saxta olar."""
    from datasets import load_dataset

    fleurs = load_dataset("google/fleurs", "az_az", split="test")
    test_texts = {normalize(t) for t in fleurs["transcription"]}
    hits = [i for i in items if normalize(i["text"]) in test_texts]
    print(f"\nFLEURS test cümləsi ilə eyni transkript: {len(hits)} / {len(items)} (yalnız tam uyğunluq yoxlanır)")


def save_samples(items, count=10):
    import soundfile as sf

    (OUT_DIR / "samples").mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "localdoc_sample.jsonl", "w", encoding="utf-8") as f:
        for i in items:
            row = {key: val for key, val in i.items() if key != "array"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for idx, i in enumerate(items[:count]):
        sf.write(OUT_DIR / "samples" / f"{idx:02d}.wav", i["array"], i["sr"])
    print(f"\nNümunələr yazıldı: {OUT_DIR}/ (ilk {count} səs .wav kimi, qulaq asmaq üçün)")


def asr_check(items, n, model="openai/whisper-large-v3"):
    """Hazır Whisper bu datanı necə tanıyır? Çox yüksək WER transkriptlərin səslə uyğun gəlmədiyini göstərə bilər."""
    from eval.metrics import per_sample, score
    from eval.run_eval import transcribe_whisper

    subset = items[:n]
    ds = [{"audio": {"array": i["array"], "sampling_rate": i["sr"]}} for i in subset]
    hyps, _, _ = transcribe_whisper(ds, model, batch_size=4, num_beams=1, max_new_tokens=200)
    refs = [i["text"] for i in subset]

    s = score(refs, hyps)
    print(f"\nWhisper large-v3 (zero-shot) bu data üzərində: WER={s['wer']:.1%} CER={s['cer']:.1%} (n={s['n']})")
    print("Ən pis 8 cümlə (transkript səhvi ola bilər, ya da model səhvi):")
    for wer, ref, hyp in sorted(per_sample(refs, hyps), reverse=True)[:8]:
        print(f"  WER={wer:.0%}\n  REF: {ref}\n  HYP: {hyp}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--asr-check", type=int, default=0, help="Whisper ilə ilk N nümunəni yoxla (GPU lazımdır)")
    args = p.parse_args()

    rows = sample_rows(args.n, args.seed)
    text_col, audio_col = find_columns(rows[0])
    print("Sütunlar:", list(rows[0]), "| mətn:", text_col, "| audio:", audio_col)

    items = to_items(rows, text_col, audio_col)
    report(items)
    leakage_with_fleurs(items)
    save_samples(items)
    if args.asr_check:
        asr_check(items, args.asr_check)


if __name__ == "__main__":
    main()  