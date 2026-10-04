"""WER/CER hesablama."""
import jiwer

from eval.normalize import normalize


def _clean_pairs(refs, hyps):
    """Normallaşdırır və düzgün mətni boş qalan cütləri atır (jiwer boş REF qəbul etmir)."""
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    return [(r, h) for r, h in pairs if r]


def score(refs, hyps):
    """Korpus səviyyəsində WER və CER (sözlərin sayına görə çəkili)."""
    pairs = _clean_pairs(refs, hyps)
    if not pairs:
        return {"wer": float("nan"), "cer": float("nan"), "n": 0}
    r, h = zip(*pairs)
    return {"wer": jiwer.wer(list(r), list(h)), "cer": jiwer.cer(list(r), list(h)), "n": len(pairs)}


def per_sample(refs, hyps):
    """Hər cümlə üçün (WER, normallaşmış REF, normallaşmış HYP) siyahısı."""
    return [(jiwer.wer(r, h), r, h) for r, h in _clean_pairs(refs, hyps)]