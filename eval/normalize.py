"""Mətn normallaşdırma.

WER/CER hesablamadan əvvəl həm düzgün mətnə, həm model çıxışına tətbiq olunur,
ki müqayisə yalnız sözlərə baxsın, böyük/kiçik hərf və durğu işarələrinə yox.
"""
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # Python-un lower() funksiyası "İ"-ni "i" + birləşən nöqtəyə (U+0307) çevirir.
    # Bu nöqtə "söz hərfi" sayılmır və sonrakı addımda boşluğa çevrilərdi
    # ("i̇nsanlar" -> "i nsanlar"), ona görə onu əvvəlcədən silirik.
    text = text.replace("\u0307", "")
    # Azərbaycan dilinin böyük/kiçik hərf qaydası: I -> ı, İ -> i
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = re.sub(r"[^\w\s]", " ", text)  # durğu işarələrini sil
    return re.sub(r"\s+", " ", text).strip()