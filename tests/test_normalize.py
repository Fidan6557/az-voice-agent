from eval.normalize import normalize


def test_punctuation_and_case():
    assert normalize("Salam, dünya!") == "salam dünya"


def test_azerbaijani_dotted_and_dotless_i():
    assert normalize("İnsanlar") == "insanlar"
    assert normalize("IŞIQ") == "ışıq"


def test_combining_dot_above_is_removed():
    # FLEURS mətnlərində cümlə əvvəlindəki "İ" bu formada gəlir: i + U+0307
    assert normalize("i\u0307nsanlar") == "insanlar"
    assert normalize("i\u0307nternet həm kütləvi") == "internet həm kütləvi"


def test_whitespace_collapsed():
    assert normalize("  salam   dünya \n") == "salam dünya"