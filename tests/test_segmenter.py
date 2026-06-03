from agent.segmenter import SentenceSegmenter


def _feed_all(seg, tokens):
    out = []
    for t in tokens:
        out.extend(seg.feed(t))
    tail = seg.flush()
    if tail:
        out.append(tail)
    return out


def test_emits_on_sentence_boundary():
    seg = SentenceSegmenter()
    out = _feed_all(seg, ["Hallo", " Welt", ".", " Wie", " geht", "'s", "?"])
    assert out == ["Hallo Welt.", "Wie geht's?"]


def test_flush_returns_trailing_partial():
    seg = SentenceSegmenter()
    out = _feed_all(seg, ["Kein", " Punkt", " hier"])
    assert out == ["Kein Punkt hier"]


def test_german_abbreviation_does_not_split():
    seg = SentenceSegmenter()
    # "z.B." must not split into three fragments.
    out = _feed_all(seg, ["Das", " ist", " z.B.", " ein", " Test", "."])
    assert out == ["Das ist z.B. ein Test."]


def test_empty_stream_flushes_none():
    seg = SentenceSegmenter()
    assert seg.flush() is None
