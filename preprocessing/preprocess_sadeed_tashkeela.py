"""
Preprocessing pipeline for Sadeed_Tashkeela, implementing the Method > "1.
Data Preprocessing & Alignment" steps (a-f), plus the extra requirements the
PDF's own "Comment & Assessment" section (p.7) adds on top of that list
(g-j below), in order. Each step is checked against what this dataset
actually needs (see notebooks/sadeed_tashkeela_eda.ipynb) and skipped with a
comment where the corpus already doesn't need it, rather than run as a
no-op.
"""

import csv
import hashlib
import os
import random
import re
import unicodedata

import pandas as pd
from pyarabic import araby

DATASET_NAME = "Sadeed_Tashkeela"
# preprocessing -> repo root
DATASET_DIR = os.path.join(os.path.dirname(__file__), "../datasets", DATASET_NAME)
TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH = os.path.join(DATASET_DIR, "test.csv")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")

CHUNKSIZE = 50_000

TASHKEEL = set(araby.TASHKEEL)


def is_arabic_letter(ch: str) -> bool:
    return "؀" <= ch <= "ۿ" and ch.isalpha() and ch not in TASHKEEL


# ---------------------------------------------------------------------------
# Step (g): Unicode normalization (NFC).
# ---------------------------------------------------------------------------
# Needed - and first, before anything else touches the text. A full-corpus
# scan found 92% of rows are not NFC-normalized: stacked diacritics (e.g.
# shadda+fatha on one letter) are stored in inconsistent codepoint order.
# Two rows can render identically and still produce two different
# character-level diacritic labels downstream, silently fragmenting the
# label vocabulary the Method section is built around.


# ---------------------------------------------------------------------------
# Step (a): Removal of HTML/XML tags, digits, and invalid Unicodes.
# ---------------------------------------------------------------------------
# - HTML/XML tags: not needed. A full-corpus character scan found zero
#   '<'/'>' characters in train.csv - the source is already free of markup.
# - Digits: needed. Digits 0-9 occur ~1.9M times across the corpus. Stripping
#   a footnote/verse reference like "(165)" leaves an empty "()" behind, so
#   that's cleaned up right after, before whitespace collapsing.
DIGIT_RE = re.compile(r"\d+")
EMPTY_PARENS_RE = re.compile(r"\(\s*\)")
WHITESPACE_RE = re.compile(r"\s+")

# - Invalid Unicode: needed, but rare. A full-corpus scan found a handful of
#   bidi/zero-width control characters (e.g. U+200D ZERO WIDTH JOINER, 4
#   occurrences) that would otherwise leak into the character vocabulary.
INVALID_UNICODE_RE = re.compile(r"[\u061C\u200B-\u200F\u202A-\u202E\u2066-\u2069]")

# - Dash-run footnote separators: not in the PDF's list, but needed. Found in
#   0.03% of segments (mostly hadith-chain books) - a run of 3+ literal
#   dashes used as a divider line in the original digitized book, with no
#   diacritic content of its own.
DASH_RUN_RE = re.compile(r"-{3,}")


# ---------------------------------------------------------------------------
# Step (b): Fixing common diacritic mistakes (e.g. tanwin before alef),
# removing tatweel, and fixing space around punctuation.
# ---------------------------------------------------------------------------
# - Tanwin before alef: not needed. Checked ALEF+FATHATAN vs. FATHATAN+ALEF
#   ordering across a 100k-row sample: 133k correctly-ordered occurrences
#   (mark on the consonant, alef after), 0 incorrectly-ordered ones.
# - Tatweel: not needed. Zero tatweel characters (U+0640) found in a
#   300k-row scan.
# - Space before punctuation: needed. ~9% of sampled rows have a stray space
#   before punctuation (e.g. "word ."); pyarabic's fix_spaces collapses it.


def clean_text(text: str) -> str:
    """Steps (g), (a) and (b): character-level cleanup of one raw `output`
    row, in that order - normalize first, so every later regex/word-count
    operates on a single canonical encoding of the text."""
    text = unicodedata.normalize("NFC", text)
    text = DIGIT_RE.sub("", text)
    # Nested refs like "( (1) )" need repeated passes: removing the inner
    # "(1)" first leaves an outer "( )" that a single pass would miss.
    prev = None
    while prev != text:
        prev = text
        text = EMPTY_PARENS_RE.sub("", text)
    text = INVALID_UNICODE_RE.sub("", text)
    text = DASH_RUN_RE.sub("", text)
    # Collapse the gaps left behind by removing digits/invalid-Unicode chars.
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = araby.fix_spaces(text)
    return text


# ---------------------------------------------------------------------------
# Step (c): Splitting based on context (sentence enders and/or clause
# separators).
# Step (e): Segmenting each sample into a word-count range.
# Step (f): Context-based splitting for long sentences.
# ---------------------------------------------------------------------------
# All three needed together: rows in this dataset are pre-chunked but not
# sentence-level - 69% of a 200k-row sample contain an embedded newline and
# 29% contain more than one sentence-ending mark, and the median row is
# already 52 words, above the range below. Range and separators match this
# repo's existing Shamela pipeline convention
# (tools/dataset-prep/dataset-preparer/src/utils/prepare_dataset/), so both
# datasets get segmented the same way.
MIN_WORDS = 5
MAX_WORDS = 40

SENTENCE_ENDERS_RE = re.compile(r"[.!?؟\n\r]")
CLAUSE_SEPARATORS_RE = re.compile(r"[,،:;؛()\[\]{}«»]")


def segment(text: str) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_ENDERS_RE.split(text) if s.strip()]

    segments = []
    for sentence in sentences:
        if MIN_WORDS <= len(sentence.split()) <= MAX_WORDS:
            segments.append(sentence)
        else:
            fragments = [f.strip() for f in CLAUSE_SEPARATORS_RE.split(sentence) if f.strip()]
            segments.extend(pack_fragments(fragments))

    return segments


def pack_fragments(fragments: list[str]) -> list[str]:
    """(f) Greedily merge adjacent clause fragments (in order) into
    MAX_WORDS-sized chunks, instead of dropping long sentences outright."""
    packed = []
    buffer = ""

    for frag in fragments:
        candidate = f"{buffer} {frag}".strip() if buffer else frag
        if len(candidate.split()) <= MAX_WORDS:
            buffer = candidate
        else:
            if buffer:
                packed.append(buffer)
            buffer = frag

    if buffer:
        if packed and len(buffer.split()) < MIN_WORDS:
            packed[-1] = f"{packed[-1]} {buffer}"
        else:
            packed.append(buffer)

    return packed


# ---------------------------------------------------------------------------
# Step (d): Ensuring each sample has a minimum average of diacritics per
# word.
# ---------------------------------------------------------------------------
# Needed: without this, a segment with no (or almost no) diacritics - e.g. a
# stray heading fragment - would pass straight into training. Threshold
# matches this repo's existing convention for the same check on Shamela.
MIN_AVG_DIACRITICS_PER_WORD = 2.2


def has_min_diacritics(segment_text: str) -> bool:
    words = segment_text.split()
    if not words:
        return False
    n_diacritics = sum(ch in TASHKEEL for ch in segment_text)
    return (n_diacritics / len(words)) >= MIN_AVG_DIACRITICS_PER_WORD


# ---------------------------------------------------------------------------
# Step (j): Partial / inconsistent diacritization at the word level
# (Comment & Assessment, PDF p.7) - a separate check from step (d)'s
# sentence-average, so each is independently measurable.
# ---------------------------------------------------------------------------
# Needed, but narrowly. A naive whitespace word count treats stray
# punctuation tokens ('-', '/', '()') as "words", which overstates the
# problem (18.8% of segments). Counting only tokens that actually contain an
# Arabic letter drops that to 2.4%, and most of those are themselves an
# artifact of step (a)'s digit removal: stripping a reference like "المادة
# 165 و 166" strands an undiacritized "و" behind. A ratio-based threshold -
# drop only when *most* of a segment's real words are bare - catches
# genuinely low-quality segments without discarding one over a single
# stranded artifact word.
BARE_WORD_RATIO_THRESHOLD = 0.3


def has_acceptable_bare_word_ratio(segment_text: str) -> bool:
    real_words = [w for w in segment_text.split() if any(is_arabic_letter(ch) for ch in w)]
    if not real_words:
        return False
    n_bare = sum(1 for w in real_words if not any(ch in TASHKEEL for ch in w))
    return (n_bare / len(real_words)) <= BARE_WORD_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Step (h): Deduplication.
# ---------------------------------------------------------------------------
# Needed, applied after splitting (not before): a full-corpus row-level scan
# found 0% duplicates, but splitting into short clauses/sentences produces
# genuine duplicates - the same short phrase recurring across different
# books - at ~2.9% of segments. Tracked separately per output file (train,
# val, test), since duplicate detection is about redundancy within a split,
# not about deduplicating across splits.


# ---------------------------------------------------------------------------
# Step (i): Splitting validation off by book, before chunking.
# ---------------------------------------------------------------------------
# Needed: the plan's Training & Validation step calls for a held-out signal
# for early stopping/checkpointing, but Sadeed_Tashkeela only ships
# train/test. Carving out whole books (not sentences) keeps a book's
# passages from appearing in both train and val at once. train/test
# themselves already don't leak - EDA found 0 book overlap between them.
VAL_RATIO = 0.1
VAL_SEED = 42


def get_book_names(path: str) -> set[str]:
    names = set()
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, dtype=str, keep_default_na=False, usecols=["filename"]):
        names.update(chunk["filename"])
    return names


def split_books_for_validation(book_names: set[str], val_ratio: float, seed: int) -> set[str]:
    shuffled = sorted(book_names)  # sort first so shuffling doesn't depend on set iteration order
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_ratio)
    return set(shuffled[:n_val])


def process_split(src_path: str, out_paths: dict[str, str], val_books: set[str] | None = None) -> dict:
    """
    Stream `src_path`, apply steps (g), (a), (b), (c), (e), (f), (d), (j)
    and (h) in order, and write surviving segments to one CSV per target in
    `out_paths` (e.g. {"train": ..., "val": ...} or just {"test": ...}).
    """
    writers = {}
    files = []
    try:
        for name, path in out_paths.items():
            f = open(path, "w", newline="", encoding="utf-8")
            files.append(f)
            writer = csv.writer(f)
            writer.writerow(["filename", "output"])
            writers[name] = writer

        stats = {
            "n_rows_in": 0,
            "n_segments_out": 0,
            "n_dropped_low_diacritics": 0,
            "n_dropped_bare_words": 0,
            "n_dropped_dup": 0,
            "by_split": {name: 0 for name in out_paths},
        }
        seen_hashes = {name: set() for name in out_paths}

        for chunk in pd.read_csv(src_path, chunksize=CHUNKSIZE, dtype=str, keep_default_na=False):
            for row in chunk.itertuples(index=False):
                stats["n_rows_in"] += 1

                target = "val" if val_books and row.filename in val_books else next(iter(out_paths))

                cleaned = clean_text(row.output)
                for seg in segment(cleaned):
                    if not has_min_diacritics(seg):
                        stats["n_dropped_low_diacritics"] += 1
                        continue
                    if not has_acceptable_bare_word_ratio(seg):
                        stats["n_dropped_bare_words"] += 1
                        continue

                    digest = hashlib.md5(seg.encode("utf-8")).hexdigest()
                    if digest in seen_hashes[target]:
                        stats["n_dropped_dup"] += 1
                        continue
                    seen_hashes[target].add(digest)

                    writers[target].writerow([row.filename, seg])
                    stats["n_segments_out"] += 1
                    stats["by_split"][target] += 1
    finally:
        for f in files:
            f.close()

    return stats


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("[1/2] Processing train.csv (carving out a val split by book) ...")
    train_books = get_book_names(TRAIN_PATH)
    val_books = split_books_for_validation(train_books, VAL_RATIO, VAL_SEED)
    print(f"  {len(train_books)} books total, {len(val_books)} held out for val")

    train_stats = process_split(
        TRAIN_PATH,
        {
            "train": os.path.join(PROCESSED_DIR, "train.csv"),
            "val": os.path.join(PROCESSED_DIR, "val.csv"),
        },
        val_books=val_books,
    )
    print(f"  {train_stats}")

    print("[2/2] Processing test.csv ...")
    test_stats = process_split(TEST_PATH, {"test": os.path.join(PROCESSED_DIR, "test.csv")})
    print(f"  {test_stats}")

    print("Done. Wrote processed dataset to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
