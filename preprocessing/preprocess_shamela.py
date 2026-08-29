"""
Preprocessing pipeline for Shamela, implementing the same Method > "1. Data
Preprocessing & Alignment" steps (a-j) as preprocessing/preprocess_sadeed_tashkeela.py,
in order - but tuned against notebooks/shamela_eda.ipynb's actual findings
rather than assuming Sadeed_Tashkeela's numbers carry over. Output is
unified to Sadeed_Tashkeela's shape: one train/val/test.csv (not one file
per book) with just `filename` + `output` columns.
"""

import csv
import hashlib
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from pyarabic import araby

DATASET_NAME = "shamela"
# preprocessing -> repo root
SHAMELA_DIR = os.path.join(os.path.dirname(__file__), "../datasets", DATASET_NAME)
PROCESSED_DIR = os.path.join(SHAMELA_DIR, "processed")

CHUNKSIZE = 50_000

TASHKEEL = set(araby.TASHKEEL)


def is_arabic_letter(ch: str) -> bool:
    return "؀" <= ch <= "ۿ" and ch.isalpha() and ch not in TASHKEEL


# ---------------------------------------------------------------------------
# Step (g): Unicode normalization (NFC).
# ---------------------------------------------------------------------------
# Needed - and first. The EDA found 20% of rows are not NFC-normalized
# (lower than Sadeed_Tashkeela's 92%, but still a fifth of the corpus).


# ---------------------------------------------------------------------------
# Step (a): Removal of HTML/XML tags, digits, and invalid Unicodes.
# ---------------------------------------------------------------------------
# None of these are needed here. Unlike Sadeed_Tashkeela (a raw external
# dataset), Shamela's CSVs already went through this repo's own
# `dataset-preparer` during `scripts/prepare_shamela.py`'s build step, whose
# `preprocess.py` already ran `remove_tags`, `remove_digits` and
# `remove_bidi_controls` on every row. The EDA confirms all three are at 0
# across the full 17.67M-row corpus: 0 HTML-tag rows, 0 rows containing a
# digit, 0 rows matching a bidi/zero-width control character.


# ---------------------------------------------------------------------------
# Step (b): Fixing common diacritic mistakes (e.g. tanwin before alef),
# removing tatweel, and fixing space around punctuation.
# ---------------------------------------------------------------------------
# - Tanwin before alef: mostly already correct (`dataset-preparer` also ran
#   pyarabic's `autocorrect`, which fixes this) - 3,229,556 correctly-ordered
#   occurrences vs. only 160 incorrectly-ordered ones. Non-zero, so
#   `autocorrect` is re-run here too; it's idempotent and this cleans up the
#   small residue.
# - Tatweel: needed, unlike Sadeed_Tashkeela. 153,935 rows (0.87%) contain
#   tatweel ('ـ') - `dataset-preparer`'s cleanup didn't strip it.
# - Space before punctuation: not needed. `dataset-preparer` already ran
#   pyarabic's `fix_spaces`; the EDA found 0 rows with a stray space before
#   punctuation across the full corpus.


def clean_text(text: str) -> str:
    """Steps (g), (a) and (b): character-level cleanup of one raw
    `DIACRRITIC` row, in that order."""
    text = unicodedata.normalize("NFC", text)
    text = araby.autocorrect(text)
    text = araby.strip_tatweel(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step (c): Splitting based on context (sentence enders and/or clause
# separators).
# Step (e): Segmenting each sample into a word-count range.
# Step (f): Context-based splitting for long sentences.
# ---------------------------------------------------------------------------
# Needed, but lighter than for Sadeed_Tashkeela: the EDA shows Shamela rows
# are already close to sentence-level (median 9 words, p99 38 words) rather
# than Sadeed's median-52-word chunks, so most rows will pass through
# untouched and only the long tail gets split. Same range and separators as
# preprocess_sadeed_tashkeela.py, for consistency between the two datasets.
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
# Needed. Same 2.2 threshold as Sadeed_Tashkeela/Shamela's own dataset-
# preparer convention, re-applied here because our own re-splitting (c/e/f)
# can produce new short fragments that weren't in the original row.
MIN_AVG_DIACRITICS_PER_WORD = 2.2


def has_min_diacritics(segment_text: str) -> bool:
    words = segment_text.split()
    if not words:
        return False
    n_diacritics = sum(ch in TASHKEEL for ch in segment_text)
    return (n_diacritics / len(words)) >= MIN_AVG_DIACRITICS_PER_WORD


# ---------------------------------------------------------------------------
# Step (j): Partial / inconsistent diacritization at the word level.
# ---------------------------------------------------------------------------
# Needed. The EDA found 151,832 rows (0.86%) where a large share of real
# (letter-containing) words carry zero diacritics despite passing step (d)'s
# average - the "undiacritized intro + diacritized quote" pattern (e.g.
# "قال تعالى في سورة النحل ..." followed by a fully-marked Qur'an quote).
# Same 30% ratio threshold as Sadeed_Tashkeela.
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
# Needed, and more aggressively than for Sadeed_Tashkeela: the EDA found 24%
# of rows are exact duplicates of another row, and 81% of those duplicate
# groups (811,331 of 996,093) span more than one book - the same Qur'an
# verse or hadith quoted verbatim across many tafsir/fiqh books. A per-split
# hash set (like Sadeed_Tashkeela's) would only catch duplicates *within* a
# split, so the same sentence could still land in both train and val/test -
# a direct leak of the held-out signal. Dedup here is therefore GLOBAL: one
# hash set shared across train/val/test, so once a text is written to any
# split, every later occurrence anywhere is dropped.


# ---------------------------------------------------------------------------
# Step (i): Splitting train/val/test by book, before chunking.
# ---------------------------------------------------------------------------
# Needed. Unlike Sadeed_Tashkeela (shipped with its own train/test split),
# Shamela has no pre-existing split - all three have to be carved out of the
# same 2,695 books ourselves. Carving out whole books (not rows) keeps a
# book's passages from crossing splits. 80/10/10 is not specified anywhere
# in the plan; it's a plain default, applied the same way (seeded shuffle of
# book names) as Sadeed_Tashkeela's val split.
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42


def split_books(book_names: list[str], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    shuffled = sorted(book_names)  # sort first so shuffling doesn't depend on glob order
    random.Random(seed).shuffle(shuffled)

    n_val = round(len(shuffled) * val_ratio)
    n_test = round(len(shuffled) * test_ratio)

    assignment = {}
    for book in shuffled[:n_val]:
        assignment[book] = "val"
    for book in shuffled[n_val : n_val + n_test]:
        assignment[book] = "test"
    for book in shuffled[n_val + n_test :]:
        assignment[book] = "train"
    return assignment


def process_corpus(book_paths: list, out_paths: dict[str, str], book_split: dict[str, str]) -> dict:
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
        seen_hashes: set[str] = set()  # global across all splits - see step (h) above

        for path in book_paths:
            target = book_split[path.stem]

            for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, dtype=str, keep_default_na=False):
                for row in chunk.itertuples(index=False):
                    stats["n_rows_in"] += 1

                    cleaned = clean_text(row.DIACRRITIC)
                    for seg in segment(cleaned):
                        if not has_min_diacritics(seg):
                            stats["n_dropped_low_diacritics"] += 1
                            continue
                        if not has_acceptable_bare_word_ratio(seg):
                            stats["n_dropped_bare_words"] += 1
                            continue

                        digest = hashlib.md5(seg.encode("utf-8")).hexdigest()
                        if digest in seen_hashes:
                            stats["n_dropped_dup"] += 1
                            continue
                        seen_hashes.add(digest)

                        writers[target].writerow([row.FILENAME, seg])
                        stats["n_segments_out"] += 1
                        stats["by_split"][target] += 1
    finally:
        for f in files:
            f.close()

    return stats


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    book_paths = sorted(Path(SHAMELA_DIR).glob("*.csv"))
    book_split = split_books([p.stem for p in book_paths], VAL_RATIO, TEST_RATIO, SPLIT_SEED)
    n_by_split = Counter(book_split.values())
    print(f"{len(book_paths)} books total -> {dict(n_by_split)}")

    stats = process_corpus(
        book_paths,
        {
            "train": os.path.join(PROCESSED_DIR, "train.csv"),
            "val": os.path.join(PROCESSED_DIR, "val.csv"),
            "test": os.path.join(PROCESSED_DIR, "test.csv"),
        },
        book_split,
    )
    print(stats)
    print("Done. Wrote processed dataset to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
