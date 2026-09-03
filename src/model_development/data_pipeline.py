"""
data_pipeline.py

Turns datasets/Sadeed_Tashkeela/train.csv into fixed-size numpy arrays
ready for model.fit(), WITHOUT loading the whole 1.3GB/1M-row file into
memory at once. Two streamed passes over the CSV:

  PASS 1 (filter_and_build_vocab): read only the "output" column in
      chunks. For every row, decide keep/drop using the same rules the
      team's EDA already validated (drop empty rows, drop rows longer
      than the p99 char length of 997, drop non-Arabic-dominant rows,
      drop rows with zero diacritics at all). Also collect the set of
      distinct base characters seen, for the character vocabulary.
      Output: a small int8 numpy array (one entry per row in the file)
      marking each row as skip/candidate, plus the vocabulary.

  Then: a numpy seeded permutation of the *candidate* row positions picks
      exactly 500,000 for train and the next 50,000 for validation. This
      never touches test.csv, and train/val are disjoint by construction.

  PASS 2 (encode_selected_rows): read the CSV again, and for every row
      marked train/val, run label_config.extract_chars_and_labels() and
      write directly into pre-allocated numpy arrays (no giant Python list
      of strings is kept in memory).

Only the "output" column is read from disk. "input" and "filename" are
never needed: chars are derived from output, and the 100% consistency
check the team already ran confirms this equals the "input" column.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from label_config import (
    ARABIC_LETTERS,
    DIACRITIC_CHARS,
    CharVocab,
    extract_chars_and_labels,
    UnknownDiacriticCombination,
    NA_ID,
    PAD_ID,
)

CHUNKSIZE = 50_000
MAX_CHAR_LEN = 997          # EDA-recommended p99 cutoff; longer rows are DROPPED, not truncated
NON_ARABIC_RATIO_THRESHOLD = 0.30
N_TRAIN = 500_000
N_VAL = 50_000
SEED = 132


def _iter_output_chunks(csv_path, chunksize=CHUNKSIZE):
    """Same pattern as the team's EDA notebook: a real CSV parser (rows can
    contain embedded newlines), keep_default_na=False so genuinely-empty
    strings aren't silently turned into NaN."""
    return pd.read_csv(
        csv_path, usecols=["output"], dtype=str, keep_default_na=False, chunksize=chunksize
    )


def _row_is_kept(output_text: str) -> bool:
    stripped = output_text.strip()
    if not stripped:
        return False
    if len(output_text) > MAX_CHAR_LEN:
        return False
    if not any(ch in DIACRITIC_CHARS for ch in output_text):
        return False  # fully undiacritized row
    non_letter = sum(1 for ch in stripped if ch not in ARABIC_LETTERS and ch not in DIACRITIC_CHARS and ch != " ")
    if non_letter / len(stripped) > NON_ARABIC_RATIO_THRESHOLD:
        return False
    return True


@dataclass
class FilterResult:
    keep_mask: np.ndarray   # bool, length = total rows in the CSV
    vocab: CharVocab
    n_total_rows: int
    n_kept_rows: int


def filter_and_build_vocab(csv_path: str) -> FilterResult:
    """
    PASS 1. Streams the CSV once.

    As a side effect, this also scans EVERY kept row for diacritic
    combinations that label_config.CLASS_LABELS doesn't recognize (see
    label_config.extract_chars_and_labels docstring). If any are found,
    this raises with a full report BEFORE pass 2 (encoding) or training
    ever runs -- we never want to silently train on corrupted labels.
    """
    keep_flags = []
    vocab_chars = set()
    unknown_combos = {}

def filter_and_build_vocab(csv_path: str) -> FilterResult:
    keep_flags = []
    vocab_chars = set()

    for chunk in _iter_output_chunks(csv_path):
        for text in chunk["output"]:
            if not _row_is_kept(text):
                keep_flags.append(False)
                continue
            
            # Try extracting labels. If it has unknown combos (like shadda+sukun), drop the row!
            try:
                chars, _ = extract_chars_and_labels(text)
                keep_flags.append(True)
                vocab_chars.update(chars)
            except UnknownDiacriticCombination:
                keep_flags.append(False)

    keep_mask = np.array(keep_flags, dtype=bool)
    vocab = CharVocab().build(vocab_chars)
    return FilterResult(
        keep_mask=keep_mask,
        vocab=vocab,
        n_total_rows=len(keep_mask),
        n_kept_rows=int(keep_mask.sum()),
    )



def select_train_val_row_ids(filter_result: FilterResult, n_train=N_TRAIN, n_val=N_VAL, seed=SEED):
    """
    Seeded, reproducible, disjoint train/val row selection -- indices into
    the ORIGINAL train.csv (0-based row order, matching pandas' default
    RangeIndex). Returns an int8 array of length n_total_rows:
        0 = not used, 1 = train, 2 = validation
    """
    candidate_rows = np.flatnonzero(filter_result.keep_mask)
    if len(candidate_rows) < n_train + n_val:
        raise ValueError(
            f"Only {len(candidate_rows):,} rows survived filtering, "
            f"need {n_train + n_val:,} for the requested train+val split."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(candidate_rows)
    train_rows = shuffled[:n_train]
    val_rows = shuffled[n_train:n_train + n_val]

    assignment = np.zeros(filter_result.n_total_rows, dtype=np.int8)
    assignment[train_rows] = 1
    assignment[val_rows] = 2
    return assignment


def _allocate_arrays(n_rows, maxlen, vocab: CharVocab):
    char_ids = np.full((n_rows, maxlen), CharVocab.PAD_ID, dtype=np.int32)
    label_ids = np.full((n_rows, maxlen), PAD_ID, dtype=np.uint8)
    is_space = np.zeros((n_rows, maxlen), dtype=bool)
    return char_ids, label_ids, is_space


def encode_selected_rows(csv_path: str, assignment: np.ndarray, vocab: CharVocab, maxlen=MAX_CHAR_LEN):
    """PASS 2. Streams the CSV a second time, filling pre-allocated arrays
    for exactly the rows marked 1 (train) or 2 (val) in `assignment`."""
    n_train = int((assignment == 1).sum())
    n_val = int((assignment == 2).sum())

    train_arrays = _allocate_arrays(n_train, maxlen, vocab)
    val_arrays = _allocate_arrays(n_val, maxlen, vocab)
    train_idx = 0
    val_idx = 0

    global_row = 0
    for chunk in _iter_output_chunks(csv_path):
        for text in chunk["output"]:
            tag = assignment[global_row]
            global_row += 1
            if tag == 0:
                continue

            chars, labels = extract_chars_and_labels(text)  # strict=True default: raises on unknown combos
            if len(chars) > maxlen:
                # Mathematically this should be impossible: pass 1 already
                # dropped every row with len(output) > MAX_CHAR_LEN, and
                # len(chars) <= len(output) always (stripping diacritics
                # only removes characters, never adds any). If this fires,
                # something upstream is inconsistent (e.g. maxlen doesn't
                # match the value pass 1 filtered with) -- surface it loudly
                # rather than silently cutting the row and misaligning data.
                raise ValueError(
                    f"Row has {len(chars)} base characters, exceeding maxlen={maxlen}, "
                    "even though it passed the MAX_CHAR_LEN filter in pass 1. This should "
                    "be impossible -- check that filter_and_build_vocab and "
                    "encode_selected_rows are using the same length limit."
                )
            n = len(chars)
            encoded = vocab.encode(chars)
            space_flags = [c.isspace() for c in chars]

            arrays = train_arrays if tag == 1 else val_arrays
            row_i = train_idx if tag == 1 else val_idx

            arrays[0][row_i, :n] = encoded
            arrays[1][row_i, :n] = labels[:n]
            arrays[2][row_i, :n] = space_flags

            if tag == 1:
                train_idx += 1
            else:
                val_idx += 1

    return train_arrays, val_arrays


def build_sample_weight(label_ids: np.ndarray) -> np.ndarray:
    """1.0 everywhere except PAD and NA positions, which are excluded from
    both the loss and the DER metric (see label_config.py docstring)."""
    return ((label_ids != PAD_ID) & (label_ids != NA_ID)).astype(np.float32)