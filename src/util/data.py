"""
tf.data pipeline for Arabic diacritization.

Architecture-agnostic: builds (char_ids, diac_ids, sample_weight,
case_ending_mask) tuples that ANY sequence-tagging model (Conv1D n-gram,
LSTM, GRU, ...) can consume, since they all take the same
(batch, MAX_LEN) integer input and produce the same per-timestep labels.

PERFORMANCE NOTES (read before changing epochs/batch size)
------------------------------------------------------------
The heavy step here is `_process_batch`: it runs as a `tf.py_function`,
which means:
  1. It executes in plain Python, not as a TF graph op -- no XLA, no GPU.
  2. `num_parallel_calls=tf.data.AUTOTUNE` gives you only PARTIAL overlap.
     CPython's GIL means multiple py_function calls cannot run their
     Python bytecode concurrently, so this does not scale the way a pure
     graph op would. It still helps a little (I/O and numpy-C portions can
     overlap), just don't expect the speed-up `AUTOTUNE` implies elsewhere.
  3. The original code cached the RAW csv dataset, before this map step,
     so `_process_batch` re-ran from scratch every epoch. That is almost
     certainly the biggest single cost in the whole training loop. Fixed
     below by caching AFTER preprocessing.

If this is still your bottleneck after that fix, the actual fastest option
is to stop parsing text at train time altogether: run this preprocessing
ONCE, write the resulting int arrays to TFRecord/npz on disk, and have the
training `tf.data.Dataset` read pre-tokenized examples. That removes the
py_function (and the GIL bottleneck) from the training loop entirely. That
is a bigger change than what was asked for here, so it isn't implemented
below, but it is the direction to go if caching-after-map isn't enough.
"""
import re

import numpy as np
import tensorflow as tf

from util.build_case_ending import build_case_ending_mask
from util.vocab import (
    CHAR_LOOKUP, DIAC_LOOKUP, DIACRITIC_MARKS, PAD_TOKEN,
    NONE_DIAC_TOKEN, SPACE_DIAC_TOKEN, DIAC_PAD_TOKEN, MAX_LEN,
)


DIACRITIZED_COLUMN = "output"

# Compiled once at import time. Matches "one non-diacritic char, followed
# by zero or more diacritic marks" -- re.finditer skips over any text that
# doesn't match (e.g. a stray leading diacritic with no base char), which
# reproduces the original hand-rolled loop's behavior of skipping orphan
# diacritics, but lets the C-level regex engine do the scanning instead of
# a python `while` loop over every character.
_DIAC_CLASS = "".join(re.escape(m) for m in DIACRITIC_MARKS)
_UNIT_PATTERN = re.compile(rf"([^{_DIAC_CLASS}])([{_DIAC_CLASS}]*)")


def split_units(text):
    """Text -> list of (base_char, diacritic_run) tuples."""
    return [(base, diac) for base, diac in _UNIT_PATTERN.findall(text)]


def units_to_string_arrays(units, word_boundary_chars=(" ",), max_len=MAX_LEN):
    """
    Convert units into fixed-length arrays of STRINGS (not ids) -- id
    conversion is StringLookup's job, done as a graph op afterward.

    Three distinct diacritic-label sentinels, not one:
      - positions beyond the real sequence  -> DIAC_PAD_TOKEN
      - positions whose base char is a space -> SPACE_DIAC_TOKEN
      - real letters with no diacritic mark  -> NONE_DIAC_TOKEN
    Callbacks that read pad_id / space_id directly off y_val rely on these
    staying distinguishable.
    """
    units = units[:max_len]
    base_chars = np.full(max_len, PAD_TOKEN, dtype=object)
    diac_strs = np.full(max_len, DIAC_PAD_TOKEN, dtype=object)
    for i, (base, diac) in enumerate(units):
        base_chars[i] = base
        if base in word_boundary_chars:
            diac_strs[i] = SPACE_DIAC_TOKEN
        else:
            diac_strs[i] = diac if diac != "" else NONE_DIAC_TOKEN
    length = len(units)
    return base_chars, diac_strs, length


def build_sample_weights(lengths, max_len=MAX_LEN):
    """1.0 for real positions, 0.0 for padding."""
    weights = np.zeros((len(lengths), max_len), dtype=np.float32)
    for i, length in enumerate(lengths):
        weights[i, :length] = 1.0
    return weights


def _process_batch(diac_text_batch):
    """
    Runs inside tf.py_function -- plain Python/NumPy. Derives BOTH the
    input characters and the diacritic labels from the diacritized column
    alone (via split_units), so the input and label sequences can't drift
    out of sync due to differing normalization between an undiacritized
    and a diacritized column.

    Also builds the case-ending mask here, because build_case_ending_mask
    needs the raw base_chars list with word boundaries intact -- exactly
    what split_units() produces before it gets flattened into arrays.
    """
    diac_texts = [t.decode("utf-8") for t in diac_text_batch.numpy()]
    n = len(diac_texts)
    batch_base_chars = np.full((n, MAX_LEN), PAD_TOKEN, dtype=object)
    batch_diac_strs = np.full((n, MAX_LEN), DIAC_PAD_TOKEN, dtype=object)
    batch_weights = np.zeros((n, MAX_LEN), dtype=np.float32)
    batch_case_ending_mask = np.zeros((n, MAX_LEN), dtype=np.float32)

    for i, text in enumerate(diac_texts):
        units = split_units(text)
        base_chars, diac_strs, length = units_to_string_arrays(units)
        batch_base_chars[i] = base_chars
        batch_diac_strs[i] = diac_strs
        batch_weights[i, :length] = 1.0

        raw_base_chars = [u[0] for u in units]
        ce_mask = build_case_ending_mask(raw_base_chars, word_boundary_chars=(" ",))
        batch_case_ending_mask[i, :length] = np.asarray(ce_mask, dtype=np.float32)[:length]

    return (
        batch_base_chars.astype(str),
        batch_diac_strs.astype(str),
        batch_weights,
        batch_case_ending_mask,
    )


def preprocess_csv_batch(column_dict):
    """.map() function to append after a csv dataset."""
    diac_col = column_dict[DIACRITIZED_COLUMN]
    base_char_strs, diac_strs, weights, case_ending_mask = tf.py_function(
        func=_process_batch,
        inp=[diac_col],
        Tout=[tf.string, tf.string, tf.float32, tf.float32],
    )
    # tf.py_function erases static shape info -- restore it so Keras can
    # build the graph.
    base_char_strs.set_shape([None, MAX_LEN])
    diac_strs.set_shape([None, MAX_LEN])
    weights.set_shape([None, MAX_LEN])
    case_ending_mask.set_shape([None, MAX_LEN])

    char_ids = CHAR_LOOKUP(base_char_strs)
    diac_ids = DIAC_LOOKUP(diac_strs)
    return char_ids, diac_ids, weights, case_ending_mask


def make_dataset(raw_csv_dataset, cache=True, prefetch=True):
    """
    Returns 4-tuples: (char_ids, diac_ids, sample_weight, case_ending_mask).

    cache=True caches AFTER preprocess_csv_batch, not before -- this is
    the actual fix for the "reruns the Python parsing loop every epoch"
    problem. Trade-off: this caches the fully parsed int tensors in
    memory (or on disk if you pass make_dataset(..., cache='/path/file')-
    style; see the note below), which costs more RAM than caching raw
    strings did, but means the expensive py_function step only ever runs
    ONCE per example, not once per example per epoch.

    If your dataset doesn't comfortably fit in memory, replace
    `ds.cache()` with `ds.cache(filename)` to spill to disk instead -- the
    parsing cost is still paid only once, you just trade RAM for disk I/O.
    """
    ds = raw_csv_dataset.map(preprocess_csv_batch, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def to_training_triplet(char_ids, diac_ids, weights, case_ending_mask, case_ending_boost=0.0):
    """
    Drop the case-ending mask for model.fit()/compile(), which expects
    exactly (x, y, sample_weight).

    case_ending_boost > 0 makes the LOSS itself weight case-ending
    positions more heavily during training, not just evaluation.
    """
    if case_ending_boost > 0:
        weights = weights * (1.0 + case_ending_boost * case_ending_mask)
    return char_ids, diac_ids, weights


def materialize_validation_set(test_ds_full):
    """
    Consumes the 4-tuple dataset ONE time, fully, into numpy arrays for
    callbacks that index directly into x_val/y_val (e.g. to run their own
    model.predict() in on_epoch_end). Only appropriate for a validation
    split that comfortably fits in memory.
    """
    char_ids_batches, diac_ids_batches, mask_batches = [], [], []
    for char_ids, diac_ids, _weights, case_ending_mask in test_ds_full:
        char_ids_batches.append(char_ids.numpy())
        diac_ids_batches.append(diac_ids.numpy())
        mask_batches.append(case_ending_mask.numpy())

    x_val = np.concatenate(char_ids_batches, axis=0)
    y_val = np.concatenate(diac_ids_batches, axis=0)
    case_ending_mask_bool = np.concatenate(mask_batches, axis=0).astype(bool)
    return x_val, y_val, case_ending_mask_bool

def encode_text(text, max_len=MAX_LEN):
    """
    Encode undiacritized Arabic text exactly as expected by the model.
    """
    chars = list(text)[:max_len]

    padded = np.full(max_len, PAD_TOKEN, dtype=object)
    padded[:len(chars)] = chars

    char_ids = CHAR_LOOKUP(
        tf.constant(padded.reshape(1, -1), dtype=tf.string)
    )

    return char_ids