"""
train.py

Run from the repository root:
    uv run python src/model_development/train.py

Or, to run ONLY the pass-1 safety checks (no encoding, no models, no
training) before committing to a real run:
    uv run python src/model_development/train.py --audit-only

What the normal (non-audit) run does, step by step:
    1. Resolve the train.csv path via the existing Dataset helper
       (get_files_split only -- load_dataset() is NOT used, since it
       returns raw tf.data string tensors and doesn't produce aligned
       character/label arrays, which is what this task actually needs).
    2. Stream train.csv ONCE to filter rows, build the character
       vocabulary, and scan for unknown/ambiguous diacritic combinations
       (data_pipeline.filter_and_build_vocab).
    3. Pick a seeded, reproducible, disjoint 500,000-row train set and
       50,000-row validation set from the surviving rows.
    4. Stream train.csv a SECOND time to encode exactly those rows into
       numpy arrays.
    5. Build BiLSTM and BiGRU with identical hyperparameters/batch size.
    6. Train both, evaluate DER + WER on the validation set.
    7. Print a comparison table and save both models to checkpoints/.

--audit-only stops after step 2 and reports what it found -- it never
encodes arrays, never builds a model, and never calls model.fit().

This does NOT touch test.csv, and does NOT build the N-gram baseline
(that is a deliberately separate, later step).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# --- make sibling/sibling-package modules importable ---
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))                                   # label_config, data_pipeline, models
sys.path.insert(0, str(REPO_ROOT / "src" / "helpers" / "datasets_helper"))  # dataset.py (unmodified)
sys.path.insert(0, str(REPO_ROOT / "src" / "metrics" / "der"))       # diacritic_error_rate.py (fixed)
sys.path.insert(0, str(REPO_ROOT / "src" / "callbacks" / "wer"))

from dataset import Dataset                                # noqa: E402
from diacritic_error_rate import DiacriticErrorRate         # noqa: E402
from word_error_rate import WordErrorRateCallback            # noqa: E402

from label_config import PAD_ID, UnknownDiacriticCombination  # noqa: E402
from data_pipeline import (                                  # noqa: E402
    filter_and_build_vocab,
    select_train_val_row_ids,
    encode_selected_rows,
    build_sample_weight,
    N_TRAIN,
    N_VAL,
    SEED,
)
from models import build_bilstm_model, build_bigru_model, compile_model  # noqa: E402

BATCH_SIZE = 128
EPOCHS = 5  # model-selection run only; more epochs come later once a winner is picked

tf.random.set_seed(SEED)
np.random.seed(SEED)


def run_audit_only(train_csv: str) -> None:
    """
    Runs ONLY pass-1 (filtering + vocab + diacritic-combination scan) and
    reports the results, then exits. Never touches pass 2 (encoding),
    never builds a model, never calls model.fit(). Never opens test.csv --
    only `train_csv` (already resolved via Dataset.get_files_split) is
    read.
    """
    required = N_TRAIN + N_VAL
    print(f"Using train CSV: {train_csv}")
    print("\nRunning pass-1 filtering, vocabulary build, and diacritic-combination scan...")

    try:
        filter_result = filter_and_build_vocab(train_csv)
    except UnknownDiacriticCombination as e:
        print("\n=== AUDIT FAILED: unknown/ambiguous diacritic combinations found ===\n")
        print(str(e))
        sys.exit(1)

    n_total = filter_result.n_total_rows
    n_kept = filter_result.n_kept_rows
    n_dropped = n_total - n_kept
    vocab_size = len(filter_result.vocab)

    print("\n=== Audit report ===")
    print(f"  total rows in file:                              {n_total:,}")
    print(f"  rows surviving current EDA filters:               {n_kept:,}")
    print(f"  rows dropped:                                     {n_dropped:,}")
    print(f"  character vocabulary size (incl. PAD/UNK):        {vocab_size}")
    print(f"  unknown/ambiguous diacritic combinations found:   none")
    print(f"  usable rows required for {N_TRAIN:,} train + {N_VAL:,} val:   {required:,}")

    if n_kept < required:
        print(
            f"\n=== AUDIT FAILED: only {n_kept:,} usable rows remain, "
            f"but {required:,} are needed for the planned split. ==="
        )
        sys.exit(1)

    print(f"  usable rows available: {n_kept:,} >= {required:,} required -- OK")
    print("\nAUDIT PASSED — no training was started.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Mushakkil model-development training/audit script")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run only the pass-1 filtering/vocab/diacritic-combination checks, "
             "then exit. No encoding, no model building, no training.",
    )
    args = parser.parse_args()

    ds = Dataset(name="sadeed_tashkeal")
    train_csv = ds.get_files_split(split="train")

    if args.audit_only:
        run_audit_only(train_csv)
        return  # unreachable (run_audit_only always calls sys.exit), kept for clarity

    print(f"Using train CSV: {train_csv}")

    print("\n[1/4] Streaming train.csv to filter rows and build vocabulary...")
    filter_result = filter_and_build_vocab(train_csv)
    print(f"  total rows in file:  {filter_result.n_total_rows:,}")
    print(f"  rows kept after EDA filters: {filter_result.n_kept_rows:,}")
    print(f"  character vocab size (incl. PAD/UNK): {len(filter_result.vocab)}")

    print(f"\n[2/4] Selecting {N_TRAIN:,} train / {N_VAL:,} val rows (seed={SEED})...")
    assignment = select_train_val_row_ids(filter_result, seed=SEED)

    print("\n[3/4] Streaming train.csv again to encode the selected rows...")
    (X_train, y_train, space_train), (X_val, y_val, space_val) = encode_selected_rows(
        train_csv, assignment, filter_result.vocab
    )
    print(f"  X_train shape: {X_train.shape}   y_train shape: {y_train.shape}")
    print(f"  X_val shape:   {X_val.shape}     y_val shape:   {y_val.shape}")

    w_train = build_sample_weight(y_train)
    w_val = build_sample_weight(y_val)
    print(f"  avg fraction of positions counted (not PAD/NA), train: {w_train.mean():.3f}")

    vocab_size = len(filter_result.vocab)
    candidates = {
        "bilstm": build_bilstm_model(vocab_size),
        "bigru": build_bigru_model(vocab_size),
    }

    print("\n[4/4] Training both models with identical hyperparameters...")
    results = {}
    for name, model in candidates.items():
        print(f"\n--- {name} ---")
        compile_model(model, DiacriticErrorRate(pad_id=PAD_ID))
        wer_cb = WordErrorRateCallback(val_data=(X_val, y_val, space_val), pad_id=PAD_ID)

        history = model.fit(
            X_train, y_train,
            sample_weight=w_train,
            validation_data=(X_val, y_val, w_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[
                wer_cb,
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_der", mode="min", patience=2, restore_best_weights=True
                ),
            ],
        )

        final_der = history.history["val_der"][-1]
        final_wer = history.history["val_wer"][-1]
        results[name] = {"val_der": final_der, "val_wer": final_wer}

        checkpoint_dir = REPO_ROOT / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        model.save(checkpoint_dir / f"{name}.keras")

    print("\n=== Comparison (lower is better) ===")
    for name, r in results.items():
        print(f"  {name}: DER={r['val_der']:.4f}  WER={r['val_wer']:.4f}")
    winner = min(results, key=lambda k: results[k]["val_der"])
    print(f"\nSelected model (lowest DER): {winner}")


if __name__ == "__main__":
    main()