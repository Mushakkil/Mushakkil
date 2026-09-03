import os
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from metrics.der.diacritic_error_rate import DiacriticErrorRate
from callbacks.ce_der.case_ending_der import CaseEndingDERCallback
from callbacks.wer.word_error_rate import WordErrorRateCallback
from helpers.datasets_helper.dataset import Dataset

from util.vocab import DIAC_PAD_ID
from util.data import make_dataset, to_training_triplet, materialize_validation_set
from models import MODEL_BUILDERS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_BUILDERS), default="ngram")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--case-ending-boost", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    """
    THE PROBLEM MIGHT BE HERE. IDK
    """
    train_ds = tf.data.experimental.make_csv_dataset(
        "datasets/Sadeed_Tashkeela/train.csv", 
        batch_size=3).take(100000)
    test_ds = tf.data.experimental.make_csv_dataset("datasets/Sadeed_Tashkeela/test.csv", batch_size=16)
    
    train_ds_full = make_dataset(train_ds, prefetch=True)
    test_ds_full = make_dataset(test_ds, prefetch=True)

    train_ds = train_ds_full.map(
        lambda c, d, w, m: to_training_triplet(c, d, w, m, case_ending_boost=args.case_ending_boost)
    )
    test_ds = test_ds_full.map(
        lambda c, d, w, m: to_training_triplet(c, d, w, m, case_ending_boost=0.0)
    )


    x_val, y_val, case_ending_mask_val = materialize_validation_set(test_ds_full)

    build_model = MODEL_BUILDERS[args.model]
    model = build_model(hidden_dim=args.hidden_dim, num_layers=args.num_layers) \
        if args.model in ("lstm", "gru") else \
        build_model(hidden_dim=args.hidden_dim, num_conv_layers=args.num_layers)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="char_accuracy"),
            # THIS IS OUR METRIC, DER 
            # DiacriticErrorRate(pad_id=DIAC_PAD_ID),
        ],
        weighted_metrics=[],
    )

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=[
            # THOSE ARE OUR CALLBACKS WER AND CE-DER
            #WordErrorRateCallback(val_data=(x_val, y_val)),
            #CaseEndingDERCallback(val_data=(x_val, y_val, case_ending_mask_val)),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ],
    )
    return model, history


if __name__ == "__main__":
    main()