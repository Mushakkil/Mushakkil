"""
models.py

Model Architecture (proposal Method step 2):
    a. N-gram baseline
    b. BiLSTM vs BiGRU comparison (same hyperparams/batch size, no regularization)
    c/d. best of (b) gets compared to baseline, then hyper-tuned
"""

from collections import Counter, defaultdict
from typing import List, Sequence

import tensorflow as tf
import keras
from keras import layers, models
from label_config import NUM_REAL_CLASSES as NUM_CLASSES, CharVocab

# ---------------------------------------------------------------------------
# a. N-gram baseline
# ---------------------------------------------------------------------------
class NgramBaseline:
    """
    Context-window frequency baseline: for each character, look at the
    (n-1) preceding characters as context and predict the most frequent
    diacritic label observed for that (context, char) pair in training data.
    Falls back to unigram (char-only) frequency, then to global majority
    class, when a context was never seen -- standard backoff.

    This is intentionally simple; it exists to give the BiLSTM/BiGRU models
    a lower bound to justify the extra complexity/compute.
    """

    def __init__(self, n: int = 3):
        self.n = n  # total context window size, e.g. 3 = 2 preceding chars + current
        self.ngram_counts = defaultdict(Counter)   # (context_tuple, char) -> Counter(label)
        self.unigram_counts = defaultdict(Counter)  # char -> Counter(label)
        self.global_counts = Counter()              # label -> count
        self._ngram_best = {}
        self._unigram_best = {}
        self._global_best = 0

    def fit(self, char_sequences: Sequence[List[str]], label_sequences: Sequence[List[int]]) -> "NgramBaseline":
        pad = "\u0000"
        ctx_len = self.n - 1
        for chars, labels in zip(char_sequences, label_sequences):
            padded = [pad] * ctx_len + list(chars)
            for idx, (ch, lbl) in enumerate(zip(chars, labels)):
                context = tuple(padded[idx: idx + ctx_len])
                self.ngram_counts[(context, ch)][lbl] += 1
                self.unigram_counts[ch][lbl] += 1
                self.global_counts[lbl] += 1
        self._finalize()
        return self

    def _finalize(self):
        self._ngram_best = {k: c.most_common(1)[0][0] for k, c in self.ngram_counts.items()}
        self._unigram_best = {k: c.most_common(1)[0][0] for k, c in self.unigram_counts.items()}
        self._global_best = self.global_counts.most_common(1)[0][0] if self.global_counts else 0

    def predict(self, chars: List[str]) -> List[int]:
        pad = "\u0000"
        ctx_len = self.n - 1
        padded = [pad] * ctx_len + list(chars)
        preds = []
        for idx, ch in enumerate(chars):
            context = tuple(padded[idx: idx + ctx_len])
            if (context, ch) in self._ngram_best:
                preds.append(self._ngram_best[(context, ch)])
            elif ch in self._unigram_best:
                preds.append(self._unigram_best[ch])
            else:
                preds.append(self._global_best)
        return preds


# ---------------------------------------------------------------------------
# b. BiLSTM / BiGRU builders -- identical shape/hyperparams so the comparison
#    (proposal 2.b.ii) is fair; only the recurrent cell type differs.
# ---------------------------------------------------------------------------
def _build_recurrent_model(
    vocab_size: int,
    rnn_layer_cls,
    embedding_dim: int = 128,
    rnn_units: int = 256,
    num_rnn_layers: int = 2,
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(None,), dtype="int32", name="char_ids")
    x = layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="char_embedding"
    )(inputs)

    for i in range(num_rnn_layers):
        x = layers.Bidirectional(
            rnn_layer_cls(rnn_units, return_sequences=True),
            name=f"bidirectional_{i}",
        )(x)

    outputs = layers.TimeDistributed(
        layers.Dense(num_classes, activation="softmax"), name="diacritic_output"
    )(x)

    return models.Model(inputs=inputs, outputs=outputs)


def build_bilstm_model(vocab_size: int, **kwargs) -> tf.keras.Model:
    return _build_recurrent_model(vocab_size, layers.LSTM, **kwargs)


def build_bigru_model(vocab_size: int, **kwargs) -> tf.keras.Model:
    return _build_recurrent_model(vocab_size, layers.GRU, **kwargs)


def compile_model(
    model: tf.keras.Model,
    der_metric,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        weighted_metrics=["accuracy", der_metric],
    )
    return model