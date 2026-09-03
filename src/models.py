import numpy as np
import tensorflow as tf

from util.vocab import (
    CHAR_LOOKUP, NUM_DIACRITIC_CLASSES, MAX_LEN, CHAR_PAD_ID,
    CHAR_VOCAB_LIST, DIAC_VOCAB_LIST, NONE_DIAC_TOKEN, SPACE_DIAC_TOKEN, DIAC_PAD_TOKEN,
)


def build_sequence_tagger(
    encoder_fn,
    vocab_size=None,
    num_classes=NUM_DIACRITIC_CLASSES,
    embed_dim=64,
    max_len=MAX_LEN,
    name="sequence_tagger",
):
    """Shared skeleton: Embedding -> encoder_fn(x) -> per-timestep Dense softmax."""
    if vocab_size is None:
        vocab_size = CHAR_LOOKUP.vocabulary_size()

    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="char_ids")
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim, mask_zero=False, name="char_embedding"
    )(inputs)
    x = encoder_fn(x)
    # Dense on a 3D tensor is applied independently at every timestep in
    # tf.keras -- one classification per character position, same for every
    # encoder below.
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="diacritic_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


# ---------------------------------------------------------------------
# Encoders -- each returns a callable (tensor) -> tensor
# ---------------------------------------------------------------------

def ngram_conv_encoder(hidden_dim=128, kernel_size=7, num_layers=2):
    """
    kernel_size IS the n-gram order: 7 = 3 chars left + itself + 3 right.
    Stacking layers widens the EFFECTIVE receptive field without growing
    kernel_size directly -- two kernel_size=7 layers see a combined span
    of 13 characters, cheaper than one much larger kernel.
    """
    def _encoder(x):
        for i in range(num_layers):
            x = tf.keras.layers.Conv1D(
                filters=hidden_dim, kernel_size=kernel_size, padding="same",
                activation="relu", name=f"ngram_conv_{i}",
            )(x)
        return x
    return _encoder


def lstm_encoder(hidden_dim=128, num_layers=1, bidirectional=True, dropout=0.0):
    """
    Bidirectional by default: diacritization is non-causal (a word's
    ending diacritic can depend on words that come after it, e.g. case
    marking driven by a following particle), so seeing right-context
    matters here, unlike e.g. next-token language modeling.
    """
    def _encoder(x):
        for i in range(num_layers):
            layer = tf.keras.layers.LSTM(
                hidden_dim, return_sequences=True, dropout=dropout, name=f"lstm_{i}"
            )
            if bidirectional:
                layer = tf.keras.layers.Bidirectional(layer, name=f"bilstm_{i}")
            x = layer(x)
        return x
    return _encoder


def gru_encoder(hidden_dim=128, num_layers=1, bidirectional=True, dropout=0.0):
    """Same shape as lstm_encoder; GRU trades a little modeling capacity
    for fewer parameters and faster steps (no separate cell state)."""
    def _encoder(x):
        for i in range(num_layers):
            layer = tf.keras.layers.GRU(
                hidden_dim, return_sequences=True, dropout=dropout, name=f"gru_{i}"
            )
            if bidirectional:
                layer = tf.keras.layers.Bidirectional(layer, name=f"bigru_{i}")
            x = layer(x)
        return x
    return _encoder


# ---------------------------------------------------------------------
# Convenience constructors -- thin wrappers so call sites read the same
# way they did before (build_ngram_model(...)), just backed by the shared
# factory instead of duplicated model code.
# ---------------------------------------------------------------------

def build_ngram_model(hidden_dim=128, kernel_size=7, num_conv_layers=2, embed_dim=64, **kw):
    return build_sequence_tagger(
        ngram_conv_encoder(hidden_dim, kernel_size, num_conv_layers),
        embed_dim=embed_dim, name="ngram_diacritizer", **kw,
    )


def build_lstm_model(hidden_dim=128, num_layers=1, bidirectional=True, dropout=0.0, embed_dim=64, **kw):
    return build_sequence_tagger(
        lstm_encoder(hidden_dim, num_layers, bidirectional, dropout),
        embed_dim=embed_dim, name="lstm_diacritizer", **kw,
    )


def build_gru_model(hidden_dim=128, num_layers=1, bidirectional=True, dropout=0.0, embed_dim=64, **kw):
    return build_sequence_tagger(
        gru_encoder(hidden_dim, num_layers, bidirectional, dropout),
        embed_dim=embed_dim, name="gru_diacritizer", **kw,
    )


MODEL_BUILDERS = {
    "ngram": build_ngram_model,
    "lstm": build_lstm_model,
    "gru": build_gru_model,
}


def decode_predictions(pred_probs, char_ids):
    """
    Turn model output back into diacritized text. Match this logic in any
    WER/CE-DER callback so decoding stays consistent everywhere. Uses the
    vocab lists cached once in vocab.py, not re-fetched here.
    """
    pred_classes = np.argmax(pred_probs, axis=-1)
    texts = []
    for chars, classes in zip(char_ids, pred_classes):
        out = []
        for c_id, d_id in zip(chars, classes):
            if c_id == CHAR_PAD_ID:
                break
            diac = DIAC_VOCAB_LIST[d_id]
            diac = "" if diac in (NONE_DIAC_TOKEN, SPACE_DIAC_TOKEN, DIAC_PAD_TOKEN) else diac
            out.append(CHAR_VOCAB_LIST[c_id] + diac)
        texts.append("".join(out))
    return texts