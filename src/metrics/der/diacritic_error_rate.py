import tensorflow as tf

class DiacriticErrorRate(tf.keras.metrics.Metric):
    """
    DER = (incorrectly predicted diacritic characters) / (total diacriticable characters)
    y_true: (batch, seq_len) integer class ids, padded with pad_id
    y_pred: (batch, seq_len, num_classes) softmax probabilities

    **Usage**:\n
        model.compile(
            optimizer=optimizer,
            loss=selected_loss
            metrics=[DiacriticErrorRate(pad_id=PAD_ID)]
        )

    **Example**:
        Sentance: ذَهَبَ الْوَلَدُ
        predicted/Diacritize sentance: ذَهَبَ الْوَلَدِ
        
        total diacriticable characters = 8
        incorrectly predicted diacritic = 1

        DER = 1/8 = %12.5
    """

    def __init__(self, pad_id, name="der", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_id = pad_id
        self.wrong = self.add_weight(name="wrong", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int64)
        pred_ids = tf.argmax(y_pred, axis=-1)

        mask = tf.cast(tf.not_equal(y_true, self.pad_id), tf.float32)  # exclude padding
        correct = tf.cast(tf.equal(y_true, pred_ids), tf.float32)
        wrong = (1.0 - correct) * mask

        self.wrong.assign_add(tf.reduce_sum(wrong))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.wrong / (self.total + tf.keras.backend.epsilon())

    def reset_state(self):
        self.wrong.assign(0.0)
        self.total.assign(0.0)