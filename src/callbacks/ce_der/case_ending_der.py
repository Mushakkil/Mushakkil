from keras.callbacks import Callback

class CaseEndingDERCallback(Callback):
    """
    DER case ending = Sigma(words [last letter of words wrong]) / words

    ***Usage***:
        model.fit(
            callbacks=[
            CaseEndingDERCallback(
                val_data=(x_val, y_val),
                case_ending_mask=val_case_mask, @see src/util/build_case_ending_mask.py
                pad_id=PAD_ID,
            ),
        ]
    )
    """
    def __init__(self, val_data, case_ending_mask, pad_id):
        super().__init__()
        self.x_val, self.y_val = val_data
        self.case_ending_mask = case_ending_mask  # (num_samples, seq_len) bool array, precomputed
        self.pad_id = pad_id

    def on_epoch_end(self, epoch, logs=None):
        pred_ids = self.model.predict(self.x_val, verbose=0).argmax(axis=-1)

        valid = (self.y_val != self.pad_id)
        wrong = (pred_ids != self.y_val)

        case_mask = self.case_ending_mask & valid
        noncase_mask = (~self.case_ending_mask) & valid

        der_case = wrong[case_mask].sum() / max(case_mask.sum(), 1)
        der_noncase = wrong[noncase_mask].sum() / max(noncase_mask.sum(), 1)

        logs["val_der_case_ending"] = der_case
        logs["val_der_non_case_ending"] = der_noncase
        print(f" - val_der_case_ending: {der_case:.4f} - val_der_non_case_ending: {der_noncase:.4f}")