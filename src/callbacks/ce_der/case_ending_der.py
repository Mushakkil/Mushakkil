from tensorflow.keras.callbacks import Callback

class CaseEndingDERCallback(Callback):
    """
    DER case ending = Sigma(words [last letter of words wrong]) / words

    ***Usage***:
        model.fit(
            callbacks=[
            CaseEndingDERCallback(
                val_data=(x_val, y_val, case_ending_mask_val),
            ),
        ]
    )
    """
    def __init__(self, val_data, pad_id, prediction_cache):
        super().__init__()
        self.x_val, self.y_val, self.case_ending_mask = val_data
        self.pad_id = pad_id

        # model.predict() over the full validation set runs once per epoch
        # instead of once per callback per epoch
        self.prediction_cache = prediction_cache

    def on_epoch_end(self, epoch, logs=None):
        if self.prediction_cache is not None:
            pred_ids = self.prediction_cache.get(self.model, self.x_val, epoch)
        else:
            pred_ids = self.model.predict(self.x_val, verbose=0).argmax(axis=-1)

        valid = (self.y_val != self.pad_id)
        wrong = (pred_ids != self.y_val)

        case_mask = self.case_ending_mask & valid
        noncase_mask = (~self.case_ending_mask) & valid 
        der_case = wrong[case_mask].sum() / max(case_mask.sum(), 1)

        logs["val_der_case_ending"] = der_case
        print(f" - val_der_case_ending: {der_case:.4f}")