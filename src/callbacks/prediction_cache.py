
class PredictionCache:
    """
    WordErrorRateCallback and CaseEndingDERCallback each call
    `self.model.predict(self.x_val, verbose=0)` independently in
    on_epoch_end. Since both callbacks are handed the SAME x_val, that's a
    full forward pass over the entire validation set run TWICE per epoch,
    for every epoch of training -- pure duplicated GPU/CPU work with no
    benefit. If you add a third val-set callback later without this cache,
    you'd be at 3x, and so on.
    """ 
    def __init__(self):
        self._epoch = -1
        self._pred_ids = None
 
    def get(self, model, x_val, epoch):
        if epoch != self._epoch:
            self._pred_ids = model.predict(x_val, verbose=0).argmax(axis=-1)
            self._epoch = epoch
        return self._pred_ids
