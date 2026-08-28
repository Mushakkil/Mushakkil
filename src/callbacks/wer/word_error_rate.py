from keras.callbacks import Callback

class WordErrorRateCallback(Callback):
    """
    WER stands for Word Error Rate
    WER = Words wrong/ Words total
    * Words wrong: words containing at least one incorrectly predicted diacritic

    **Usage**
        model.fit(
            callbacks=[
            WordErrorRateCallback(
                val_data=(x_val, y_val),
                space_id=space_id,
                pad_id=PAD_ID,
            ),
        ]
    )

    to get space_id:
        keras tokenizer: space_id = tokenizer.word_index.get(" ")
        Hugging Face tokenizer: space_id = tokenizer.convert_tokens_to_ids(" ")

    **Example**
        Sentance: ذَهَبَ الْوَلَدُ
        predicted/Diacritize sentance: ذَهَبَ الْوَلَدِ
        
        total words = 2
        Words wrong = 1

        WER = 1/2 = 50%

    """
    def __init__(self, val_data, space_id, pad_id):
        super().__init__()
        self.x_val, self.y_val = val_data
        self.space_id = space_id
        self.pad_id = pad_id

    def on_epoch_end(self, epoch, logs=None):
        pred_ids = self.model.predict(self.x_val, verbose=0).argmax(axis=-1)
        wrong_words, total_words = 0, 0

        for gold_seq, pred_seq in zip(self.y_val, pred_ids):
            gw, pw, gold_words, pred_words = [], [], [], []
            for g, p in zip(gold_seq, pred_seq):
                if g == self.pad_id:
                    break
                if g == self.space_id:
                    gold_words.append(tuple(gw)); pred_words.append(tuple(pw))
                    gw, pw = [], []
                else:
                    gw.append(g); pw.append(p)
            if gw:
                gold_words.append(tuple(gw)); pred_words.append(tuple(pw))

            for gword, pword in zip(gold_words, pred_words):
                total_words += 1
                if gword != pword:
                    wrong_words += 1

        wer = wrong_words / max(total_words, 1)
        logs["val_wer"] = wer  # makes it show up in history
        print(f" — val_wer: {wer:.4f}")