"""
Character and diacritic vocabularies for Arabic diacritization.
"""

import tensorflow as tf
import pyarabic.araby as araby

try:
    BASE_LETTERS = list(araby.LETTERS)
    DIACRITIC_MARKS = set(araby.TASHKEEL)
except AttributeError:
    BASE_LETTERS = list("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي")
    DIACRITIC_MARKS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652')


EXTRA_CHARS = list(" .,،؛؟!0123456789-")
PAD_TOKEN = "<PAD>"

# StringLookup reserves index 0 for mask_token automatically
CHAR_VOCAB = BASE_LETTERS + EXTRA_CHARS

CHAR_LOOKUP = tf.keras.layers.StringLookup(
    vocabulary=CHAR_VOCAB, mask_token=PAD_TOKEN, oov_token="[UNK]", name="char_lookup"
)

try:
    _FATHA, _DAMMA, _KASRA, _SUKUN = araby.FATHA, araby.DAMMA, araby.KASRA, araby.SUKUN
    _FATHATAN, _DAMMATAN, _KASRATAN, _SHADDA = (
        araby.FATHATAN, araby.DAMMATAN, araby.KASRATAN, araby.SHADDA
    )
except AttributeError:
    _FATHA, _DAMMA, _KASRA, _SUKUN = '\u064E', '\u064F', '\u0650', '\u0652'
    _FATHATAN, _DAMMATAN, _KASRATAN, _SHADDA = '\u064B', '\u064C', '\u064D', '\u0651'

NONE_DIAC_TOKEN = "<NONE>"     # no diacritic on diacritizable letter
SPACE_DIAC_TOKEN = "<SPACE>"   # word boundary
DIAC_PAD_TOKEN = "<PAD>"       # padding sentinel

DIACRITIC_CLASSES = [
    NONE_DIAC_TOKEN,
    SPACE_DIAC_TOKEN,
    _FATHA, _DAMMA, _KASRA, _SUKUN,
    _FATHATAN, _DAMMATAN, _KASRATAN,
    _SHADDA,
    _SHADDA + _FATHA, _SHADDA + _DAMMA, _SHADDA + _KASRA,
    _SHADDA + _FATHATAN, _SHADDA + _DAMMATAN, _SHADDA + _KASRATAN,
]

DIAC_LOOKUP = tf.keras.layers.StringLookup(
    vocabulary=DIACRITIC_CLASSES, mask_token=DIAC_PAD_TOKEN, oov_token="[UNK]", name="diac_lookup"
)
NUM_DIACRITIC_CLASSES = DIAC_LOOKUP.vocabulary_size()


# IDs and reverse-lookup lists, computed ONCE.
CHAR_VOCAB_LIST = CHAR_LOOKUP.get_vocabulary()
DIAC_VOCAB_LIST = DIAC_LOOKUP.get_vocabulary()

CHAR_PAD_ID = CHAR_VOCAB_LIST.index(PAD_TOKEN)
DIAC_PAD_ID = DIAC_VOCAB_LIST.index(DIAC_PAD_TOKEN)
DIAC_SPACE_ID = DIAC_VOCAB_LIST.index(SPACE_DIAC_TOKEN)

MAX_LEN = 400  # ~50-60 Arabic words per Sadeed