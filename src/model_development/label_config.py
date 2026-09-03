"""
label_config.py

Single source of truth for how we turn diacritized Arabic text into
model-ready labels. Every other file (encoding, DER metric, WER callback,
training script) imports its class ids and helper functions from here, so
there is exactly one place that defines what a "label" means.

LABEL SCHEME (agreed with the team):
    15 diacritic classes (ids 0-14), including "no diacritic" as id 0
    +1 NA class  (id 15) for positions that are not an Arabic letter at all
       (spaces, punctuation, digits, newlines) -- these can never carry a
       diacritic, so they are NOT the same thing as "letter with no mark".
    +1 PAD id (id 16), used only to fill batches up to a fixed length.

    NUM_TOTAL_CLASSES = 17. The model's output layer has 17 units so that
    PAD is a numerically valid (but always-ignored-via-sample_weight) index.
"""

from typing import List, Tuple

# --- Arabic diacritic (harakat) unicode code points ---
FATHA = "\u064E"
DAMMA = "\u064F"
KASRA = "\u0650"
SHADDA = "\u0651"
SUKUN = "\u0652"
FATHATAN = "\u064B"
DAMMATAN = "\u064C"
KASRATAN = "\u064D"

DIACRITIC_CHARS = frozenset({FATHA, DAMMA, KASRA, SHADDA, SUKUN, FATHATAN, DAMMATAN, KASRATAN})

# The 8 marks pyarabic's araby.TASHKEEL recognizes (used by the EDA notebook
# too) collapse into 15 classes once shadda+vowel combinations are merged
# into single ids (a model with one softmax head can only output one class
# per position, so "shadda AND fatha on the same letter" has to be one id,
# not two separate labels).
CLASS_LABELS: List[str] = [
    "",                              # 0  no diacritic
    FATHA,                           # 1
    DAMMA,                           # 2
    KASRA,                           # 3
    SUKUN,                           # 4
    FATHATAN,                        # 5
    DAMMATAN,                        # 6
    KASRATAN,                        # 7
    SHADDA,                          # 8  shadda with no vowel following (rare)
    SHADDA + FATHA,                  # 9
    SHADDA + DAMMA,                  # 10
    SHADDA + KASRA,                  # 11
    SHADDA + FATHATAN,               # 12
    SHADDA + DAMMATAN,               # 13
    SHADDA + KASRATAN,               # 14
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_LABELS)}
NO_DIACRITIC_ID = CLASS_TO_ID[""]     # 0
NUM_DIACRITIC_CLASSES = len(CLASS_LABELS)  # 15

NA_ID = 15          # not an Arabic letter: space, punctuation, digit, newline, ...
PAD_ID = 16         # batch padding only, never a real prediction target
NUM_REAL_CLASSES = NA_ID + 1          # 16 (15 diacritic classes + NA)
NUM_TOTAL_CLASSES = PAD_ID + 1        # 17 (used as the model's output width)

# The 36 standard Arabic letters (incl. all hamza forms, teh marbuta, alef
# maksura). Anything NOT in this set is treated as NA -- it structurally
# cannot carry a diacritic. Tatweel (\u0640) is deliberately NOT included:
# if it turns up in this corpus it will be labeled NA. Flagging this as a
# known edge case rather than silently guessing either way.
ARABIC_LETTERS = frozenset("ءآأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهويةى")


class UnknownDiacriticCombination(ValueError):
    """Raised when a letter's diacritic marks don't match any known class,
    OR when a letter has more than one non-shadda vowel mark (ambiguous --
    e.g. fatha+damma on the same letter, most likely malformed/corrupt
    source data). We refuse to guess in either case."""


def _classify_letter_marks(marks: List[str]):
    """
    Returns the composed class-lookup key for a letter's following marks,
    or None if the combination is ambiguous (more than one shadda, or more
    than one non-shadda vowel mark on the same letter -- this should never
    happen in valid Arabic and is treated the same as an unknown class,
    not silently resolved by picking the first mark.
    """
    shadda_count = marks.count(SHADDA)
    vowel_marks = [m for m in marks if m != SHADDA]
    if shadda_count > 1 or len(vowel_marks) > 1:
        return None
    vowel = vowel_marks[0] if vowel_marks else ""
    has_shadda = shadda_count == 1
    return (SHADDA + vowel) if (has_shadda and vowel) else (SHADDA if has_shadda else vowel)


def extract_chars_and_labels(
    diacritized_text: str,
    strict: bool = True,
    unknown_combo_counter=None,
) -> Tuple[List[str], List[int]]:
    """
    Walk a fully-diacritized string (the "output" column) and produce:
        chars      -- the base characters, in order (equal to the "input"
                       column, per the team's 100% consistency check)
        label_ids  -- one id per char, using the scheme in this module

    len(chars) == len(label_ids) always, by construction.

    Handles shadda+vowel in EITHER unicode order (shadda-then-vowel or
    vowel-then-shadda) since we collect ALL marks following a base
    character before deciding the class id.

    SAFETY (strict / unknown_combo_counter): if a letter's marks don't
    resolve to one of the 15 known classes -- either because the composed
    key isn't in CLASS_LABELS, or because the marks are ambiguous (e.g.
    two vowel marks on one letter) -- this used to silently fall back to
    NO_DIACRITIC_ID, which corrupts training data without any warning.
    Now:
        strict=True  (default): raises UnknownDiacriticCombination
                      immediately. Use this for the real encoding pass --
                      we want training to stop, not continue on bad labels.
        unknown_combo_counter=<a dict/Counter>: instead of raising, records
                      (char, key_or_marks) -> count and falls back to
                      NO_DIACRITIC_ID. Use this for an audit/reporting pass
                      over the whole corpus so you get one full report
                      instead of stopping at the first offender.
        If both are given, the counter takes precedence (report, don't
        raise) so a full-corpus scan can complete.
    """
    chars: List[str] = []
    label_ids: List[int] = []

    i, n = 0, len(diacritized_text)
    while i < n:
        ch = diacritized_text[i]
        if ch in DIACRITIC_CHARS:
            i += 1
            continue

        j = i + 1
        marks = []
        while j < n and diacritized_text[j] in DIACRITIC_CHARS:
            marks.append(diacritized_text[j])
            j += 1

        if ch in ARABIC_LETTERS:
            key = _classify_letter_marks(marks)
            if key is not None and key in CLASS_TO_ID:
                label_id = CLASS_TO_ID[key]
            else:
                descriptor = (ch, tuple(marks))
                if unknown_combo_counter is not None:
                    unknown_combo_counter[descriptor] = unknown_combo_counter.get(descriptor, 0) + 1
                    label_id = NO_DIACRITIC_ID
                elif strict:
                    raise UnknownDiacriticCombination(
                        f"Unrecognized diacritic combination on {ch!r}: marks={marks!r} "
                        f"(composed key={key!r}). Add it to CLASS_LABELS if it's valid, "
                        f"or investigate the source row if it's corrupt data."
                    )
                else:
                    label_id = NO_DIACRITIC_ID
        else:
            # Not a letter -> NA, regardless of any (malformed) marks that
            # might follow it. Space/punctuation/digits can't be diacritized.
            label_id = NA_ID

        chars.append(ch)
        label_ids.append(label_id)
        i = j

    return chars, label_ids


class CharVocab:
    """Character <-> id mapping for the model's INPUT side (separate id
    space from the diacritic labels above). id 0 = padding, id 1 = unknown."""

    PAD_ID = 0
    UNK_ID = 1

    def __init__(self):
        self.char_to_id = {}

    def build(self, chars) -> "CharVocab":
        vocab = sorted(set(chars))
        self.char_to_id = {c: i + 2 for i, c in enumerate(vocab)}
        return self

    def encode(self, chars: List[str]) -> List[int]:
        return [self.char_to_id.get(c, self.UNK_ID) for c in chars]

    def __len__(self) -> int:
        return len(self.char_to_id) + 2  # +PAD +UNK