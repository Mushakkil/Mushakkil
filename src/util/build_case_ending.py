def build_case_ending_mask(base_chars: list, word_boundary_chars=(" ",)) -> list:
    """
    base_chars: list of characters for one sample, in order (matches your y_true sequence)
    Used to prepare case_ending mask for CaseEndingDERCallback @see CaseEndingDERCallback
    @return list of bools, same length, True at each word-final diacritizable letter.
    """
    mask = [False] * len(base_chars)
    n = len(base_chars)

    for i in range(n):
        ch = base_chars[i]
        if ch in word_boundary_chars:
            continue
        # this char is the end of a word if the NEXT char is a boundary/EOS
        is_last_of_word = (i == n - 1) or (base_chars[i + 1] in word_boundary_chars)
        if is_last_of_word:
            mask[i] = True

    return mask