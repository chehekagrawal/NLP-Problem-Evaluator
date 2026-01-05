import numpy as np
import re


def get_manual_features(text):
    """
    Extracts 6 specific features from the text:
    1. Text Length
    2. Math Symbol Count
    3. Keyword Frequency
    4. Word Count
    5. Avg Word Length
    6. Number Count

    Returns: A list of 6 numerical values.
    """
    # 1. Math Symbols
    math_symbols = ['+', '-', '*', '/', '=', '<', '>', '^', '_', '{', '}', '$', '%']
    math_count = sum(text.count(s) for s in math_symbols)

    # 2. Keywords
    keywords = ['graph', 'tree', 'dp', 'recursion', 'array', 'greedy', 'binary', 'modulo']
    keyword_freq = sum(text.lower().count(k) for k in keywords)

    # 3. Text Length
    text_len = len(text)

    # 4. Word Count
    words = text.split()
    word_count = len(words)

    # 5. Average Word Length
    if word_count > 0:
        avg_word_len = np.mean([len(w) for w in words])
    else:
        avg_word_len = 0.0

    # 6. Number Count
    number_count = len(re.findall(r'\d+', text))

    return [text_len, math_count, keyword_freq, word_count, avg_word_len, number_count]