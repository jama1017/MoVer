import math
from decimal import Decimal
from pyrealb import NO

def word_category(word, vocab):
    rendered_vocab = realize_cont_to_str(vocab)
    if word in rendered_vocab:
        return True
    return False


def map_dict_type(word: str, dictionary: dict | list):
    ## list
    if dictionary.__class__ == list:
        return word_category(word, dictionary)
    
    ## dict
    for type, words_by_pos in dictionary.items():
        for pos, words in words_by_pos.items():
            rendered_words = realize_cont_to_str(words)
            if word in rendered_words:
                return type
    return None


def realize_cont_to_str(conts: list):
    res = []
    for cont in conts:
        if isinstance(cont, str):
            res.append(cont)
        else:
            if cont.constType == "V":
                res.append(cont.t("b").realize())
            else:
                res.append(cont.realize())
    return res


def num_cont_no_zero(num: float):
    """Handle precise number formatting for PyRealB - extracted from original."""
    if num.is_integer():
        dur_num_cont = NO(int(num))
    else:
        decimal = Decimal(str(num)).normalize()
        precision = decimal.as_tuple().exponent
        dur_num_cont = NO(float(decimal)).dOpt({"mprecision": abs(precision)})
    return dur_num_cont


def compute_x_y_mag(mag, direction):
    """Decompose magnitude by direction vector - extracted from original."""
    if not mag or not direction:
        return None, None
    
    x = abs(direction[0])
    y = abs(direction[1])
    hyp = math.sqrt(x**2 + y**2)
    ratio = mag / hyp
    x_mag = x * ratio
    y_mag = y * ratio
    return x_mag, y_mag


def divide_list_into_three(lst):
    """Generate all possible prefix/infix/suffix splits - extracted from original."""
    n = len(lst)
    results = []

    # Iterate through all possible split indices for three parts
    for i in range(n + 1):  # First split point (can be at the start or end)
        for j in range(i, n + 1):  # Second split point (can be at or after i)
            part1 = lst[:i]
            part2 = lst[i:j]
            part3 = lst[j:]
            results.append((part1, part2, part3))
    
    return results