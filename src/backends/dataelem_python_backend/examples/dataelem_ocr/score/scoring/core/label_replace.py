#   God Bless You.
#
#   file name: label_replace.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2019/04/30
#   description:
#

import os

CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

chars1 = 'Υǒ◆⒐§◎≈'
chars2 = 'Yo·9$〇='

biaodian = '.,;:/(){}?-%-+、'

REPLACE_DICT = {}


def load_replace_dict():
    replace_dict_file = os.path.join(CURRENT_FILE_DIRECTORY,
                                     './replace_chars.txt')
    replace_dict = {}
    for line in open(replace_dict_file, 'r'):
        line = line.replace('\n', '')
        k, v = line.split(' ', 1)
        replace_dict[k] = v

    # load chars1 -> chars2
    for i, k in enumerate(chars1):
        replace_dict[k] = chars2[i]

    return replace_dict


DICT = set()


def replace_unknow_chars(text, vocab_file):
    """replace chars in text which is not in vocab_file with 卍.
    The texts are first replaced by the `replaceFullToHalf`
    Args:
        text: str. The text to replace.
        vocab_file: str. The file path of the vocab_file

    Returns: str. replaced text.

    """
    global DICT
    if len(DICT) == 0:
        DICT = set(open(vocab_file, 'r').read().splitlines())
    text = replaceFullToHalf(text)
    text = text.replace(' ', '')
    text = text.replace('　', '')
    new_text = ''
    for char in text:
        if char not in DICT:
            new_text += '卍'
        else:
            new_text += char
    return new_text


def replaceFullToHalf(string):
    """TODO: Docstring for replaceFullToHalf.

    Args:
        string (TODO): TODO

    Returns: TODO

    """
    global REPLACE_DICT
    if len(REPLACE_DICT) == 0:
        REPLACE_DICT = load_replace_dict()
    new_str = ''
    for char in string:
        if char in REPLACE_DICT:
            new_str += REPLACE_DICT[char]
        else:
            new_str += char
    # for char in biaodian:
    # new_str = new_str.replace(char, "")
    return new_str


def main():
    save_file = os.path.join(CURRENT_FILE_DIRECTORY, './replace_chars.txt')
    replace_dict = load_replace_dict()
    with open(save_file, 'w') as f:
        for k in sorted(replace_dict.keys()):
            line = '{} {}\n'.format(k, replace_dict[k])
            f.write(line)


if __name__ == '__main__':
    main()
