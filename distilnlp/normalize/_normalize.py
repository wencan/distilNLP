import re
from typing import Literal, Union, Sequence
from functools import partial
from io import StringIO
from joblib import Parallel, delayed
from ._emoji import EMOJI_DICT

__all__ = [
    'normalize'
]

std_replace_table = {
    ' ': ' ', # added at 2024-01-22
    '　': ' ',
    # '！': '!',
    # '＂': '"',
    '＃': '#',
    '＄': '$',
    '％': '%',
    '＆': '&',
    '＇': "'",
    # '（': '(',
    # '）': ')',
    '＊': '*',
    '＋': '+',
    '，': ',',
    '－': '-',
    # '．': '.',
    '／': '/',
    '０': '0',
    '１': '1',
    '２': '2',
    '３': '3',
    '４': '4',
    '５': '5',
    '６': '6',
    '７': '7',
    '８': '8',
    '９': '9',
    # '：': ':',
    # '；': ';',
    '＜': '<',
    '＝': '=',
    '＞': '>',
    # '？': '?',
    '＠': '@',
    'Ａ': 'A',
    'Ｂ': 'B',
    'Ｃ': 'C',
    'Ｄ': 'D',
    'Ｅ': 'E',
    'Ｆ': 'F',
    'Ｇ': 'G',
    'Ｈ': 'H',
    'Ｉ': 'I',
    'Ｊ': 'J',
    'Ｋ': 'K',
    'Ｌ': 'L',
    'Ｍ': 'M',
    'Ｎ': 'N',
    'Ｏ': 'O',
    'Ｐ': 'P',
    'Ｑ': 'Q',
    'Ｒ': 'R',
    'Ｓ': 'S',
    'Ｔ': 'T',
    'Ｕ': 'U',
    'Ｖ': 'V',
    'Ｗ': 'W',
    'Ｘ': 'X',
    'Ｙ': 'Y',
    'Ｚ': 'Z',
    '［': '[',
    '＼': '\\',
    '］': ']',
    '＾': '^',
    '＿': '_',
    '｀': '`',
    'ａ': 'a',
    'ｂ': 'b',
    'ｃ': 'c',
    'ｄ': 'd',
    'ｅ': 'e',
    'ｆ': 'f',
    'ｇ': 'g',
    'ｈ': 'h',
    'ｉ': 'i',
    'ｊ': 'j',
    'ｋ': 'k',
    'ｌ': 'l',
    'ｍ': 'm',
    'ｎ': 'n',
    'ｏ': 'o',
    'ｐ': 'p',
    'ｑ': 'q',
    'ｒ': 'r',
    'ｓ': 's',
    'ｔ': 't',
    'ｕ': 'u',
    'ｖ': 'v',
    'ｗ': 'w',
    'ｘ': 'x',
    'ｙ': 'y',
    'ｚ': 'z',
    '｛': '{',
    '｜': '|',
    '｝': '}',
    '～': '~',
}

en_replace_table = {
    '！': '!',
    '＂': '"',
    '（': '(',
    '）': ')',
    '．': '.',
    '：': ':',
    '；': ';',
    '？': '?',
    '“': '"',
    '”': '"',
    '「': '"',
    '」': '"',
    '『': '"',
    '』': '"',
}

zh_replace_table = {
    '!': '！',
    # '"': '＂',
    '(': '（',
    ')': '）',
    '.': '。',
    '．': '。',
    '｡': '。',
    ':': '：',
    ';': '；',
    '?': '？',
    ',': '，',
    '「': '“',
    '」': '”',
    '『': '“',
    '』': '”',
}

space_pattern = re.compile(r'\s+')
half_split_pattern = re.compile(r'([\x21-\x7E\s]{3,}[^\x00-\x7F]?[\x21-\x7E\s]{3,})')
half_match_pattern = re.compile(r'^[\x21-\x7E\s]{3,}[^\x00-\x7F]?[\x21-\x7E\s]{3,}$')
ol_no_pattern = re.compile(r'^\d{1,3}\.\s')

def _replace(ch, replace_table):
    replace = replace_table.get(ch)
    if not replace is None:
        return replace
    return ch


std_replace = partial(_replace, replace_table=std_replace_table)
en_replace = partial(_replace, replace_table=en_replace_table)
zh_replace = partial(_replace, replace_table=zh_replace_table)


def general_normalize(text):
    '''basic normalizate for all languages.'''
    text = map(lambda ch: '' if ch in EMOJI_DICT else ch, text) # remove emoji
    text = map(std_replace, text)
    text = filter(lambda ch: ch.isprintable() or ch in ('\n', '\t'), text)
    text = ''.join(text)

    text = space_pattern.sub(' ', text)

    text = text.strip()

    return text.strip()


def en_normalize(text):
    '''more normalizate for English.'''
    text = map(en_replace, text)
    text = ''.join(text)

    # remove unnecessary `"`
    count = 0
    if text.startswith('"') or text.endswith('"'):
        for idx, ch in enumerate(text):
            if ch == '"':
                count+=1
    if count == 1:
        if text.startswith('"'):
            text = text[1:]
        if text.endswith('"'):
            text = text[:-1]

    return text


def _zh_part_norm(part):
    part = map(zh_replace, part)
    part = ''.join(part)
    # part = space_pattern.sub('', part)
    return part


def _left_char(i, text):
    if i <= 0:
        return ''
    
    left = text[i-1]
    if left == ' ':
        return _left_char(i-1, text)

    return left


def _right_char(i, text):
    if i >= len(text)-1:
        return ''
    
    right = text[i+1]
    if right == ' ':
        return _right_char(i+1, text)

    return right


def zh_normalize(text):
    '''more normalizate for Chinese.'''

    if any(ch for ch in text if ch.isalpha()):
        parts = half_split_pattern.split(text)
        norm_parts = []
        for part in parts:
            if half_match_pattern.match(part):
                part = en_normalize(part)
                norm_parts.append(part)
                continue

            part = _zh_part_norm(part)
            norm_parts.append(part)
        text = ''.join(norm_parts)
    else:
        text = _zh_part_norm(text)

    # remove unnecessary `“` or `”`
    stack = []
    if text.startswith('“') or text.endswith('”'):
        for idx, ch in enumerate(text):
            if ch == '“':
                stack.append(ch)
            elif ch == '”':
                if stack:
                    stack = stack[:-1]
                else:
                    if idx == len(text)-1:
                        text = text[:-1]
        if stack:
            text = text[1:]
    
    quotes_count = 0
    for i, ch in enumerate(text):
        if ch == '"':
            if quotes_count % 2 == 0:   # left
                right = _right_char(i, text)
                if right and (0x0020 <= ord(right) <= 0x007E):  # 0x0020 <= ord(halfwidth char) <= 0x007E
                    pass
                else:
                    text = text[:i] + '“' + text[i+1:]
            else: # right
                left = _left_char(i, text)
                if left and (0x0020 <= ord(left) <= 0x007E):
                    pass
                else:
                    text = text[:i] + '”' + text[i+1:]
            quotes_count+=1

    return text


def normalize(lang:Literal['en', 'zh'], text: str):
    '''Normalize punctuation, remove unnecessary characters and invisible characters.
    
    :param lang: Language code. Currently supports ``en`` and ``zh``.
    :param text_or_test : String or string sequence.
    '''

    text = general_normalize(text)

    if lang == 'en':
        text = en_normalize(text)
    elif lang == 'zh':
        text = zh_normalize(text)
    
    return text