import re
import string
import unicodedata
from functools import partial
from ._emoji import EMOJI_DICT

__all__ = [
    'normalize'
]

std_replace_table = {
    ' ': ' ', # added at 2024-01-22
    '　': ' ',
    # '！': '!',
    '＂': '"',
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


_right_full2half_table = {
    '！': '!',
    # '＂': '"',
    '）': ')',
    '．': '.',
    '。': '.',
    '：': ':',
    '；': ';',
    '？': '?',
    '“': '"',
}

_right_half2full_table = {
    ',': '，',
    '!': '！',
    ')': '）',
    '.': '。',
    ':': '：',
    ';': '；',
    '?': '？',
}

_left_full2half_table = {
    '（': '(',
    '”': '"',
}

_left_half2full_table = {
    '(': '（',
}

space_pattern = re.compile(r'\s+')

def _replace(ch, replace_table):
    replace = replace_table.get(ch)
    if not replace is None:
        return replace
    return ch


std_replace = partial(_replace, replace_table=std_replace_table)


def general_normalize(text):
    '''basic normalizate for all languages.'''
    text = map(lambda ch: '' if ch in EMOJI_DICT else ch, text) # remove emoji
    text = map(std_replace, text)
    text = filter(lambda ch: ch.isprintable() or ch in ('\n', '\t'), text)
    text = ''.join(text)

    text = space_pattern.sub(' ', text)

    text = text.strip()

    return text.strip()


def char_kind(ch):
    tag = ''
    if ch in string.ascii_letters:
        return 'E' # English and digit
    elif '\u4e00' <= ch <= '\u9fff':
        return 'C' # Chinese
    if ch == ' ':
        tag = 'S' # space
    elif '\uff00' <= ch <= '\uffef':
        tag = 'F' # full width punctuation
    elif 33 <= ord(ch) <= 126:
        tag = 'H' # falf width punctuation
    else:
        tag = 'O' # Other
    return tag


def width_form_normalize(text):
    chs = []
    kinds = [char_kind(ch) for ch in text]
    # print(text, kinds)

    for idx, ch in enumerate(text):
        kind = kinds[idx]
        
        pre_ch_kind = ''
        i = idx-1
        while i>=0:
            if kinds[i] == 'S': # space
                i+=1
                continue
            pre_ch_kind = kinds[i]
            break
            # if kinds[i] in ('E', 'C'):
            #     pre_ch_kind = kinds[i]
            #     break
            # i-=1
        
        next_ch_kind = ''
        i = idx+1
        while i<len(text):
            # if kinds[i] in ('E', 'C'):
            #     next_ch_kind = kinds[i]
            # i+=1
            if kinds[i] == 'S': # space
                i-=1
                continue
            next_ch_kind = kinds[i]
            break

        if kind == 'F':
            if pre_ch_kind == 'E':
                if ch in _right_full2half_table.keys():
                    ch = _right_full2half_table[ch]
                    kind = 'H'
            if next_ch_kind == 'E':
                if ch in _left_full2half_table.keys():
                    ch = _left_full2half_table[ch]
                    kind = 'H'
        elif kind == 'H':
            if pre_ch_kind == 'C':
                if ch in _right_half2full_table.keys():
                    ch = _right_half2full_table[ch]
                    kind = 'F'
            if next_ch_kind == 'C':
                if ch in _left_half2full_table.keys():
                    ch = _left_half2full_table[ch]
                    kind = 'F'
        
        chs.append(ch)

    text = ''.join(chs)
    return text


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


def quote_normalize(text):
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


def normalize(text: str):
    '''Normalize punctuation, remove unnecessary characters and invisible characters.
    '''

    text = general_normalize(text)
    text = width_form_normalize(text)
    text = quote_normalize(text)
    
    return text