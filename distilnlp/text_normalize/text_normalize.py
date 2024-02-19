import re
from functools import partial
from typing import Union, List

from .emoji import EMOJI_DICT
from distilnlp._utils.modelfile import downloaded_model_filepath
from .punctuation_normalize import punctuation_normalize

__all__ = [
    'text_normalize'
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
    # '，': ',',
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
    return text


def remove_unnecessary(text):
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
    
    return text


def text_punctuation_normalize(texts):
    model_name = 'punctuation_normalize'
    model_version = '20240218211723'
    url = 'https://raw.githubusercontent.com/wencan/distilNLP/150add0adb22af560591df5674101ac8ba3fe324/assets/model/punctuation_normalize/state_dict_20240218211723.pt'
    filepath = downloaded_model_filepath(model_name, model_version, url)

    texts = punctuation_normalize(filepath, texts)
    return texts


def text_normalize(texts_or_text: Union[List[str], str], enable_punctuation_normalize: bool=True):
    '''Text normalization processing removes redundant characters and corrects incorrect punctuation.
    If you are sure that the widths of the punctuations in the text are correct, please set enable_punctuation_normalize to False.
    '''

    is_str = False
    if isinstance(texts_or_text, str):
        texts = [texts_or_text]
        is_str = True
    else:
        texts = texts_or_text

    texts = [general_normalize(text) for text in texts]
    texts = [remove_unnecessary(text) for text in texts]

    if enable_punctuation_normalize:
        texts = text_punctuation_normalize(texts)

    if is_str:
        return texts[0]
    return texts
