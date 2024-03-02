import unicodedata
import collections
import argparse
import string
from typing import Optional

import tqdm

from distilnlp.utils.unicode import is_printable_symbol


def save_vocab(vocab: collections.OrderedDict, filepath: str):
    with open(filepath, 'w') as outfile:
        for idx, (s, i) in enumerate(vocab.items()):
            if idx != 0:
                outfile.write('\n')
            outfile.write(f'{s}\t{i}')


def load_vocab(filepath:str) -> collections.OrderedDict:
    ordered_dict = collections.OrderedDict()

    with open(filepath) as infile:
        while True:
            s = infile.read(1)
            if s == '': # char
                break
            t = infile.read(1)
            while t != '\t':    # words
                s += t
                t = infile.read(1)
            
            digits = ''
            while True:
                d = infile.read(1)
                if d in ('\n', ''):
                    break
                digits += d
            i = int(digits)

            ordered_dict[s] = i

    return ordered_dict


def generate_char_counter(filepaths) -> collections.Counter:
    counter = collections.Counter()
    for filepath in filepaths:
        print(filepath)
        with open(filepath) as infile:
            for line in infile:
                for ch in line:
                    counter[ch] +=1
    return counter


def clean_char_counter(char_counter: collections.Counter,
                       min_freq:int = 100,
                       filter_freq:int = 100000) -> collections.Counter:
    new_counter = collections.Counter()

    for ch, count in tqdm.tqdm(char_counter.items()):
        if not is_printable_symbol(ch):
            continue

        if count < min_freq:
            continue
        elif count < filter_freq:
            cateory = unicodedata.category(ch)
            try:
                name = unicodedata.name(ch)
            except ValueError:
                continue

            if cateory in ('Lo', 'So'): # Other Letter, Other Symbol
                if not name.startswith('CJK '):
                    continue
            if not name.startswith('CJK '):
                continue
            new_counter[ch] = count
        else:
            name = unicodedata.name(ch)
            name_prefix = name.split(' ')[0]
            if name_prefix in ('HANGUL', 'ARABIC', 'KATAKANA', 'HIRAGANA', 'CYRILLIC', 'THAI'):
                continue
            new_counter[ch] = count
    return new_counter


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Generate the vocab.')
    arg_parser.add_argument('--file_paths', required=True, help='Paths to the corpus files. Multiple path are separated by commas.')
    arg_parser.add_argument('--save_filepath', required=True, help='The save path of preprocessed results.')
    arg_parser.add_argument('--min_freq', type=int, default=100, help='The minimum frequency needed to include a token in the vocabulary.')
    arg_parser.add_argument('--filter_freq', type=int, default=100000, help='The minimum frequency needed to filter a token in the vocabulary.')
    args = arg_parser.parse_args()

    file_paths = args.file_paths.split(',')
    save_filepath = args.save_filepath
    min_freq = args.min_freq
    filter_freq = args.filter_freq

    counter = generate_char_counter(file_paths)
    print(f'All: {len(counter)}')
    counter = clean_char_counter(counter, min_freq=min_freq, filter_freq=filter_freq)
    print(f'Accept: {len(counter)}')

    # https://pytorch.org/text/stable/vocab.html#torchtext.vocab.vocab
    sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = collections.OrderedDict(sorted_by_freq)

    save_vocab(ordered_dict, save_filepath)

    print(f'Vocab: {ordered_dict}')
    print(f'Total: {len(ordered_dict)}')
    print(f'min_freq: {min_freq}')