import unicodedata
import collections
import argparse
import pickle
from typing import Union

import torchtext
import tqdm

from distilnlp._utils.unicode import is_printable_symbol


def save_vocab(stoi_filepath:str, vocab: Union[torchtext.vocab.Vocab, collections.OrderedDict]):
    if isinstance(vocab, torchtext.vocab.Vocab):
        stoi = vocab.get_stoi()
    elif isinstance(vocab, collections.OrderedDict):
        stoi = vocab
    else:
        raise ValueError(f'invalid vocab type: {type(vocab)}')
    
    with open(stoi_filepath, 'w') as outfile:
        outfile.write('\n'.join(stoi.keys()))


def load_vocab(stoi_filepath:str) -> torchtext.vocab.Vocab:
    with open(stoi_filepath, 'r') as infile:
        data = infile.read()
        stoi = collections.OrderedDict([(s, i//2) for i, s in enumerate(data) if i%2==0])
    vocab = torchtext.vocab.vocab(stoi)

    return vocab


def generate_char_counter(filepaths) -> collections.Counter:
    counter = collections.Counter()
    for filepath in filepaths:
        print(filepath)
        with open(filepath) as infile:
            for line in infile:
                for ch in line:
                    counter[ch] +=1
    return counter


def reduce_char_counter(char_counter: collections.Counter,
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
    arg_parser = argparse.ArgumentParser(description='generate char counter.')
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
    counter = reduce_char_counter(counter, min_freq=min_freq, filter_freq=filter_freq)
    print(f'Accept: {len(counter)}')

    with open(save_filepath, 'wb') as outfile:
        pickle.dump(counter, outfile)