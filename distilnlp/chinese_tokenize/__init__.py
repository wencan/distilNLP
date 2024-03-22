import collections
from typing import Sequence

import torch

from distilnlp.utils.modelfile import downloaded_model_filepath
from .predict import Tokenizer
from .vocab import load_vocab

__all__ = ('chinese_tokenize')


DEVICE = (
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def chinese_tokenize(texts: Sequence[str]):
    model_name = 'chinese_tokenize'
    commit_version = 'a4989493cb8cd2bd14d725e13557e9f7b3760c2c'
    state_dict_version = '20240322093302'
    state_dict_url = f'https://raw.githubusercontent.com/wencan/distilNLP/{commit_version}/assets/model/chinese_tokenize/state_dict_{state_dict_version}.pt'
    state_dict_filepath = downloaded_model_filepath(model_name, state_dict_version, state_dict_url, content_type='state_dict', postfix='pt')
    vocab_version = '20240322093302'
    vocab_url = f'https://raw.githubusercontent.com/wencan/distilNLP/{commit_version}/assets/model/chinese_tokenize/vocab_{vocab_version}.txt'
    vocab_filepath = downloaded_model_filepath(model_name, vocab_version, vocab_url, content_type='vocab', postfix='txt')

    vocab_ordered_dict = load_vocab(vocab_filepath)
    model_state_dict = torch.load(state_dict_filepath, map_location=torch.device(DEVICE))

    tokenizer = Tokenizer(attention_implementation='local-attention',
                          attention_window_size=3,
                          model_state_dict=model_state_dict,
                          vocab_ordered_dict=vocab_ordered_dict
                          )
    return tokenizer(texts)