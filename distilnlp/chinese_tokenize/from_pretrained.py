import collections
import pickle
import argparse

import transformers
import torch
from sklearn.decomposition import IncrementalPCA

from .vocab import save_vocab

def decomposition_weight(weight: torch.tensor, new_dim:int, batch_size:int = 256):
    pca = IncrementalPCA(n_components=new_dim, batch_size=batch_size)
    new_weight = pca.fit_transform(weight)
    return torch.tensor(new_weight)


def embedding_from_pretained(char_counter: collections.Counter, 
                             model_name:str,
                             tokenizer_class = transformers.AutoTokenizer,
                             pretraining_class = transformers.AutoModelForPreTraining,
                             pad_token:str = '[PAD]', unk_token:str = '[UNK]',
                             allow_decomposition:bool = True) -> tuple[collections.OrderedDict, torch.Tensor]:
    tokenizer = tokenizer_class.from_pretrained(model_name)
    stoi = collections.OrderedDict()
    for s, i in tokenizer.vocab.items():
        if s in (pad_token, unk_token):
            stoi[s] = i
        elif len(s) == 1 and s in char_counter:
            stoi[s] = i

    assert pad_token in stoi
    assert unk_token in stoi

    model = pretraining_class.from_pretrained(model_name)
    weight = model.electra.embeddings.word_embeddings.weight.data
    new_weight = weight[list(stoi.values())[:]]

    # rebuild index
    for idx, s in enumerate(stoi.keys()):
        stoi[s] = idx
    
    # decomposition
    if allow_decomposition:
        rate = round(weight.size(0) / new_weight.size(0))
        if rate > 1:
            rate = round(rate/2)*2
            new_dim = weight.size(1) // rate
            new_weight = decomposition_weight(new_weight, new_dim)

    return stoi, new_weight


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Extracting Vocab and Embedding from Pre-trained Models.')
    arg_parser.add_argument('--char_counter_filepath', required=True, help='Path to char counter file.')
    arg_parser.add_argument('--vocab_filepath', required=True, help='Path to save vocab dict.')
    arg_parser.add_argument('--embedding_filepath', required=True, help='Path to save embedding weight.')
    args = arg_parser.parse_args()

    char_counter_filepath = args.char_counter_filepath
    vocab_filepath = args.vocab_filepath
    embedding_filepath = args.embedding_filepath
    pretained_model = 'hfl/chinese-electra-180g-small-discriminator' # no space symbol!!!
    tokenizer_class = transformers.ElectraTokenizer
    pretraining_class = transformers.ElectraForPreTraining
    pad_token:str = '[PAD]'
    unk_token:str = '[UNK]'

    with open(char_counter_filepath, 'rb') as infile:
        char_counter = pickle.load(infile)
    
    stoi, weight = embedding_from_pretained(char_counter, 
                                            pretained_model,
                                            tokenizer_class,
                                            pretraining_class)
    
    save_vocab(vocab_filepath, stoi)
    torch.save(weight, embedding_filepath)

    print(stoi)
    print(weight)
    print(f'vocab size: {len(stoi)}')
    print(f'pad index: {stoi[pad_token]}')
    print(f'unk index: {stoi[unk_token]}')
    print(f'weight size: {weight.size()}')