import collections
import pickle
import argparse

import transformers
import torch
from sklearn.decomposition import IncrementalPCA

from .vocab import save_vocab, load_vocab

def decomposition_weight(weight: torch.tensor, new_dim:int, batch_size:int = 256):
    pca = IncrementalPCA(n_components=new_dim, batch_size=batch_size)
    new_weight = pca.fit_transform(weight)
    return torch.tensor(new_weight)


def embedding_from_pretained(ordered_dict: collections.OrderedDict, 
                             model_name:str,
                             tokenizer_class = transformers.AutoTokenizer,
                             pretraining_class = transformers.AutoModelForPreTraining,
                             pad_token:str = '[PAD]', unk_token:str = '[UNK]',
                             allow_decomposition:bool = True) -> tuple[collections.OrderedDict, torch.Tensor]:
    pretrained_tokenizer = tokenizer_class.from_pretrained(model_name)
    pretrained_vocab = pretrained_tokenizer.get_vocab()
    pretrained_model = pretraining_class.from_pretrained(model_name)
    if isinstance(pretrained_model, transformers.ElectraForPreTraining):
        pretrained_weight = pretrained_model.electra.embeddings.word_embeddings.weight.data
    else:
        raise NotImplementedError(f'{type(pretrained_model)} is not supported')

    filtered_dict = collections.OrderedDict()
    filtered_weight = []
    filtered_dict[pad_token] = 0 # assigned later
    filtered_weight.append(pretrained_weight[pretrained_vocab[pad_token]])
    filtered_dict[unk_token] = 0 # assigned later
    filtered_weight.append(pretrained_weight[pretrained_vocab[unk_token]])
    for s, i in ordered_dict.items():
        try:
            idx = pretrained_vocab[s]
            filtered_dict[s] = i    # origin freq
        except KeyError:
            pass
        else:
            filtered_weight.append(pretrained_weight[idx])
    filtered_weight = torch.stack(filtered_weight)
    
    # decomposition
    if allow_decomposition:
        rate = round(pretrained_weight.size(0) / filtered_weight.size(0))
        if rate > 1:
            rate = round(rate/2)*2
            new_dim = pretrained_weight.size(1) // rate
            filtered_weight = decomposition_weight(filtered_weight, new_dim)

    return filtered_dict, filtered_weight


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Extracting Vocab and Embedding from Pre-trained Models.')
    arg_parser.add_argument('--vocab_filepath', required=True, help='Path to the vocab.')
    arg_parser.add_argument('--new_vocab_filepath', required=True, help='Path to save the new vocab.')
    arg_parser.add_argument('--embedding_filepath', required=True, help='Path to save the embedding weight.')
    arg_parser.add_argument('--min_freq', type=int, default=100, help='The minimum frequency needed to include a token in the vocabulary.')
    args = arg_parser.parse_args()

    vocab_filepath = args.vocab_filepath
    new_vocab_filepath = args.new_vocab_filepath
    embedding_filepath = args.embedding_filepath
    min_freq = args.min_freq
    pretrained_model = 'hfl/chinese-electra-180g-small-discriminator' # no space symbol!!!
    tokenizer_class = transformers.ElectraTokenizer
    pretraining_class = transformers.ElectraForPreTraining
    pad_token:str = '[PAD]'
    unk_token:str = '[UNK]'

    ordered_dict = load_vocab(vocab_filepath)
    print(ordered_dict)
    
    ordered_dict, embedding_weight = embedding_from_pretained(ordered_dict, 
                                                              pretrained_model,
                                                              tokenizer_class,
                                                              pretraining_class
                                                              )
    ordered_dict[pad_token] = min_freq
    ordered_dict[unk_token] = min_freq
    tokens = list(ordered_dict.keys())
    default_index = tokens.index(unk_token)
    padding_index = tokens.index(pad_token)

    save_vocab(ordered_dict, new_vocab_filepath)
    torch.save(embedding_weight, embedding_filepath)

    print(f'vocab: {ordered_dict}')
    print(f'vocab size: {len(ordered_dict)}')
    print(f'weight size: {embedding_weight.size()}')
    print(f'padding index: {padding_index}')
    print(f'default index: {default_index}')