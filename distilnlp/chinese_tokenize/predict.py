
import argparse
import collections
from typing import Literal, Sequence

import torch
import torch.utils.data
import torchtext

from .model import AttentionTCN, Codec
from .vocab import load_vocab
from .feature import is_start_label


DEVICE = (
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class Tokenizer(torch.nn.Module):
    def __init__(self, 
                 attention_implementation: Literal['local-attention', 'natten'],
                 attention_window_size:int,
                 model_state_dict,
                 vocab_ordered_dict: collections.OrderedDict, 
                 padding_index:int=0,
                 default_index:int=1,
                 vocab_min_freq:int=100,
                 ):
        super(Tokenizer, self).__init__()

        vocab = torchtext.vocab.vocab(vocab_ordered_dict, vocab_min_freq)
        vocab.set_default_index(default_index)
        self.codec = Codec(vocab, attention_window_size, padding_index)

        embedding_weight = model_state_dict['embedding.weight']
        self.model = AttentionTCN(attention_implementation, 
                                  attention_window_size, 
                                  embedding_weight,
                                  pad_idx=padding_index
                                  )
        self.model.load_state_dict(model_state_dict)
        self.model.to(DEVICE)
    
    def forward(self, texts:Sequence[str]):
        input_ids_seqs, _ = self.codec.Encode(texts, device=DEVICE)
        logits_seqs = self.model(input_ids_seqs)
        indices_seqs = torch.argmax(logits_seqs, dim=2)

        segments_seqs = []
        for i, text in enumerate(texts):
            segments = []
            segment = []
            for j, ch in enumerate(text):
                label = indices_seqs[i][j]
                if is_start_label(label):
                    if segment:
                        segments.append(''.join(segment))
                        segment = []
                segment.append(ch)
            if segment:
                segments.append(''.join(segment))
            if segments:
                segments_seqs.append(segments)
        return segments_seqs


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='train chinese tokenize model.')
    arg_parser.add_argument('--attention_implementation', default='local-attention', help='Implementation of Attention Mechanisms. Options: local-attention and natten.')
    arg_parser.add_argument('--attention_window_size', type=int, default=3, help='Size of attention window.')
    arg_parser.add_argument('--state_dict_filepath', default='', help='The file path of the state dict of the pre-trained model. If not provided, the script will train a new model from scratch.')
    arg_parser.add_argument('--vocab_filepath', help='Path to vocab dict. If training a new model from scratch, this is required.')
    arg_parser.add_argument('--padding_index', type=int, default=0, help='Index of padding token.')
    arg_parser.add_argument('--default_index', type=int, default=1, help='Index of unknown token.')
    arg_parser.add_argument('--min_freq', type=int, default=100, help='The minimum frequency needed to include a token in the vocabulary.')

    args = arg_parser.parse_args()

    attention_implementation: Literal['local-attention', 'natten'] = args.attention_implementation
    attention_window_size = args.attention_window_size
    state_dict_filepath = args.state_dict_filepath
    vocab_filepath = args.vocab_filepath
    padding_index = args.padding_index
    default_index = args.default_index
    min_freq = args.min_freq

    ordered_dict = load_vocab(vocab_filepath)
    state_dict = torch.load(state_dict_filepath, map_location=torch.device(DEVICE))

    tokenizer = Tokenizer(attention_implementation, attention_window_size,
                          state_dict, 
                          ordered_dict, padding_index, default_index, min_freq)

    texts = ['6. 讲习班的参加者是在国家和区域应急机构和服务部门的管理岗位上工作了若干年的专业人员。', 
             'An implementation of local windowed attention for language modeling', 
             '朝散大夫右諫議大夫權御史中丞充理檢使上護軍賜紫金魚袋臣司馬光奉敕編集'
            ]
    segments_seqs = tokenizer(texts)
    for segments in segments_seqs:
        print(segments)