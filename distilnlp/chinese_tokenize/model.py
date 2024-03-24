import math
import array
from typing import Literal, Union, Sequence, Optional, Tuple

import torch
import torchtext
import torch.utils
from torch.nn.utils.parametrizations import weight_norm

from distilnlp._thirdparty.tcn import TemporalConvNet
from distilnlp.utils.residual import GatedResidualNet

from .feature import num_labels, label_pad


def new_attention(implementation: Literal['local-attention', 'natten'],
                  dim, num_heads, window_size, dropout=0.2):
    '''local attention or windowed attention'''
    if implementation == 'natten':
        # https://shi-labs.com/natten/
        # https://github.com/SHI-Labs/NATTEN
        from natten import NeighborhoodAttention1D

        attention = NeighborhoodAttention1D(dim=dim, num_heads=num_heads, kernel_size=window_size)
        return attention
    elif implementation == 'local-attention':
        # https://github.com/lucidrains/local-attention
        from distilnlp._thirdparty.local_attention import LocalAttention

        class MultiHeadLocalAttention(torch.nn.Module):
            def __init__(self, dim, num_heads, window_size, dropout=0.2, ):
                super(MultiHeadLocalAttention, self).__init__()
                self.num_heads = num_heads
                self.dim_head = dim // num_heads
                
                self.qkv = torch.nn.Linear(dim, dim*3)
                self.attention = LocalAttention(dim=self.dim_head, window_size=window_size, dropout=dropout)
            
            def forward(self, x, **kwargs): # (batch_size, max_length, dim) -> (batch_size, max_length, dim) 
                batch_size = x.size(0)
                max_length = x.size(1)

                q, k, v = self.qkv(x).chunk(3, dim=-1) # -> (batch_size, max_length, dim_head*heads), ..., ...
                q, k, v = map(lambda t: t.view(batch_size, max_length, self.num_heads, -1).transpose(1, 2), (q, k, v)) # -> (batch_size, heads, max_length, dim_head), ..., ...

                out = self.attention(q, k, v, **kwargs) # -> (batch_size, heads, max_length, dim_head) 
                out = out.transpose(1, 2).reshape(batch_size, max_length, -1) # -> (batch_size, max_length, dim_head*heads) 

                return out
            
        attention = MultiHeadLocalAttention(dim=dim, num_heads=num_heads, window_size=window_size, dropout=dropout)
    else:
        raise ValueError(f'invalid implementation: {implementation}')
    
    return attention


def FeedForward(input_size, output_size, dropout=0.2):
    return torch.nn.Sequential(
        weight_norm(torch.nn.Linear(input_size, output_size)),
        torch.nn.LayerNorm(output_size),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
    )


class TCN(TemporalConvNet):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__(num_inputs, num_channels, kernel_size, dropout)

    def forward(self, x):
        x = x.transpose(-2, -1)
        out = super(TCN, self).forward(x)
        out = out.transpose(-2, -1)
        return out


class AttentionTCN(torch.nn.Module):
    _width = 256

    _input_dim = _width
    _input_dropout = 0.1

    _attention_stack_depth = 6
    _attention_dim = _width
    _attention_num_heads = 8
    _attention_dropout = 0.1

    _tcn_stack_depth = 3
    _tcn_dim = _width
    _tcn_kernel_size = 2
    _tcn_dropout = 0.1

    _fc_output_size = num_labels
    _fc_dropout = 0.1

    def __init__(self, 
                 attention_implementation: Literal['local-attention', 'natten'],
                 attention_window_size:int,
                 embedding: torch.nn.Embedding,
                 ):
        super(AttentionTCN, self).__init__()
        self.embedding = embedding

        if self.embedding.embedding_dim != self._attention_dim:
            self.input = FeedForward(self.embedding.embedding_dim, self._input_dim, self._input_dropout)
        else:
            self.input = torch.nn.Identity()

        self.attentions = GatedResidualNet(torch.nn.Sequential(*[GatedResidualNet(torch.nn.Sequential(
            new_attention(attention_implementation, 
                          dim=self._attention_dim, 
                          num_heads=self._attention_num_heads, 
                          window_size=attention_window_size,
                          dropout=self._attention_dropout,
                          ),
            FeedForward(self._attention_dim, self._attention_dim, self._attention_dropout)
        ), 'attention', 'max', self._attention_dim) for _ in range(self._attention_stack_depth)]),
        'attentions', 'max', self._attention_dim)

        self.forward_tcn = GatedResidualNet(torch.nn.Sequential(
            TCN(self._tcn_dim, [self._tcn_dim]*self._tcn_stack_depth, self._tcn_kernel_size, self._tcn_dropout),
            FeedForward(self._tcn_dim, self._tcn_dim, self._tcn_dropout),
        ), 'tcn', 'max', self._tcn_dim)

        self.reversed_tcn = GatedResidualNet(torch.nn.Sequential(
            TCN(self._tcn_dim, [self._tcn_dim]*self._tcn_stack_depth, self._tcn_kernel_size, self._tcn_dropout),
            FeedForward(self._tcn_dim, self._tcn_dim, self._tcn_dropout),
        ), 'tcn', 'max', self._tcn_dim)

        fc_input_size = self._input_dim + self._attention_dim + self._tcn_dim*2
        fc_hidden_size = 2**math.ceil(round(math.log2(fc_input_size))) // 2
        self.fc = torch.nn.Sequential(
            FeedForward(fc_input_size, fc_hidden_size, self._fc_dropout),
            torch.nn.Sequential(
                weight_norm(torch.nn.Linear(fc_hidden_size, self._fc_output_size)),
                torch.nn.Dropout(self._fc_dropout),
            ),
        )

    @classmethod
    def from_pretrained_embedding(cls, 
                                  attention_implementation: Literal['local-attention', 'natten'],
                                  attention_window_size:int,
                                  embedding_weight:torch.tensor,
                                  padding_idx:int,
                                  freeze_embedding=True,
                                  ):
        # the tensor does not get updated in the learning process.
        embedding =  torch.nn.Embedding.from_pretrained(embedding_weight, freeze=freeze_embedding, padding_idx=padding_idx, max_norm=True)
        return cls(attention_implementation, attention_window_size, embedding)

    @classmethod
    def from_new_embedding(cls, 
                           attention_implementation: Literal['local-attention', 'natten'],
                           attention_window_size:int,
                           num_embeddings:int,
                           embedding_dim:int,
                           padding_idx:int=0,
                           ):
        embedding =  torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx, max_norm=True)
        return cls(attention_implementation, attention_window_size, embedding)

    def forward(self, features_seqs):
        out = self.embedding(features_seqs)
        out = self.input(out)
        input_out = out.clone()

        out = self.attentions(out)
        attention_out = out.clone()

        out = self.forward_tcn(out)
        forward_out = out.clone()

        out = torch.flip(out, (1, )) # reversed
        out = self.reversed_tcn(out)
        out = torch.flip(out, (1, ))

        out = torch.cat((input_out, attention_out, forward_out, out), dim=-1)
        out = self.fc(out)

        return out


class Codec:
    '''encode and decode'''
    def __init__(self, 
                 vocab:torchtext.vocab.Vocab, 
                 attention_window_size:int, 
                 feature_pad_value:int=0,
                 ):
        self.vocab = vocab
        self.attention_window_size = attention_window_size
        self.feature_pad_value = feature_pad_value
        self.label_pad_value = label_pad
    
    def encode(self, texts:Sequence[str], 
               labels_seqs:Optional[Sequence[str]]=None, 
               return_tensor:bool=True,
               device:Union[str, torch.device, int]=None,
               ) -> Tuple[torch.tensor, torch.tensor, Sequence[int]]:
        lengths = [len(text) for text in texts]
        max_length = max(lengths)

        # sequence length must be divisible by window size for local attention
        if max_length % self.attention_window_size != 0:
            padded_length = (max_length // self.attention_window_size + 1) * self.attention_window_size
        else:
            padded_length = max_length

        vocab_indices_seqs = [self.vocab.lookup_indices(list(text)) for text in texts]

        features_seqs = []
        targets_seqs = []
        features_seqs = [array.array('i', features) + array.array('i', [self.feature_pad_value]*(padded_length-lengths[idx])) for idx, features in enumerate(vocab_indices_seqs)]
        if labels_seqs:
            targets_seqs = [array.array('i', labels) + array.array('i', [self.label_pad_value]*(padded_length - lengths[idx])) for idx, labels in enumerate(labels_seqs)]

        if return_tensor:
            features_seqs = torch.tensor(features_seqs, device=device)
            if labels_seqs:
                targets_seqs = torch.tensor(targets_seqs, device=device)
            lengths = torch.tensor(lengths)

        if labels_seqs:
            return features_seqs, targets_seqs, lengths
        return features_seqs, lengths


if __name__ == '__main__':
    import argparse
    from torchinfo import summary
    from .vocab import load_vocab

    arg_parser = argparse.ArgumentParser(description='Inspect the model.')
    arg_parser.add_argument('--vocab_filepath', help='Path to vocab dict.')
    arg_parser.add_argument('--embedding_filepath', default='', help='Path to embedding weight file.')
    arg_parser.add_argument('--min_freq', type=int, default=100, help='The minimum frequency needed to include a token in the vocabulary.')
    arg_parser.add_argument('--padding_index', type=int, default=0, help='Index of padding token.')
    arg_parser.add_argument('--unknown_index', type=int, default=1, help='Index of unknown token.')
    args = arg_parser.parse_args()

    vocab_filepath = args.vocab_filepath
    padding_index = args.padding_index
    unknown_index = args.unknown_index
    embedding_filepath = args.embedding_filepath
    min_freq = args.min_freq
    attention_window_size = 3

    ordered_dict = load_vocab(vocab_filepath)
    vocab = torchtext.vocab.vocab(ordered_dict, min_freq)
    vocab.set_default_index(unknown_index)
    print(f'vocab size: {len(vocab)}')

    embedding_weight = None
    if embedding_filepath:
        with open(embedding_filepath, 'rb') as infile:
            embedding_weight = torch.load(infile)

    codec = Codec(vocab, attention_window_size)

    if embedding_weight is not None:
        model = AttentionTCN.from_pretrained_embedding('local-attention', attention_window_size, embedding_weight, padding_index)
    else:
        model = AttentionTCN.from_new_embedding('local-attention', attention_window_size, num_embeddings=len(vocab), embedding_dim=32, padding_idx=padding_index)
    inputs = ['6. 讲习班的参加者是在国家和区域应急机构和服务部门的管理岗位上工作了若干年的专业人员。', 
              'An implementation of local windowed attention for language modeling', 
              '朝散大夫右諫議大夫權御史中丞充理檢使上護軍賜紫金魚袋臣司馬光奉敕編集',
              ]
    inputs, _ = codec.encode(inputs)
    print(summary(model, input_data=inputs, depth=5, verbose=0))

    if embedding_weight is not None:
        model = AttentionTCN.from_pretrained_embedding('natten', attention_window_size, embedding_weight, padding_index)
    else:
        model = AttentionTCN.from_new_embedding('natten', attention_window_size, num_embeddings=len(vocab), embedding_dim=32, padding_idx=padding_index)
    print(summary(model, input_data=inputs, depth=5, verbose=0))