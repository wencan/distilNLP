import collections
from typing import Literal, Union, Sequence, Optional, Tuple

import torch
import torchtext
import torch.utils

from distilnlp._thirdparty.tcn import TemporalConvNet
from distilnlp._utils.residual import GatedResidualBlock

from .feature import num_labels, label_pad


def new_attention(implementation: Literal['local-attention', 'natten'],
                  dim, num_heads, window_size):
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
            def __init__(self, dim, num_heads, window_size, **kwargs):
                super(MultiHeadLocalAttention, self).__init__()
                self.num_heads = num_heads
                self.dim_head = dim // num_heads
                
                self.qkv = torch.nn.Linear(dim, dim*3)
                self.attention = LocalAttention(dim=self.dim_head, window_size=window_size, **kwargs)
            
            def forward(self, x, **kwargs): # (batch_size, max_length, dim) -> (batch_size, max_length, dim) 
                batch_size = x.size(0)
                max_length = x.size(1)

                q, k, v = self.qkv(x).chunk(3, dim=-1) # -> (batch_size, max_length, dim_head*heads), ..., ...
                q, k, v = map(lambda t: t.view(batch_size, max_length, self.num_heads, -1).transpose(1, 2), (q, k, v)) # -> (batch_size, heads, max_length, dim_head), ..., ...

                out = self.attention(q, k, v, **kwargs) # -> (batch_size, heads, max_length, dim_head) 
                out = out.transpose(1, 2).reshape(batch_size, max_length, -1) # -> (batch_size, max_length, dim_head*heads) 

                return out
            
        attention = MultiHeadLocalAttention(dim=dim, num_heads=num_heads, window_size=window_size)
    else:
        raise ValueError(f'invalid implementation: {implementation}')
    
    return attention

class AttentionTCN(torch.nn.Module):    
    _attention_num_heads = 8
    _attention_dropout = 0.1

    _tcn0_output_size = 128
    _tcn0_num_channels = [128, _tcn0_output_size]
    _tcn0_kernel_size = 2
    _tcn0_dropout = 0.1

    # inverted
    _tcn1_input_size = _tcn0_output_size
    _tcn1_output_size = 128
    _tcn1_num_channels = [128, _tcn0_output_size]
    _tcn1_kernel_size = 2
    _tcn1_dropout = 0.1

    _pool_output_size = _tcn1_output_size // 2
    _fc_input_size = _pool_output_size
    _fc_output_size = num_labels

    def __init__(self, 
                 attention_implementation: Literal['local-attention', 'natten'],
                 attention_window_size:int,
                 embedding_weight:torch.tensor,
                 pad_idx:int,
                 ):
        super(AttentionTCN, self).__init__()

        # the tensor does not get updated in the learning process.
        self.embedding =  torch.nn.Embedding.from_pretrained(embedding_weight, freeze=True, padding_idx=pad_idx, max_norm=True)

        self._attention_dim = self.embedding.embedding_dim
        self.attention = new_attention(attention_implementation, 
                                       dim=self._attention_dim, 
                                       num_heads=self._attention_num_heads, 
                                       window_size=attention_window_size,
                                      )
        self.attention_layer_norm = torch.nn.LayerNorm(self._attention_dim)
        self.attention_gelu = torch.nn.GELU()
        self.attention_dropout = torch.nn.Dropout(self._attention_dropout)

        self._tcn0_input_size = self._attention_dim
        self.tcn0 = TemporalConvNet(self._tcn0_input_size, self._tcn0_num_channels, self._tcn0_kernel_size, self._tcn0_dropout)
        self.tcn0_res_block = GatedResidualBlock('max', self._tcn0_output_size)
        self.tcn0_layer_norm = torch.nn.LayerNorm(self._tcn0_output_size)
        self.tcn0_gelu = torch.nn.GELU()
        self.tcn0_dropout = torch.nn.Dropout(self._tcn0_dropout)

        self._tcn1_input_size = self._attention_dim + self._tcn0_output_size
        self.tcn1 = TemporalConvNet(self._tcn1_input_size, self._tcn1_num_channels, self._tcn1_kernel_size, self._tcn1_dropout)
        self.tcn1_res_block = GatedResidualBlock('max', self._tcn1_output_size)
        self.tcn1_layer_norm = torch.nn.LayerNorm(self._tcn1_output_size)
        self.tcn1_gelu = torch.nn.GELU()
        self.tcn1_dropout = torch.nn.Dropout(self._tcn1_dropout)

        self.pool = torch.nn.AdaptiveMaxPool1d(self._pool_output_size)
        self.fc = torch.nn.Linear(self._fc_input_size, self._fc_output_size)

    def forward(self, features_seqs):
        out = self.embedding(features_seqs)

        out = self.attention(out)
        out = self.attention_layer_norm(out)
        out = self.attention_gelu(out)
        out = self.attention_dropout(out)
        attention_out = out.clone()

        out = out.transpose(1, 2)
        out = self.tcn0(out) # (batch_size, *, max_length) -> (batch_size, *, max_length)
        out = out.transpose(1, 2)
        out = self.tcn0_res_block(attention_out, out)
        out = self.tcn0_layer_norm(out)
        out = self.tcn0_gelu(out)
        out = self.tcn0_dropout(out)
        tcn0_out = out.clone()

        inverted_attention_out = torch.flip(attention_out, (1, )) # inverted
        out = torch.cat((inverted_attention_out, out), dim=2)
        out = out.transpose(1, 2)
        out = self.tcn1(out) # (batch_size, *, max_length) -> (batch_size, *, max_length)
        out = out.transpose(1, 2)
        out = self.tcn1_res_block(tcn0_out, out)
        out = self.tcn1_layer_norm(out)
        out = self.tcn1_gelu(out)
        out = self.tcn1_dropout(out)

        out = self.pool(out)
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
    
    def Encode(self, texts:Sequence[str], 
               labels_seqs:Optional[Sequence[str]]=None, 
               device:Union[str, torch.device, int]=None,
               ) -> Union[Tuple[torch.tensor, torch.tensor, Sequence[int]], Tuple[torch.tensor, Sequence[int]]]:
        lengths = torch.tensor([len(text) for text in texts])
        max_length = torch.max(lengths)

        # sequence length must be divisible by window size for local attention
        if max_length % self.attention_window_size != 0:
            padded_length = (max_length // self.attention_window_size + 1) * self.attention_window_size
        else:
            padded_length = max_length

        features_seqs = []
        targets_seqs = []
        features_seqs = [self.vocab.lookup_indices(list(text)) for text in texts]
        features_seqs = [features + [self.feature_pad_value]*(padded_length - lengths[idx]) for idx, features in enumerate(features_seqs)]
        features_seqs = torch.tensor(features_seqs, device=device) # Optimized for memory allocation
        if labels_seqs:
            targets_seqs = [labels + [self.label_pad_value]*(padded_length - lengths[idx]) for idx, labels in enumerate(labels_seqs)]
            targets_seqs = torch.tensor(targets_seqs, device=device) # Optimized for memory allocation

        if labels_seqs:
            return features_seqs, targets_seqs, lengths
        return features_seqs, lengths


if __name__ == '__main__':
    import argparse
    from .vocab import load_vocab

    arg_parser = argparse.ArgumentParser(description='Inspect the model.')
    arg_parser.add_argument('--vocab_filepath', help='Path to vocab dict.')
    arg_parser.add_argument('--embedding_filepath', help='Path to embedding weight file.')
    arg_parser.add_argument('--padding_index', type=int, default=0, help='Index of padding token.')
    arg_parser.add_argument('--unknown_index', type=int, default=1, help='Index of unknown token.')
    args = arg_parser.parse_args()

    vocab_filepath = args.vocab_filepath
    padding_index = args.padding_index
    unknown_index = args.unknown_index
    embedding_filepath = args.embedding_filepath
    attention_window_size = 3

    ordered_dict = load_vocab(vocab_filepath)
    vocab = torchtext.vocab.vocab(ordered_dict)
    vocab.set_default_index(unknown_index)
    print(f'vocab size: {len(vocab)}')

    with open(embedding_filepath, 'rb') as infile:
        embedding_weight = torch.load(infile)

    codec = Codec(vocab, attention_window_size)
    model = AttentionTCN('local-attention', attention_window_size, embedding_weight, padding_index)
    print(model)
    for name, p in model.named_parameters():
        print(f'{name} parameters: {p.numel()}')
    print(f'Total number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')
    # test
    inputs = ['6. 讲习班的参加者是在国家和区域应急机构和服务部门的管理岗位上工作了若干年的专业人员。', 
              'An implementation of local windowed attention for language modeling', 
              '朝散大夫右諫議大夫權御史中丞充理檢使上護軍賜紫金魚袋臣司馬光奉敕編集'
              ]
    inputs, _ = codec.Encode(inputs)
    outputs = model(inputs)

    model = AttentionTCN('natten', attention_window_size, embedding_weight, padding_index)
    print(model)
    for name, p in model.named_parameters():
        print(f'{name} parameters: {p.numel()}')
    print(f'Total number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # test
    outputs = model(inputs)