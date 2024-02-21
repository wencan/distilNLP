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

    _tcn_output_size = 128
    _tcn_num_channels = [128, 128, 128, _tcn_output_size]
    _tcn_kernel_size = 2
    _tcn_dropout = 0.1

    _fc_input_size = _tcn_output_size
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

        self.tcn = TemporalConvNet(self._attention_dim, self._tcn_num_channels, self._tcn_kernel_size, self._tcn_dropout)
        self.tcn_res_block = GatedResidualBlock('avg', self._tcn_output_size)
        self.tcn_layer_norm = torch.nn.LayerNorm(self._tcn_output_size)
        self.tcn_gelu = torch.nn.GELU()
        self.tcn_dropout = torch.nn.Dropout(self._tcn_dropout)

        self.fc = torch.nn.Linear(self._fc_input_size, self._fc_output_size)

    def forward(self, features_seqs):
        out = self.embedding(features_seqs)

        out = self.attention(out)
        out = self.attention_layer_norm(out)
        out = self.attention_gelu(out)
        out = self.attention_dropout(out)
        attention_out = out.clone()

        out = out.transpose(1, 2)
        out = self.tcn(out) # (batch_size, *, max_length) -> (batch_size, *, max_length)
        out = out.transpose(1, 2)
        out = self.tcn_res_block(attention_out, out)
        out = self.tcn_layer_norm(out)
        out = self.tcn_gelu(out)
        out = self.tcn_dropout(out)

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
               ) -> Tuple[torch.tensor, Optional[torch.tensor], Sequence[int]]:
        lengths = torch.tensor([len(text) for text in texts])
        max_length = torch.max(lengths)

        # sequence length must be divisible by window size for local attention
        if max_length % self.attention_window_size != 0:
            padded_length = (max_length // self.attention_window_size + 1) * self.attention_window_size
        else:
            padded_length = max_length
        pad_length = padded_length - max_length

        features_seqs = []
        targets_seqs = []
        features_seqs = [torch.tensor([self.vocab[ch] for ch in text], device=device) for text in texts]
        features_seqs = torch.nn.utils.rnn.pad_sequence(features_seqs, batch_first=True, padding_value=self.feature_pad_value)
        if pad_length:
            features_seqs = torch.nn.functional.pad(features_seqs, (0, pad_length), value=self.feature_pad_value)

        if labels_seqs:
            targets_seqs = [torch.tensor(labels, device=device) for labels in labels_seqs]
            targets_seqs = torch.nn.utils.rnn.pad_sequence(targets_seqs, batch_first=True, padding_value=self.label_pad_value)
            if pad_length:
                targets_seqs = torch.nn.functional.pad(targets_seqs, (0, pad_length), value=self.label_pad_value)

        if labels_seqs:
            return features_seqs, targets_seqs, lengths
        return features_seqs, lengths


class Tokenizer(torch.nn.Module, Codec):
    def __init__(self, 
                 attention_implementation: Literal['local-attention', 'natten'],
                 attention_window_size:int,
                 vocab: torchtext.vocab.Vocab, 
                 embedding_weight:torch.tensor,
                 feature_pad_value=0,
                 label_pad_value:int=0,
                 ):
        torch.nn.Module.__init__(self)
        Codec.__init__(self, vocab, attention_window_size, feature_pad_value, label_pad_value)

        self.model = AttentionTCN(attention_implementation, 
                                  attention_window_size, 
                                  embedding_weight,
                                  label_pad_value,
                                  )
    
    def forward(self, texts:Sequence[str]):
        features_seqs, _ = self.Encode(texts)

        out = self.model(features_seqs)

        # return features_seqs, lengths, paded_length


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

    vocab = load_vocab(vocab_filepath, unknown_index)
    print(f'vocab size: {len(vocab)}')

    with open(embedding_filepath, 'rb') as infile:
        embedding_weight = torch.load(infile)
    embedding_weight = embedding_weight.type(torch.get_default_dtype())

    codec = Codec(vocab, attention_window_size)
    model = AttentionTCN('local-attention', attention_window_size, embedding_weight, padding_index)
    print(model)
    for name, p in model.named_parameters():
        print(f'{name} parameters: {p.numel()}')
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')
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
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # test
    outputs = model(inputs)