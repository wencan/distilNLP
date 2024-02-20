import collections
from typing import Literal, Union, Sequence

import torch
import torchtext

from distilnlp._thirdparty.tcn import TemporalConvNet
from distilnlp._utils.residual import GatedResidualBlock

from .feature import num_labels


def new_attention(implementation: Literal['local-attention', 'natten'],
                  dim, num_heads, window_size):
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
                self.heads = num_heads
                self.dim_head = dim // num_heads
                
                self.qkv = torch.nn.Linear(dim, self.dim_head*num_heads*3)
                self.attention = LocalAttention(dim=self.dim_head, window_size=window_size, **kwargs)
            
            def forward(self, x, **kwargs): # (batch_size, max_length, dim) -> (batch_size, max_length, dim) 
                batch_size = x.size(0)
                max_length = x.size(1)

                q, k, v = self.qkv(x).chunk(3, dim=-1) # -> (batch_size, max_length, dim_head*heads), ..., ...
                q, k, v = map(lambda t: t.view(batch_size, max_length, self.heads, -1).transpose(1, 2), (q, k, v)) # -> (batch_size, heads, max_length, dim_head), ..., ...

                out = self.attention(q, k, v, **kwargs) # -> (batch_size, heads, max_length, dim_head) 
                out = out.transpose(1, 2).reshape(batch_size, max_length, -1) # -> (batch_size, max_length, dim_head*heads) 
                return out
            
        attention = MultiHeadLocalAttention(dim=dim, num_heads=num_heads, window_size=window_size)
    else:
        raise ValueError(f'invalid implementation: {implementation}')
    
    return attention
    

class AttentionTCN(torch.nn.Module):    
    _attention_window_size = 3
    _attention_num_heads = 8
    _attention_output_dim = _attention_num_heads * 16
    _attention_dropout = 0.1

    _tcn_input_size = _attention_output_dim
    _tcn_output_size = 128
    _tcn_num_channels = [_tcn_input_size, 128, 128, _tcn_output_size]
    _tcn_kernel_size = 2
    _tcn_dropout = 0.1

    _fc_input_size = _tcn_output_size
    _fc_output_size = num_labels

    def __init__(self, 
                 attention_implementation: Literal['local-attention', 'natten'],
                 vocab: Union[torchtext.vocab.Vocab, str], 
                 pad_idx:int,
                 unk_idx:int,
                 embedding_weight:torch.tensor,
                 ):
        super(AttentionTCN, self).__init__()
        self.pad_idx = pad_idx

        if isinstance(vocab, torchtext.vocab.Vocab):
            self.vocab = vocab
        elif isinstance(vocab, collections.OrderedDict):
            self.vocab = torchtext.vocab.vocab(vocab)
        self.vocab.set_default_index(unk_idx)
        self.embedding =  torch.nn.Embedding.from_pretrained(embedding_weight, padding_idx=pad_idx, max_norm=True)

        self.attention = new_attention(attention_implementation, 
                                       dim=self.embedding.embedding_dim, 
                                       num_heads=self._attention_num_heads, 
                                       window_size=self._attention_window_size
                                      )
        self.attention_res_block = GatedResidualBlock('max', self._attention_output_dim)
        self.attention_layer_norm = torch.nn.LayerNorm(self._attention_output_dim)
        self.attention_gelu = torch.nn.GELU()
        self.attention_dropout = torch.nn.Dropout(self._attention_dropout)

        self.tcn = TemporalConvNet(self._tcn_input_size, self._tcn_num_channels, self._tcn_kernel_size, self._tcn_dropout)
        self.tcn_res_block = GatedResidualBlock('avg', self._tcn_output_size)
        self.tcn_layer_norm = torch.nn.LayerNorm(self._tcn_output_size)
        self.tcn_gelu = torch.nn.GELU()
        self.tcn_dropout = torch.nn.Dropout(self._tcn_dropout)

        self.fc = torch.nn.Linear(self._fc_input_size, self._fc_output_size)

    def forward(self, features_seqs):
        out = self.embedding(features_seqs)
        embed_out = out.clone()

        out = self.attention(out)
        out = self.attention_res_block(embed_out, out)
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
    
    def encode(self, texts:Union[str, Sequence[str]]):
        if isinstance(texts, str):
            texts = [texts]
        lengths = [len(text) for text in texts]
        max_length = max(lengths)

        # sequence length must be divisible by window size for local attention
        if max_length % self._attention_window_size != 0:
            max_length = (max_length // self._attention_window_size + 1) * self._attention_window_size

        features_seqs = []
        for text in texts:
            features = [self.vocab[ch] for ch in text]
            length = len(features)
            if length < max_length:
                features += [self.pad_idx] * (max_length - length)
            features_seqs.append(features)
        features_seqs = torch.tensor(features_seqs)

        outputs = self.forward(features_seqs)
        return outputs


if __name__ == '__main__':
    import argparse
    from .vocab import load_vocab

    arg_parser = argparse.ArgumentParser(description='Inspect the model.')
    arg_parser.add_argument('--vocab_filepath', help='Path to vocab dict.')
    arg_parser.add_argument('--padding_index', type=int, default=0, help='Index of padding token.')
    arg_parser.add_argument('--unknown_index', type=int, default=1, help='Index of unknown token.')
    arg_parser.add_argument('--embedding_filepath', help='Path to embedding weight file.')
    args = arg_parser.parse_args()

    vocab_filepath = args.vocab_filepath
    padding_index = args.padding_index
    unknown_index = args.unknown_index
    embedding_filepath = args.embedding_filepath

    vocab = load_vocab(vocab_filepath)
    print(f'vocab size: {len(vocab)}')

    with open(embedding_filepath, 'rb') as infile:
        embedding_weight = torch.load(infile)
    embedding_weight = embedding_weight.type(torch.get_default_dtype())

    model = AttentionTCN('local-attention', vocab, padding_index, unknown_index, embedding_weight)
    print(model)
    for name, p in model.named_parameters():
        print(f'{name} parameters: {p.numel()}')
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # test
    inputs = ['6. 讲习班的参加者是在国家和区域应急机构和服务部门的管理岗位上工作了若干年的专业人员。', 
              'An implementation of local windowed attention for language modeling', 
              '朝散大夫右諫議大夫權御史中丞充理檢使上護軍賜紫金魚袋臣司馬光奉敕編集'
              ]
    outputs = model.encode(inputs)

    model = AttentionTCN('natten', vocab, padding_index, unknown_index, embedding_weight)
    print(model)
    for name, p in model.named_parameters():
        print(f'{name} parameters: {p.numel()}')
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # test
    outputs = model.encode(inputs)