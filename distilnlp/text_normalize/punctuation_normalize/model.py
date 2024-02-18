import torch

from .feature import num_labels
from distilnlp._utils.residual import GatedResidualBlock


class ConvBiGRU(torch.nn.Module):
    _fa_out_features = 8
    _conv_out_channels = 8
    _pool_input_size = _fa_out_features + _conv_out_channels * 3
    _dropout_1 = 0.2
    _gru_1_input_size = 8
    _gru_1_hidden_size = 32
    _dropout_2 = 0.35
    _gru_2_input_size = _gru_1_hidden_size
    _gru_2_hidden_size = 32
    _dropout_3 = 0.5
    _out_dim = num_labels

    def __init__(self):
        super(ConvBiGRU, self).__init__()
        self.fa = torch.nn.Linear(in_features=1, out_features=self._fa_out_features) # features affine layer
        self.conv_2 = torch.nn.Conv1d(in_channels=1,
                                      out_channels=self._conv_out_channels,
                                      kernel_size=2,
                                      padding=0)
        self.conv_3 = torch.nn.Conv1d(in_channels=1,
                                      out_channels=self._conv_out_channels,
                                      kernel_size=3,
                                      padding=1) # keep dim of features
        self.conv_4 = torch.nn.Conv1d(in_channels=1,
                                      out_channels=self._conv_out_channels,
                                      kernel_size=4,
                                      padding=1)
        self.pool = torch.nn.AdaptiveMaxPool1d(output_size=self._gru_1_input_size)
        self.layer_norm_1 = torch.nn.LayerNorm(self._gru_1_input_size)
        self.dropout_1 = torch.nn.Dropout(self._dropout_1)

        self.gru_1 = torch.nn.GRU(input_size=self._gru_1_input_size,
                                hidden_size=self._gru_1_hidden_size//2,
                                batch_first=True,
                                bidirectional=True)
        self.res_block_2 = GatedResidualBlock('avg', self._gru_1_hidden_size) # inputs: pool outputs, gru_1 outputs
        self.layer_norm_2 = torch.nn.LayerNorm(self._gru_1_hidden_size)
        self.dropout_2 = torch.nn.Dropout(self._dropout_2)

        self.gru_2 = torch.nn.GRU(input_size=self._gru_2_input_size,
                                hidden_size=self._gru_2_hidden_size//2,
                                batch_first=True,
                                bidirectional=True)
        self.res_block_3 = GatedResidualBlock('avg', self._gru_2_hidden_size)    # inputs: gru_1 outputs, gru_2 outputs
        self.layer_norm_3 = torch.nn.LayerNorm(self._gru_2_hidden_size)
        self.dropout_3 = torch.nn.Dropout(self._dropout_3)

        self.fc = torch.nn.Linear(in_features=self._gru_2_hidden_size, out_features=self._out_dim) # Fully Connected Layer

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'norm' in name:
                pass
            elif 'res_block' in name:
                pass
            elif 'weight' in name:
                torch.nn.init.xavier_normal_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)

    def forward(self, features_seqs, lengths):
        batch_size = features_seqs.size(0)
        max_length = features_seqs.size(1)
        inputs = features_seqs.view((batch_size, max_length, 1))

        fa_features_seqs = self.fa(inputs)
        fa_features_seqs = fa_features_seqs.view((batch_size, self._fa_out_features, max_length))

        conv_inputs = inputs.view((batch_size, 1, max_length))
        conv2_features_seqs = self.conv_2(conv_inputs)
        conv3_features_seqs = self.conv_3(conv_inputs)
        conv4_features_seqs = self.conv_4(conv_inputs)
        conv2_features_seqs = torch.nn.functional.pad(conv2_features_seqs, (0, max_length-conv2_features_seqs.size(2)))
        conv4_features_seqs = torch.nn.functional.pad(conv4_features_seqs, (0, max_length-conv4_features_seqs.size(2)))
        assert max_length == conv2_features_seqs.size(2)
        assert max_length == conv3_features_seqs.size(2)
        assert max_length == conv3_features_seqs.size(2)

        out = torch.cat((fa_features_seqs, conv2_features_seqs, conv3_features_seqs, conv4_features_seqs), dim=1) # -> (batch_size, _pool_input_size, max_length)
        out = torch.transpose(out, 1, 2) # -> (batch_size, max_length, _pool_input_size)
        out = self.pool(out) # (batch_size, max_length, _pool_input_size) -> (batch_size, max_length, _gru_input_size)
        out = self.layer_norm_1(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout_1(out)
        pool_out = torch.clone(out)

        out = torch.nn.utils.rnn.pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru_1(out)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # -> (batch_size, max_length, gru_hidden_size)
        out = self.res_block_2(pool_out, out)
        out = self.layer_norm_2(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout_2(out)
        gru_1_out = torch.clone(out)

        out = torch.nn.utils.rnn.pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru_2(out)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # -> (batch_size, max_length, gru_hidden_size)
        out = self.res_block_3(gru_1_out, out)
        out = self.layer_norm_3(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout_3(out)

        out = self.fc(out)
        return out


if __name__ == '__main__':
    model = ConvBiGRU()
    print(model)
    for name, p in model.named_parameters():
        print(f'{name} parameters: {p.numel()}')
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # test
    import random
    batch_size = 1024
    max_length = random.randint(10, 512)
    inputs = torch.randn((batch_size, max_length, 1))
    lengths = torch.tensor(max_length).repeat(batch_size)
    model(inputs, lengths)