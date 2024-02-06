
import torch

class BiGRUModule(torch.nn.Module):
    _input_size=1
    _gru_hidden_size = 16
    _num_gru_layers = 2
    _out_dim = 3 # label dim
    _dropout = 0.2

    def __init__(self):
        super().__init__()

        self.gru = torch.nn.GRU(input_size=self._input_size, 
                                hidden_size=self._gru_hidden_size//2, 
                                num_layers=self._num_gru_layers, 
                                batch_first=True, 
                                bidirectional=True)
        self.dropout = torch.nn.Dropout(self._dropout)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(in_features=self._gru_hidden_size, out_features=self._out_dim)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'fc' in name:
                    torch.nn.init.xavier_normal_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)

    def forward(self, features_seqs, lengths):
        batch_size = features_seqs.size(0)
        max_length = features_seqs.size(1)

        features_seqs = features_seqs.reshape((batch_size, max_length, 1)) # -> batch_size, max_length, input_size

        out = torch.nn.utils.rnn.pack_padded_sequence(features_seqs, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(out)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # -> batch_size, max_length, gru_hidden_size
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out)
        return out