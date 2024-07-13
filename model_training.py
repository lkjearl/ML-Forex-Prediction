import torch
import torch.nn as nn

class GRUTFTModel(nn.Module):
    def __init__(self, input_size, output_size, gr_units=32, num_layers=2, gr_dropout=0.2, tft_units=32, num_heads=8, dense_units=16, dropout=0.2):
        super(GRUTFTModel, self).__init__()
        self.gru = nn.GRU(input_size, gr_units, num_layers=num_layers, batch_first=True, dropout=gr_dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=tft_units, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(gr_units + tft_units, dense_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_out, _ = self.multihead_attn(gru_out, gru_out, gru_out)
        combined = torch.cat((gru_out, attn_out), dim=-1)
        output = self.fc1(combined)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output[:, -1, :]
        return output
