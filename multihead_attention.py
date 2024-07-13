import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        
        self.query_linear = nn.Linear(input_size, input_size)
        self.key_linear = nn.Linear(input_size, input_size)
        self.value_linear = nn.Linear(input_size, input_size)
        
        self.output_linear = nn.Linear(input_size, input_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        queries = self.query_linear(x)
        keys = self.key_linear(x)
        values = self.value_linear(x)
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attention_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        
        concatenated_values = attended_values.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        output = self.output_linear(concatenated_values)
        
        return output
