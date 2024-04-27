import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

import torch.utils.data as data
import math
import copy



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
        # return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src = src.long()  # Convert src_data to the appropriate data type
        tgt = tgt.long()  # Convert tgt_data to the appropriate data type
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    
# src_vocab_size = 10000
# tgt_vocab_size = 10000
# d_model = 512
# num_heads = 8
# num_layers = 6
# d_ff = 2048
# max_seq_length = 28
# dropout = 0.1

# transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# # Generate random sample data
# src_data = torch.randint(1, src_vocab_size, (2, max_seq_length))  # (batch_size, seq_length)
# tgt_data = torch.randint(1, tgt_vocab_size, (2, max_seq_length))  # (batch_size, seq_length)

# # Load x data from CSV
# df_x = pd.read_csv('/home/ubuntu/velox/velox/optimizer/tests/output.csv', header=None)

# # Reshape data to create input tensor x
# num_samples = len(df_x) // 2  # Each two lines represent one sample
# x = []
# for i in range(0, len(df_x), 2):
#     sample = df_x.iloc[i:i+2].values  # Get two rows as one sample
#     x.append(torch.tensor(sample, dtype=torch.float32))

# # Stack the list of tensors into a single tensor
# x_tensor = torch.stack(x)

# # Extract the 21st to 24th columns
# columns_to_normalize = x_tensor[:, :, 20:24]

# # Flatten the tensor while preserving the batch dimension
# columns_to_normalize_flat = columns_to_normalize.view(-1, 4)

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()

# # Normalize the columns
# normalized_columns_flat = scaler.fit_transform(columns_to_normalize_flat.numpy())

# # Reshape the normalized columns back to match the original shape
# normalized_columns = torch.tensor(normalized_columns_flat.reshape(-1, 2, 2), dtype=torch.float32)

# # Replace the normalized columns back into the original tensor
# x_normalized_tensor = x_tensor.clone()
# x_normalized_tensor[:, :, 20:24] = normalized_columns.view(-1, 2, 4)
# src_data = x_normalized_tensor
# # Load y data from another CSV
# df_y = pd.read_csv('/home/ubuntu/velox/velox/optimizer/tests/FFNN_threads_embedding_op2_test_latency.csv', header=None)

# # Create list of tensors y containing the target labels
# tgt_data = [torch.tensor(df_y.iloc[i].values, dtype=torch.float32) for i in range(num_samples)]
    
# Load x data from CSV
# df_x = pd.read_csv('/home/ubuntu/velox/velox/optimizer/tests/output.csv', header=None)
# num_samples = len(df_x) // 2
# # Reshape data to create input tensor x

# sequences = [torch.tensor(df_x.iloc[i:i+2].values, dtype=torch.float32) for i in range(0, len(df_x), 2)]

# # Stack the list of tensors into a single tensor
# x_tensor = torch.stack(sequences)

# # Extract the 21st to 24th columns
# columns_to_normalize = x_tensor[:, :, 20:24]

# # Flatten the tensor while preserving the batch dimension
# columns_to_normalize_flat = columns_to_normalize.view(-1, 4)

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()

# # Normalize the columns
# normalized_columns_flat = scaler.fit_transform(columns_to_normalize_flat.numpy())

# # Reshape the normalized columns back to match the original shape
# normalized_columns = torch.tensor(normalized_columns_flat.reshape(-1, 2, 4), dtype=torch.float32)

# # Replace the normalized columns back into the original tensor
# x_normalized_tensor = x_tensor.clone()
# x_normalized_tensor[:, :, 20:24] = normalized_columns.view(-1, 2, 4)
    
# Load x data from CSV
df_x = pd.read_csv('/home/ubuntu/velox/velox/optimizer/tests/output.csv', header=None)
num_samples = len(df_x) // 2  # Each two lines represent one vector
# node enigen1, token
# Reshape data to create input tensor x
x_vectors = []
for i in range(0, len(df_x), 2):
    vector = torch.tensor(df_x.iloc[i:i+2].values.flatten(), dtype=torch.float32).unsqueeze(0)
    x_vectors.append(vector)

# Stack the list of tensors into a single tensor
x_tensor = torch.cat(x_vectors, dim=0)

# Extract the 21st to 24th columns and 41st to 44th columns
columns_to_normalize_1 = x_tensor[:, 20:24]
columns_to_normalize_2 = x_tensor[:, 48:52]

# Concatenate the columns to normalize
columns_to_normalize = torch.cat((columns_to_normalize_1, columns_to_normalize_2), dim=1)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the columns
normalized_columns = scaler.fit_transform(columns_to_normalize.numpy())

# Replace the normalized columns back into the original tensor
x_normalized_tensor = x_tensor.clone()
x_normalized_tensor[:, 20:24] = torch.tensor(normalized_columns[:, :4], dtype=torch.float32)
x_normalized_tensor[:, 48:52] = torch.tensor(normalized_columns[:, 4:], dtype=torch.float32)

# Load y data from another CSV
df_y = pd.read_csv('/home/ubuntu/velox/velox/optimizer/tests/FFNN_threads_embedding_op2_test_latency.csv', header=None)

# Create list of tensors y containing the target labels
y = [torch.tensor(df_y.iloc[i].values, dtype=torch.float32) for i in range(num_samples)]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_normalized_tensor, y, test_size=0.2, random_state=42)



# Define model parameters
src_vocab_size = 100000
tgt_vocab_size = 1000  # For regression, output size is 1 (latency)
d_model = 64
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 56
dropout = 0.1
num_epochs = 10
learning_rate = 0.001

# Initialize Transformer model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)


# Convert data to TensorDataset
train_dataset = TensorDataset(x_train, torch.stack(y_train))
test_dataset = TensorDataset(x_test, torch.stack(y_test))

# Training loop
for epoch in range(num_epochs):
    transformer.train()
    running_loss = 0.0
    
    for src_data, tgt_data in train_dataset:
        optimizer.zero_grad()
        output = transformer(src_data.unsqueeze(0), tgt_data.unsqueeze(0))
        loss = criterion(output.squeeze(), tgt_data)  # Assuming tgt_data is a single scalar value (latency)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_dataset)}")

# Evaluation on test set
transformer.eval()
with torch.no_grad():
    test_loss = 0.0
    for src_data, tgt_data in test_dataset:
        output = transformer(src_data.unsqueeze(0), tgt_data.unsqueeze(0))
        test_loss += criterion(output.squeeze(), tgt_data).item()

    print(f"Test Loss: {test_loss / len(test_dataset)}")
