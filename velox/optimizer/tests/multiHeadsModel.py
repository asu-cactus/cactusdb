import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        assert input_dim % self.num_heads == 0
        self.projection_dim = input_dim // self.num_heads
        
        self.query_linear = nn.Linear(input_dim, self.projection_dim * num_heads)
        self.key_linear = nn.Linear(input_dim, self.projection_dim * num_heads)
        self.value_linear = nn.Linear(input_dim, self.projection_dim * num_heads)
        
        self.out_linear = nn.Linear(num_heads * self.projection_dim, input_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformation for queries, keys, and values
        Q = self.query_linear(query).view(batch_size, -1, self.num_heads, self.projection_dim).transpose(1, 2) # [batch_size, num_heads, seq_len, projection_dim]
        K = self.key_linear(key).view(batch_size, -1, self.num_heads, self.projection_dim).transpose(1, 2)     # [batch_size, num_heads, seq_len, projection_dim]
        V = self.value_linear(value).view(batch_size, -1, self.num_heads, self.projection_dim).transpose(1, 2) # [batch_size, num_heads, seq_len, projection_dim]
        
        # Compute scaled dot-product attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.projection_dim, dtype=torch.float32))
        
        if mask is not None:
            attention_scores += mask.unsqueeze(1) # Add mask
        
        attention_weights = F.softmax(attention_scores, dim=-1) # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply dropout
        attention_weights = F.dropout(attention_weights, p=0.1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V) # [batch_size, num_heads, seq_len, projection_dim]
        
        # Concatenate and linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.projection_dim) # [batch_size, seq_len, num_heads * projection_dim]
        output = self.out_linear(attention_output) # [batch_size, seq_len, input_dim]
        
        return output



# Load x data from CSV
df_x = pd.read_csv('/home/ubuntu/velox/velox/optimizer/tests/output.csv', header=None)

# Reshape data to create input tensor x
num_samples = len(df_x) // 2  # Each two lines represent one sample
x = [torch.tensor(df_x.iloc[i*2+1].values, dtype=torch.float32) for i in range(num_samples)]  # List of input samples
# x = [torch.tensor(df_x.iloc[i*2:i*2+2].values, dtype=torch.float32) for i in range(num_samples)]

# Load y data from another CSV
df_y = pd.read_csv('/home/ubuntu/velox/velox/optimizer/tests/FFNN_threads_embedding_op2_test_latency.csv', header=None)

# Create list of tensors y containing the target labels
y = [torch.tensor(df_y.iloc[i].values, dtype=torch.float32) for i in range(num_samples)]


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert data to TensorDataset
train_dataset = TensorDataset(torch.stack(x_train), torch.stack(y_train))
test_dataset = TensorDataset(torch.stack(x_test), torch.stack(y_test))

# Initialize your model, criterion, and optimizer
input_dim = 28  # Assuming each embedding vector has size 28
num_heads = 4   # Number of attention heads
head_dim = input_dim // num_heads
model = MultiHeadAttention(input_dim, num_heads, head_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_dataset:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(0), inputs.unsqueeze(0), inputs.unsqueeze(0))  # Pass the single sample as query, key, and value for simplicity
        
        # Compute scalar prediction
        scalar_prediction = torch.mean(outputs)  # Assuming you want to take the mean of all values in the output tensor
        
        loss = criterion(scalar_prediction.unsqueeze(0), labels.unsqueeze(0))  # Assuming labels is a tensor of shape (1,)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_dataset)}")

# Evaluate model on test set
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_dataset:
        outputs = model(inputs.unsqueeze(0), inputs.unsqueeze(0), inputs.unsqueeze(0))
        scalar_prediction = torch.mean(outputs)
        y_true.append(labels.item())
        y_pred.append(scalar_prediction.item())

# Convert lists to tensors
y_true_tensor = torch.tensor(y_true, dtype=torch.float32).unsqueeze(1)
y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32).unsqueeze(1)

# Compute MSE loss
mse = criterion(y_pred_tensor, y_true_tensor)
print(f"Mean Squared Error on Test Set: {mse.item()}")


