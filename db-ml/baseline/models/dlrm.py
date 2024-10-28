import torch 
import torch.nn as nn

class DLRM(nn.Module):
    def __init__(self, embedding_dim, num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes, top_mlp_sizes):
        super(DLRM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_numerical_features = num_numerical_features
        self.categorical_feature_sizes = categorical_feature_sizes
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim) 
            for num_embeddings in categorical_feature_sizes
        ])
        
        # Bottom MLP for numerical features
        bottom_mlp_layers = []
        last_size = num_numerical_features
        for size in bottom_mlp_sizes:
            bottom_mlp_layers.extend([
                nn.Linear(last_size, size),
                nn.ReLU(),
                # nn.BatchNorm1d(size)
            ])
            last_size = size
        self.bottom_mlp = nn.Sequential(*bottom_mlp_layers)
        
        # Top MLP
        # Interaction is performed by concatenating bottom_llm and categorical embeddings
        # Input to top MLP is concatenation of: bottom_mlp and interaction outputs
        # num_interaction_features = ((len(categorical_feature_sizes)+1) * len(categorical_feature_sizes)) // 2
        # num_concatenated_features = num_interaction_features + embedding_dim
        
        top_mlp_layers = []
        last_size = embedding_dim*(len(categorical_feature_sizes)+1)
        for size in top_mlp_sizes:
            top_mlp_layers.extend([
                nn.Linear(last_size, size),
                nn.ReLU(),
                # nn.BatchNorm1d(size)
            ])
            last_size = size
        top_mlp_layers.append(nn.Linear(last_size, 1))
        self.top_mlp = nn.Sequential(*top_mlp_layers)
        
    def forward(self, numerical_features, categorical_features):
        # Process numerical features
        numerical_output = self.bottom_mlp(numerical_features)
        
        # Process categorical features
        embedded_categorical = [embedding(categorical_features[:, i]) 
                                for i, embedding in enumerate(self.embeddings)]
        # # Interaction Inputs
        # interaction_inputs = [numerical_output] + embedded_categorical
        
        # # Feature interaction
        # interactions = []
        # for i in range(len(interaction_inputs)):
        #     for j in range(i + 1, len(interaction_inputs)):
        #         interactions.append(torch.sum(interaction_inputs[i] * interaction_inputs[j], dim=1, keepdim=True))
        
        # interaction_tensor = torch.cat(interactions, dim=1)
        
        # Concatenate all features
        concatenated = torch.cat([
            numerical_output,
            torch.cat(embedded_categorical, dim=1),
            # interaction_tensor
        ], dim=1)
        # Top MLP
        output = self.top_mlp(concatenated)
        
        return torch.sigmoid(output)
    