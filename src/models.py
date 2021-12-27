import torch
from torch import nn
torch.manual_seed(42)

class Model_v1(nn.Module):
    """concat the GRU output and categorical embeddings, feed into MLP."""
    def __init__(self, my_confg):
        super().__init__()
        self.loc_x_embedLayer=nn.Embedding(my_confg.loc_dim,my_confg.embed_loc_size)
        self.loc_y_embedLayer=nn.Embedding(my_confg.loc_dim,my_confg.embed_loc_size)
        self.time_embedLayer=nn.Embedding(my_confg.time_dim,my_confg.embed_time_size)
        self.rnn = nn.GRU(my_confg.X_dim, my_confg.num_hiddens,my_confg.num_layers,my_confg.GRU_dropout)
        concat_dim = my_confg.num_hiddens+2*my_confg.embed_loc_size+my_confg.embed_time_size
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, my_confg.l_dim),
            nn.ReLU(),
            nn.Linear(my_confg.l_dim, my_confg.output_dim)
        )

    def forward(self, X, locations,times):
        X = X.permute(1,0,2)
        X = X.to(torch.float32)
        _, state = self.rnn(X)
        loc_embedding = torch.cat((self.loc_x_embedLayer(locations[:,0]),
                    self.loc_y_embedLayer(locations[:,1])),axis=1)
        time_embedding = self.time_embedLayer(times)
        output = torch.cat((state[-1],loc_embedding,time_embedding),axis=1)
        output = self.mlp(output)
        return output


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries + keys
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1) 
        # bmm: (batch_sz, 1, num_hiddens) = (batch_sz, 1, key_size) * (batch_sz, key_size, num_hiddens)
        output = torch.bmm(self.dropout(attention_weights.unsqueeze(1)), values)
        output = output.squeeze(1) # (batch_sz, num_hiddens)
        return output


class Model_v2(nn.Module):
    """categorical embeddings as query, self attention on the GRU output."""
    def __init__(self, my_confg):
        super().__init__()
        self.loc_x_embedLayer=nn.Embedding(my_confg.loc_dim,my_confg.embed_loc_size)
        self.loc_y_embedLayer=nn.Embedding(my_confg.loc_dim,my_confg.embed_loc_size)
        self.time_embedLayer=nn.Embedding(my_confg.time_dim,my_confg.embed_time_size)
        self.rnn = nn.GRU(my_confg.X_dim, my_confg.num_hiddens,my_confg.num_layers)
        concat_dim = 2*my_confg.embed_loc_size+my_confg.embed_time_size
        self.attention = AdditiveAttention(my_confg.num_hiddens, concat_dim,
                                               my_confg.num_hiddens, my_confg.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(my_confg.num_hiddens, my_confg.l_dim),
            nn.ReLU(),
            nn.Linear(my_confg.l_dim, my_confg.output_dim)
        )

    def forward(self, X, locations,times, state=None):
        X = X.permute(1,0,2)
        X = X.to(torch.float32)
        y, _ = self.rnn(X)
        loc_embedding = torch.cat((self.loc_x_embedLayer(locations[:,0]),
                    self.loc_y_embedLayer(locations[:,1])),axis=1)
        time_embedding = self.time_embedLayer(times)
        categorical_embedding = torch.cat((loc_embedding,time_embedding),axis=1)
        key_value = y.permute(1, 0, 2)
        output = self.attention(torch.unsqueeze(categorical_embedding, dim=1),key_value,key_value)
        output = self.mlp(output)
        return output
