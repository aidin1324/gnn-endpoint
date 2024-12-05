import torch
from torch import nn, optim, Tensor

from collections import defaultdict

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul

num_users = 610
num_movies = 9724

def convert_r_mat_edge_index_to_adj_mat_edge_index(input_edge_index, input_edge_values):
    R = torch.zeros((num_users, num_movies))
    for i in range(len(input_edge_index[0])):
        row_idx = input_edge_index[0][i]
        col_idx = input_edge_index[1][i]
        R[row_idx][col_idx] = input_edge_values[i] # assign actual edge value to Interaction Matrix

    R_transpose = torch.transpose(R, 0, 1)
    
    # create adj_matrix
    adj_mat = torch.zeros((num_users + num_movies , num_users + num_movies))
    adj_mat[: num_users, num_users :] = R.clone()
    adj_mat[num_users :, : num_users] = R_transpose.clone()
    
    adj_mat_coo = adj_mat.to_sparse_coo()
    adj_mat_coo_indices = adj_mat_coo.indices()
    adj_mat_coo_values = adj_mat_coo.values()
    return adj_mat_coo_indices, adj_mat_coo_values

def convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index, input_edge_values):
    
    sparse_input_edge_index = SparseTensor(row=input_edge_index[0], 
                                           col=input_edge_index[1], 
                                           value = input_edge_values,
                                           sparse_sizes=((num_users + num_movies), num_users + num_movies))
    
    adj_mat = sparse_input_edge_index.to_dense()
    interact_mat = adj_mat[: num_users, num_users :]
    
    r_mat_edge_index = interact_mat.to_sparse_coo().indices()
    r_mat_edge_values = interact_mat.to_sparse_coo().values()
    
    return r_mat_edge_index, r_mat_edge_values

class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False, dropout_rate=0.1):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.add_self_loops = add_self_loops


        # define user and item embedding for direct look up.
        # embedding dimension: num_user/num_item x embedding_dim
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0

        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0


        # "Fills the input Tensor with values drawn from the normal distribution"
        # according to LightGCN paper, this gives better performance
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

        # create a linear layer (fully connected layer) so we can output a single value (predicted_rating)
        self.out = nn.Linear(embedding_dim + embedding_dim, 1)

    def forward(self, edge_index: Tensor, edge_values: Tensor):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """

        """
            compute /tilde{A}: symmetrically normalized adjacency matrix
            /tilde_A = D^(-1/2) * A * D^(-1/2)    according to LightGCN paper

            this is essentially a metrix operation way to get 1/ (sqrt(n_neighbors_i) * sqrt(n_neighbors_j))


            if your original edge_index look like
            tensor([[   0,    0,    0,  ...,  609,  609,  609],
                    [   0,    2,    5,  ..., 9444, 9445, 9485]])

                    torch.Size([2, 99466])

            then this will output:
                (
                 tensor([[   0,    0,    0,  ...,  609,  609,  609],
                         [   0,    2,    5,  ..., 9444, 9445, 9485]]),
                 tensor([0.0047, 0.0096, 0.0068,  ..., 0.0592, 0.0459, 0.1325])
                 )

              where edge_index_norm[0] is just the original edge_index

              and edge_index_norm[1] is the symmetrically normalization term.

            under the hood it's basically doing
                def compute_gcn_norm(edge_index, emb):
                    emb = emb.weight
                    from_, to_ = edge_index
                    deg = degree(to_, emb.size(0), dtype=emb.dtype)
                    deg_inv_sqrt = deg.pow(-0.5)
                    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

                    return norm


        """
        edge_index_norm = gcn_norm(edge_index=edge_index,
                                   add_self_loops=self.add_self_loops)

        # concat the user_emb and item_emb as the layer0 embing matrix
        # size will be (n_users + n_items) x emb_vector_len.   e.g: 10334 x 64
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0

        embs = [emb_0] # save the layer0 emb to the embs list

        # emb_k is the emb that we are actually going to push it through the graph layers
        # as described in lightGCN paper formula 7
        emb_k = emb_0

        # push the embedding of all users and items through the Graph Model K times.
        # K here is the number of layers
        for i in range(self.K):
            emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])
            embs.append(emb_k)


        # this is doing the formula8 in LightGCN paper

        # the stacked embs is a list of embedding matrix at each layer
        #    it's of shape n_nodes x (n_layers + 1) x emb_vector_len.
        #        e.g: torch.Size([10334, 4, 64])
        embs = torch.stack(embs, dim=1)

        # From LightGCn paper: "In our experiments, we find that setting α_k uniformly as 1/(K + 1)
        #    leads to good performance in general."
        emb_final = torch.mean(embs, dim=1) # E^K

        users_emb_final, items_emb_final = torch.split(emb_final,
                                                       [self.num_users, self.num_items]) # splits into e_u^K and e_i^K


        r_mat_edge_index, _ = convert_adj_mat_edge_index_to_r_mat_edge_index(edge_index, edge_values)

        src, dest =  r_mat_edge_index[0], r_mat_edge_index[1]

        # applying embedding lookup to get embeddings for src nodes and dest nodes in the edge list
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]

        # output dim: edge_index_len x 128 (given 64 is the original emb_vector_len)
        output = torch.cat([user_embeds, item_embeds], dim=1)

        # push it through the linear layer
        output = self.out(output)

        return output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

# layers = 3
# model = LightGCN(num_users=num_users,
#                  num_items=num_movies,
#                  K=layers)

model = torch.load("C:/Users/USER/Desktop/projects/GNN/app/full_model.pth")


def get_diverse_recommendations(model, new_user_ratings, num_recommendations=10, diversity_factor=0.3):
    model.eval()
    with torch.no_grad():
        # Get initial movie embeddings
        rated_movie_embeds = model.items_emb.weight[list(new_user_ratings.keys())]
        temp_user_embed = torch.mean(rated_movie_embeds, dim=0)

        # Get scores for all movies
        all_movie_embeds = model.items_emb.weight
        base_scores = torch.matmul(temp_user_embed, all_movie_embeds.t())

        # Add diversity penalty
        recommended_items = []
        already_selected = set(new_user_ratings.keys())
        # print("Best distance weight!")
        # print(torch.topk(base_scores, 15))
        for _ in range(num_recommendations):
            # Apply diversity penalty to already selected items
            scores = base_scores.clone()
            for item_id in already_selected:
                similarity = torch.matmul(all_movie_embeds[item_id], all_movie_embeds.t())
                scores -= diversity_factor * similarity

            # Get next best item
            scores[list(already_selected)] = float('-inf')
            next_item = torch.argmax(scores).item()
            recommended_items.append(next_item)
            already_selected.add(next_item)

        return recommended_items
    
# new_user_ratings = {
#     0: 5,
#     1: 5,
#     2: 5,
#     3: 5,
#     4: 5,
#     5: 5,
#     6: 5,
#     7: 5,
# }
# recommendations = get_diverse_recommendations(model,
#                                            new_user_ratings,
#                                            num_recommendations=10,
#                                            diversity_factor=0.3) # temperature/
                                               
# print(recommendations)
# movie_df.iloc[recommendations][["title", "genres"]]
from torch_geometric.data import download_url, extract_zip
import pandas as pd

url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
extract_zip(download_url(url, '.'), '.')

movie_path = './ml-latest-small/movies.csv'
movie_df = pd.read_csv(movie_path)


from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/recommend', methods=['POST'])
def recommend_movie():
    new_user_ratings = request.json.get('user_ratings', {})
    num_recommendations = request.json.get('num_recommendations', 20)
    diversity_factor = request.json.get('diversity_factor', 0.3)

    recommendations = get_diverse_recommendations(
        model,
        {int(i): int(r) for i, r in new_user_ratings.items()},
        num_recommendations=num_recommendations,
        diversity_factor=diversity_factor
    )

    recommended_movies = movie_df.iloc[recommendations][["title", "genres"]].to_dict(orient='records')

    return jsonify(recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)