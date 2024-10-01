import torch.nn as nn


# # Hyperparameters
# batch_size = 64
# learning_rate = 0.001
# num_epochs = 5


class SimpleNN(nn.Module):
    def __init__(
            self, 
            ip_dim: int, 
            op_dim: int, 
            n_hidden_layers: int, 
            hidden_layers_dims: list, 
            activation_functions: list
            ):
        super(SimpleNN, self).__init__()
        assert n_hidden_layers == len(hidden_layers_dims), "n_hidden_layers should be equal to len(hidden_layers_dims)"
        assert len(activation_functions) == len(hidden_layers_dims) + 1, "len(activation_functions) should be equal to len(hidden_layers_dims) + 1"

        layer_details = zip(
            [ip_dim] + hidden_layers_dims,
            hidden_layers_dims + [op_dim],
            activation_functions
        )

        self.linear_layers = nn.ModuleList(
            [
                act_func(nn.Linear(ip_dim, op_dim)) 
                for ip_dim, op_dim, act_func in layer_details
            ]
        )

        self.ip_dim = ip_dim
        self.op_dim = op_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layers_dims = hidden_layers_dims

    def forward(self, x):
        x = self.linear_layers(x)
        return x
