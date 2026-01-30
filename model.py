import math

import gin
import torch
import torch.nn
import torch.nn.functional as F



def _compute_posterior_given_prior(
        mut_given_prev: torch.Tensor,
        Lt_given_prev: torch.Tensor,
        x: torch.Tensor,
        H: torch.Tensor,
        Cws: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the posterior given the prior and the observations."""
    et = x - torch.einsum('ij,...j->...i', H, mut_given_prev)
    Re = H @ Lt_given_prev @ H.transpose(-1, -2) + Cws
    Kt_given_prev = Lt_given_prev @ H.transpose(-1, -2) @ torch.linalg.inv(Re)
    mut_given_cur = mut_given_prev + torch.einsum(
        '...ij,...j->...i', Kt_given_prev, et
    )
    Lt_given_cur = Lt_given_prev - Kt_given_prev @ Re @ Kt_given_prev.transpose(
        -1, -2
    )
    return mut_given_cur, Lt_given_cur


@gin.configurable
class DatadrivenNonlinearSmoother(torch.nn.Module):
    """RNN model for prediction with causal CNN preprocessing."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            num_layers: int,
            hidden_dim_dense: int,
            cnn_channels: list[int] = [16, 16],
            cnn_kernel_sizes: list[int] = [3, 3],
            cnn_output_dim: int | None = None,
            H: torch.Tensor | None = None,
            device: str = 'cpu',
        ):
        """Constructor.

        Args:
            input_dim: The dimensionality of the input data.
            output_dim: The dimensionality of the output data.
            hidden_dim: The size of the hidden layer for GRU.
            num_layers: The number of GRU layers.
            hidden_dim_dense: Dense layer hidden dimension.
            cnn_channels: List of channel sizes for CNN layers.
            cnn_kernel_sizes: List of kernel sizes for CNN layers.
            cnn_output_dim: Output dimension after CNN processing.
            H: Observation matrix.
            device: Device for computations.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.cnn_output_dim = cnn_output_dim if cnn_output_dim is not None else input_dim
        
        self.cnn_fwd_layers = torch.nn.ModuleList()
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, cnn_kernel_sizes)):
            padding = kernel_size - 1  # Causal padding
            conv_layer = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True
            )
            self.cnn_fwd_layers.append(conv_layer)
            in_channels = out_channels
        
        self.cnn_bwd_layers = torch.nn.ModuleList()
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, cnn_kernel_sizes)):
            padding = kernel_size - 1  # Anti-causal padding
            conv_layer = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True
            )
            self.cnn_bwd_layers.append(conv_layer)
            in_channels = out_channels
        
        # Final projection layers to match desired output dimension
        if len(cnn_channels) > 0:
            self.cnn_fwd_projection = torch.nn.Linear(cnn_channels[-1], self.cnn_output_dim)
            self.cnn_bwd_projection = torch.nn.Linear(cnn_channels[-1], self.cnn_output_dim)
        else:
            self.cnn_fwd_projection = torch.nn.Identity()
            self.cnn_bwd_projection = torch.nn.Identity()
        
        self.cnn_dropout = torch.nn.Dropout(0.1)
        
        self.gru_fwd = torch.nn.GRU(self.cnn_output_dim, hidden_dim, num_layers, batch_first=True)
        self.gru_bwd = torch.nn.GRU(self.cnn_output_dim, hidden_dim, num_layers, batch_first=True)
        self.gru_xhat = torch.nn.GRU(input_dim, output_dim, num_layers, batch_first=True)

        self.fc = torch.nn.Linear(hidden_dim, hidden_dim_dense)
        self.fc_mean = torch.nn.Linear(hidden_dim_dense, output_dim)
        self.fc_var = torch.nn.Linear(hidden_dim_dense, output_dim)
        
        self.map_down = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.fc_single1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim_dense),
            torch.nn.ReLU(),
        )
        self.fc_single_mean1 = torch.nn.Linear(hidden_dim_dense, output_dim)
        self.fc_single_var1 = torch.nn.Linear(hidden_dim_dense, output_dim)
        self.fc_single2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim_dense),
            torch.nn.ReLU(),
        )
        self.fc_single_mean2 = torch.nn.Linear(hidden_dim_dense, output_dim)
        self.fc_single_var2 = torch.nn.Linear(hidden_dim_dense, output_dim)
        
        self.fc_xhat = torch.nn.Sequential(
            torch.nn.Linear(output_dim, output_dim),
            torch.nn.ReLU(),
        )
        self.fc_xhat_mean = torch.nn.Linear(output_dim, output_dim)
        self.fc_xhat_var = torch.nn.Linear(output_dim, output_dim)
        
        self.H = H if H is not None else torch.eye(output_dim)
        self.H = self.H.to(device)
        self._device = device 
        
        self.fc_mu_xy = torch.nn.Linear(2 * output_dim, output_dim)
        self.fc_var_xy = torch.nn.Linear(2 * output_dim, output_dim)
        self.softplus = torch.nn.Softplus(beta=0.25, threshold=5.0)

        self._init_weights()

    def _apply_causal_cnn(self, x):
        """Apply causal CNN preprocessing for forward GRU."""
        B, T, D = x.shape
        
        x = x.transpose(1, 2)
        
        for conv_layer in self.cnn_fwd_layers:
            x = conv_layer(x)
            kernel_size = conv_layer.kernel_size[0]
            x = x[:, :, :-(kernel_size-1)] if kernel_size > 1 else x
            x = F.relu(x)
            x = self.cnn_dropout(x)
        
        x = x.transpose(1, 2)
        
        return self.cnn_fwd_projection(x)

    def _apply_anti_causal_cnn(self, x):
        """Apply anti-causal CNN preprocessing for backward GRU."""
        B, T, D = x.shape
        
        x = x.transpose(1, 2)
        
        for conv_layer in self.cnn_bwd_layers:
            x = conv_layer(x)
            kernel_size = conv_layer.kernel_size[0]
            x = x[:, :, (kernel_size-1):] if kernel_size > 1 else x
            x = F.relu(x)
            x = self.cnn_dropout(x)
        
        x = x.transpose(1, 2)
        
        return self.cnn_bwd_projection(x)

    def _init_weights(self):
        """Custom weight initialization"""
        
        # CNN initialization
        for conv_layer in self.cnn_fwd_layers:
            if isinstance(conv_layer, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
                if conv_layer.bias is not None:
                    torch.nn.init.zeros_(conv_layer.bias)
        
        for conv_layer in self.cnn_bwd_layers:
            if isinstance(conv_layer, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
                if conv_layer.bias is not None:
                    torch.nn.init.zeros_(conv_layer.bias)
        
        # CNN projection layers
        if isinstance(self.cnn_fwd_projection, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(self.cnn_fwd_projection.weight)
            torch.nn.init.zeros_(self.cnn_fwd_projection.bias)
        
        if isinstance(self.cnn_bwd_projection, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(self.cnn_bwd_projection.weight)
            torch.nn.init.zeros_(self.cnn_bwd_projection.bias)
        
        # GRU initialization
        for gru in [self.gru_fwd, self.gru_bwd, self.gru_xhat]:
            for name, param in gru.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
        
        torch.nn.init.xavier_uniform_(self.fc_mean.weight)
        torch.nn.init.zeros_(self.fc_mean.bias)
        
        var_bias_constant = -1.0
        torch.nn.init.xavier_uniform_(self.fc_var.weight)
        torch.nn.init.constant_(self.fc_var.bias, var_bias_constant)
        
        torch.nn.init.xavier_uniform_(self.fc_single1[0].weight)
        torch.nn.init.zeros_(self.fc_single1[0].bias)
        torch.nn.init.xavier_uniform_(self.fc_single_mean1.weight)
        torch.nn.init.zeros_(self.fc_single_mean1.bias)
        torch.nn.init.xavier_uniform_(self.fc_single_var1.weight)
        torch.nn.init.constant_(self.fc_single_var1.bias, var_bias_constant)
        torch.nn.init.xavier_uniform_(self.fc_single2[0].weight)
        torch.nn.init.zeros_(self.fc_single2[0].bias)
        torch.nn.init.xavier_uniform_(self.fc_single_mean2.weight)
        torch.nn.init.zeros_(self.fc_single_mean2.bias)
        torch.nn.init.xavier_uniform_(self.fc_single_var2.weight)
        torch.nn.init.constant_(self.fc_single_var2.bias, var_bias_constant)
        torch.nn.init.xavier_uniform_(self.fc_xhat[0].weight)
        torch.nn.init.zeros_(self.fc_xhat[0].bias)
        torch.nn.init.xavier_uniform_(self.fc_xhat_mean.weight)
        torch.nn.init.zeros_(self.fc_xhat_mean.bias)
        torch.nn.init.xavier_uniform_(self.fc_xhat_var.weight)
        torch.nn.init.constant_(self.fc_xhat_var.bias, var_bias_constant)
        torch.nn.init.xavier_uniform_(self.fc_mu_xy.weight)
        torch.nn.init.zeros_(self.fc_mu_xy.bias)
        torch.nn.init.xavier_uniform_(self.fc_var_xy.weight)
        torch.nn.init.constant_(self.fc_var_xy.bias, var_bias_constant)
    
    def forward(self, x, Cws):
        B, T, _ = x.shape

        x_cnn_fwd = self._apply_causal_cnn(x)
        x_cnn_bwd = self._apply_anti_causal_cnn(x)
        h_fwd, _ = self.gru_fwd(x_cnn_fwd)
        x_cnn_bwd_rev = torch.flip(x_cnn_bwd, dims=[1])
        h_bwd, _ = self.gru_bwd(x_cnn_bwd_rev)
        h_bwd = torch.flip(h_bwd, dims=[1])
        h_fwd_shifted = h_fwd[:, :-2, :]
        h_bwd_shifted = h_bwd[:, 2:, :]
        h_middle = self.map_down(
            torch.cat([h_fwd_shifted, h_bwd_shifted], dim=-1)
        )
        h = torch.cat([h_bwd[:, 1:2, :], h_middle, h_fwd[:, T-2:T-1, :]], dim=1)

        h_mid = F.relu(self.fc(h))
        mut_prior_y = self.fc_mean(h_mid)
        var_prior_y = self.fc_var(h_mid)
        Lt_prior_y = torch.diag_embed(self.softplus(var_prior_y))
        
        mut_posterior, Lt_posterior = _compute_posterior_given_prior(
            mut_prior_y, Lt_prior_y, x, self.H, Cws
        )
        
        h_xhat, _ = self.gru_xhat(mut_posterior.detach())
        mu_xhat = self.fc_xhat_mean(h_xhat)
        var_xhat = self.fc_xhat_var(h_xhat)
        
        mut_prior = torch.cat(
            [torch.zeros((B, 1, self.output_dim), device=self._device), mu_xhat[:, :-1, :]], dim=1
        ) + mut_prior_y
        var_prior = torch.cat(
            [torch.zeros((B, 1, self.output_dim), device=self._device), var_xhat[:, :-1, :]], dim=1
        ) + var_prior_y
        Lt_prior = torch.diag_embed(self.softplus(var_prior))

        mut_posterior, Lt_posterior = _compute_posterior_given_prior(
            mut_prior, Lt_prior, x, self.H, Cws
        )

        return mut_prior, Lt_prior, mut_posterior, Lt_posterior


def main():
    pass


if __name__ == "__main__":
    main()