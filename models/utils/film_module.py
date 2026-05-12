import torch
import torch.nn as nn

class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer.
       Given input feature map x and condition vector cond,
       outputs x * gamma + beta, where gamma and beta are computed from cond.
    """
    def __init__(self, feature_dim, cond_dim):
        super(FiLM, self).__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_dim)
        self.beta_fc = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        # x: [B, C, H, W], cond: [B, cond_dim]
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)      # [B, C, 1, 1]
        return x * gamma + beta
