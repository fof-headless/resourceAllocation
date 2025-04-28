import torch
import torch.nn as nn
import torch.nn.functional as F

class AllocatorNN(nn.Module):
    def __init__(self, ue_input_dim, bs_input_dim, hidden_dim, num_bs):
        super().__init__()
        self.num_bs = num_bs

        # Encoders
        self.ue_encoder = nn.Sequential(
            nn.Linear(ue_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim))

        self.bs_encoder = nn.Sequential(
            nn.Linear(bs_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim))

        # Attention with SNR consideration
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # +1 for SNR
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # +1 for SNR
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU())

        # Output heads
        self.bs_classifier = nn.Linear(hidden_dim // 2, num_bs + 1)
        self.bw_regressor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid())

    def forward(self, ue_state, bs_state, distances, snr_db):
        N_ue = ue_state.shape[0]
        N_bs = bs_state.shape[0]

        # Encode
        ue_features = self.ue_encoder(ue_state)
        bs_features = self.bs_encoder(bs_state)

        # Prepare attention inputs with SNR
        ue_exp = ue_features.unsqueeze(1).expand(-1, N_bs, -1)  # Shape: (N_ue, N_bs, hidden_dim)
        bs_exp = bs_features.unsqueeze(0).expand(N_ue, -1, -1)   # Shape: (N_ue, N_bs, hidden_dim)
        snr_exp = snr_db.unsqueeze(-1)                           # Shape: (N_ue, N_bs, 1)
        combined = torch.cat([ue_exp, bs_exp, snr_exp], dim=-1)  # Shape: (N_ue, N_bs, hidden_dim*2+1)

        # Distance penalty
        max_dist = bs_state[:, 4].unsqueeze(0).expand(N_ue, -1)
        dist_penalty = torch.where(
            distances > max_dist,
            torch.ones_like(distances) * -1e9,
            torch.zeros_like(distances))

        attn_scores = self.attention(combined).squeeze(-1) + dist_penalty
        attn_weights = F.softmax(attn_scores, dim=1)

        # Get best BS features and SNR for each UE
        best_bs_indices = torch.argmax(attn_weights, dim=1, keepdim=True).unsqueeze(-1).expand(-1, -1, bs_exp.size(-1))
        weighted_bs = torch.gather(bs_exp, 1, best_bs_indices).squeeze(1)
        best_snr = torch.gather(snr_db, 1, torch.argmax(attn_weights, dim=1, keepdim=True)).squeeze(1)

        # Decode with SNR information
        x = torch.cat([ue_features, weighted_bs, best_snr.unsqueeze(-1)], dim=1)
        x = self.decoder(x)

        # Outputs
        bs_logits = self.bs_classifier(x)
        bs_assignment = torch.argmax(bs_logits, dim=1) - 1

        bw_ratio = self.bw_regressor(x).squeeze(1)
        bw_pred = bw_ratio * ue_state[:, 4]

        # Ensure we never allocate more than needed
        bw_pred = torch.min(bw_pred, ue_state[:, 4])

        return bs_assignment, bw_pred