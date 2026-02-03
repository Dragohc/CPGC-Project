import torch
import torch.nn as nn
import torch.nn.functional as F


class CEPM(nn.Module):
    def __init__(
        self,
        in_channels,
        num_branches,
        decoder,
        alpha=0.5,
        lambda_decorr=0.1,
        eps=1e-6,
    ):
        super().__init__()
        self.Kb = num_branches
        self.alpha = alpha
        self.lambda_decorr = lambda_decorr
        self.eps = eps

        self.attn_heads = nn.ModuleList(
            [PrototypeAttention(in_channels) for _ in range(num_branches)]
        )
        self.decoder = decoder

    def build_prototypes(self, Fs, Ms):
        """
        Fs: list of support features, each [1, C, H, W]
        Ms: list of support masks, each [1, 1, H, W]
        """
        prototypes = []

        for k in range(self.Kb):
            proto = 0.0
            for F_s, M_s in zip(Fs, Ms):
                logits = self.attn_heads[k](F_s)  # [1,1,H,W]
                logits = logits.squeeze(1)

                attn = torch.exp(logits) * M_s.squeeze(1)
                attn = attn / (attn.sum(dim=(1, 2), keepdim=True) + self.eps)

                proto = proto + (F_s * attn.unsqueeze(1)).sum(dim=(2, 3))
            proto = proto / len(Fs)
            prototypes.append(proto)  # [1, C]

        return prototypes

    def dense_match(self, Fq, prototypes):
        """
        Fq: [1, C, H, W]
        prototypes: list of [1, C]
        """
        heatmaps = []
        for p in prototypes:
            p = F.normalize(p, dim=1)
            f = F.normalize(Fq, dim=1)
            sim = (f * p.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            heatmaps.append(sim)
        return heatmaps

    def decorrelation_loss(self, heatmaps):
        loss = 0.0
        K = len(heatmaps)
        for i in range(K - 1):
            for j in range(i + 1, K):
                h1 = heatmaps[i].flatten()
                h2 = heatmaps[j].flatten()
                loss += F.cosine_similarity(h1, h2, dim=0)
        return 2 * loss / (K * (K - 1) + self.eps)

    def forward(self, Fq, Fs, Ms, Mq=None):
        """
        Fs, Ms: lists of support features and masks
        """
        prototypes = self.build_prototypes(Fs, Ms)
        heatmaps = self.dense_match(Fq, prototypes)

        branch_preds = [
            self.decoder(h.unsqueeze(1), Fq) for h in heatmaps
        ]

        fused_heatmap = torch.stack(heatmaps, dim=0).max(dim=0)[0]
        coarse_pred = self.decoder(fused_heatmap.unsqueeze(1), Fq)

        if Mq is None:
            return coarse_pred

        loss_fuse = dice_loss(coarse_pred, Mq)
        loss_branch = sum(dice_loss(p, Mq) for p in branch_preds) / self.Kb
        loss_decorr = self.decorrelation_loss(heatmaps)

        loss = (
            loss_fuse
            + self.alpha * loss_branch
            + self.lambda_decorr * loss_decorr
        )
        return coarse_pred, loss
