"""Motion / environment / class conditioning encoder (spec §4)."""
import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(self, dim=256, n_classes=3, max_targets=5):
        super().__init__()
        self.phi_motion = nn.Sequential(nn.Linear(2, dim), nn.SiLU(),
                                        nn.Linear(dim, dim))
        self.phi_env = nn.Sequential(nn.Linear(3, dim), nn.SiLU(),
                                     nn.Linear(dim, dim))
        self.cls_emb = nn.Embedding(n_classes, dim)
        self.fusion = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(),
                                    nn.Linear(dim, dim))
        self.null_cond = nn.Parameter(torch.zeros(dim))

    def null(self, batch_size, device):
        return self.null_cond.to(device).expand(batch_size, -1)

    def forward(self, batch, device, dropout_p=0.0):
        v0 = batch["v0"].to(device)
        acc = batch["acc"].to(device)
        cls = batch["cls"].to(device)
        env = batch["env"].to(device)
        n = batch["n_targets"].to(device)
        B, Mmax = v0.shape
        mask = (torch.arange(Mmax, device=device)[None] < n[:, None]).float()
        mdenom = mask.sum(dim=1, keepdim=True).clamp(min=1)

        motion = self.phi_motion(torch.stack([v0, acc], dim=-1))     # (B, M, d)
        motion = (motion * mask[..., None]).sum(dim=1) / mdenom
        cemb = (self.cls_emb(cls) * mask[..., None]).sum(dim=1) / mdenom
        fused = self.fusion(torch.cat([motion, self.phi_env(env), cemb], dim=-1))

        if dropout_p > 0:
            drop = (torch.rand(B, device=device) < dropout_p)[:, None]
            fused = torch.where(drop, self.null(B, device), fused)
        return fused
