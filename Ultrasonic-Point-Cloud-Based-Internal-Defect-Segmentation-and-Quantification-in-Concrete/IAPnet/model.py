import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSegmentation(nn.Module):
    """Point-wise segmentation with encoder, global context, and skip decoder."""

    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.enc5 = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.dec1 = nn.Sequential(
            nn.Conv1d(1024 + 512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(512 + 256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv1d(256 + 128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, num_classes, 1),
        )

        self.num_classes = num_classes

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2)

        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)

        global_feat = torch.max(f5, dim=2, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, N)

        d1 = self.dec1(torch.cat([global_feat, f4], dim=1))
        d2 = self.dec2(torch.cat([d1, f3], dim=1))
        d3 = self.dec3(torch.cat([d2, f2], dim=1))

        out = self.seg_head(d3)
        out = out.transpose(1, 2)

        return out

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=2)

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=2)


class FocalLoss(nn.Module):
    """Focal loss for class imbalance. alpha: positive-class weight; gamma: focusing term."""

    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


if __name__ == "__main__":
    model = PointNetSegmentation(in_channels=5, num_classes=2)
    x = torch.randn(2, 8192, 5)
    out = model(x)
    print(f"in:  {x.shape}")
    print(f"out: {out.shape}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
