import torch
import torch.nn as nn
import numpy as np
from feature_utils import extract_features_from_frame


# ---------------------------
# Encoder Network
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, in_dim, hid=128, emb=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.BatchNorm1d(hid),

            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.BatchNorm1d(hid),

            nn.Linear(hid, emb)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Siamese Network
# ---------------------------
class Siamese(nn.Module):
    def __init__(self, in_dim, hid=128, emb=64):
        super().__init__()

        self.enc = Encoder(in_dim, hid, emb)

    def forward(self, x1, x2):

        z1 = self.enc(x1)
        z2 = self.enc(x2)

        return z1, z2


# ---------------------------
# Contrastive Loss
# ---------------------------
def contrastive_loss(z1, z2, label, margin=1.0):

    d = torch.nn.functional.pairwise_distance(z1, z2)

    pos = 0.5 * label * d**2
    neg = 0.5 * (1 - label) * torch.clamp(margin - d, min=0)**2

    return (pos + neg).mean(), d


# ---------------------------
# Load Model
# ---------------------------
def load_model(model_path, cfg):

    device = torch.device("cpu")

    checkpoint = torch.load(model_path, map_location=device)

    feat_dim = checkpoint["feat_dim"]

    model = Siamese(
        feat_dim,
        hid=cfg["hid"],
        emb=cfg["emb"]
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


# ---------------------------
# Frame Prediction
# ---------------------------
def predict_frame(frame, model, ref_real, ref_fake, cfg):

    feat = extract_features_from_frame(frame, cfg)

    feat = torch.from_numpy(feat).float().unsqueeze(0)

    with torch.no_grad():

        z_img, _ = model(feat, feat)

        dist_real = torch.nn.functional.pairwise_distance(
            z_img,
            ref_real
        )

        dist_fake = torch.nn.functional.pairwise_distance(
            z_img,
            ref_fake
        )

    if dist_real.item() < dist_fake.item():

        label = "REAL"
        distance = dist_real.item()

    else:

        label = "FAKE"
        distance = dist_fake.item()

    return label, distance
