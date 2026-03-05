import torch
import torch.nn as nn
from utils.feature_utils import extract_features_from_frame

class Encoder(nn.Module):

    def __init__(self,in_dim,hid=128,emb=64):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim,hid),
            nn.ReLU(),
            nn.BatchNorm1d(hid),

            nn.Linear(hid,hid),
            nn.ReLU(),
            nn.BatchNorm1d(hid),

            nn.Linear(hid,emb)
        )

    def forward(self,x):

        return self.net(x)


class Siamese(nn.Module):

    def __init__(self,in_dim,hid=128,emb=64):

        super().__init__()

        self.enc = Encoder(in_dim,hid,emb)

    def forward(self,x1,x2):

        return self.enc(x1),self.enc(x2)


def load_model(model_path,cfg):

    device = torch.device("cpu")

    checkpoint = torch.load(model_path,map_location=device)

    feat_dim = checkpoint["feat_dim"]

    model = Siamese(feat_dim,cfg["hid"],cfg["emb"])

    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    return model


def predict_frame(frame,model,ref_real,ref_fake,cfg):

    feat = extract_features_from_frame(frame,cfg)

    feat = torch.from_numpy(feat).float().unsqueeze(0)

    with torch.no_grad():

        z,_ = model(feat,feat)

        dist_real = torch.nn.functional.pairwise_distance(z,ref_real)

        dist_fake = torch.nn.functional.pairwise_distance(z,ref_fake)

    if dist_real.item()<dist_fake.item():

        return "REAL",dist_real.item()

    else:

        return "FAKE",dist_fake.item()
