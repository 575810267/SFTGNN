import torch
import torch.nn as nn
from torch_scatter import scatter
from .sublayer import SFTConv, RBFExpansion
from .config import Config as model_config
from data.config import Config as data_config


class SFTGNN(nn.Module):
    """
        SFTGNN Standard Model
    """

    def __init__(self, num_layers: int = 5,
                 in_dim: int = 92,
                 node_feature: int = 128,
                 dist_dim: int = 64,
                 sh_dim: int = 64,
                 use_sh: bool = True,
                 use_dist: bool = True):
        """
            SFTGNN Standard Model
        :param num_layers:
        :param in_dim: The feature dimension of the AtomFeatureType used
        :param node_feature:
        :param dist_dim:
        :param sh_dim:
        :param use_sh:
        :param use_dist:
        """
        super().__init__()
        if in_dim != 1:  # AtomFeatureType = CGCNN or CrysAtomVec
            self.atom_embedding = nn.Linear(in_dim, node_feature)
        else:  # AtomFeatureType = AtomicNumber
            self.atom_embedding = nn.Sequential(
                nn.Embedding(94, node_feature),
                nn.Linear(node_feature, node_feature)
            )
        self.num_layers = num_layers
        self.use_sh = use_sh
        self.use_dist = use_dist
        self.layers = nn.ModuleList(
            [SFTConv(node_feature, dist_dim, sh_dim, use_sh, use_dist) for _ in range(self.num_layers)])
        self.sh_mlp = nn.Sequential(
            nn.Linear(data_config.spherical_harmonics_l ** 2, 32),
            nn.SiLU(),
            nn.Linear(32, sh_dim),
        )
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0.125, vmax=1.4, bins=64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, dist_dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(node_feature, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, data) -> torch.Tensor:
        node_features = self.atom_embedding(data.x)
        index = data.edge_index
        sh = self.sh_mlp(data.sh)
        with torch.no_grad():
            dist = 1 / data.edge_attr.unsqueeze(-1)
        dist = self.rbf(dist)
        edge_attr = torch.cat([dist, sh], dim=-1)
        for layer in self.layers:
            node_features = layer(node_features, index, edge_attr)
        features = scatter(node_features, data.batch, dim=0, reduce="mean")
        out = self.fc(features)
        return out.squeeze()
