import torch
import torch.nn as nn

from .landmark_conformer import LandmarkConformer
from ..config import COORDS_PER_LM, HAND_PARENT, HAND_TREE, LH_START, RH_START

PALM_EDGES = [(5, 9), (9, 13), (13, 17)]  # knuckle web


def hand_adjacency() -> torch.Tensor:
    """Symmetric-normalised D^-1/2 (A + I) D^-1/2 over the 21-node hand graph."""
    a = torch.eye(21)
    for i, j in HAND_TREE + PALM_EDGES:
        a[i, j] = a[j, i] = 1.0
    d = a.sum(1).rsqrt()
    return d[:, None] * a * d[None, :]


class HandGraphLayer(nn.Module):
    """Pre-norm residual graph convolution over hand joints, one frame at a time.

    The fixed skeleton adjacency is augmented by a learned (21, 21) residual
    (zero-initialised) so the layer can add non-anatomical links such as
    thumb tip ↔ index tip, which matter for hand shape but are 6 hops apart in
    the skeleton. LayerNorm, not BatchNorm: no statistics over padded frames.
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("adj", hand_adjacency(), persistent=False)
        self.adj_residual = nn.Parameter(torch.zeros(21, 21))
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (N, 21, dim)
        y = torch.einsum("vw,nwc->nvc", self.adj + self.adj_residual, self.norm(h))
        return h + self.drop(self.act(self.proj(y)))


class HandGraphEncoder(nn.Module):
    """Shared per-frame hand-shape encoder: (N, 21, 5c) node features → (N, 21·out_dim).

    Joint identity is kept twice: a learned per-joint embedding is added to
    the input, and the output is flattened per joint rather than pooled.
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int, out_dim: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.joint_emb = nn.Parameter(torch.zeros(21, hidden))
        nn.init.normal_(self.joint_emb, std=0.02)
        self.layers = nn.ModuleList([HandGraphLayer(hidden, dropout) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(nodes) + self.joint_emb
        for layer in self.layers:
            h = layer(h)
        return self.out(h).flatten(1)


class GraphConformer(LandmarkConformer):
    """LandmarkConformer with the flat hand projections replaced by a hand graph.

    Only the hand position/velocity streams change (lh/rh_proj and
    lh/rh_vel_proj); pose, face, geometry, distance, fusion, Conformer blocks
    and heads are inherited unchanged, so a run against LandmarkConformer
    isolates the effect of the graph encoder.

    Per hand slot, per frame:
      - shape (shared graph encoder, both hands): wrist-relative joint
        positions, bone vectors (joint − parent) and Δ1/Δ2/Δ5 of the
        wrist-relative joints; the wrist node itself is the origin. The
        non-dominant slot holds the other hand's anatomy after
        HandDominanceModule, so its x is negated first — both hands then
        reach the shared weights with the same chirality.
      - location: the body-relative wrist position and its Δ1/Δ2/Δ5, kept out
        of the graph (hand shape alone cannot tell signs at different
        locations apart).
    Each slot has its own output projection to d_model//2, split into the
    pos/vel halves the parent's fusion layout expects.
    """

    def __init__(self, num_classes, d_model=512, graph_hidden=64, graph_layers=3,
                 graph_out=16, dropout=0.1, **kwargs):
        super().__init__(num_classes, d_model=d_model, dropout=dropout, **kwargs)
        del self.lh_proj, self.rh_proj, self.lh_vel_proj, self.rh_vel_proj

        c = COORDS_PER_LM
        self.hand_graph = HandGraphEncoder(5 * c, graph_hidden, graph_layers, graph_out, dropout)
        self.lh_hand_out = nn.Linear(21 * graph_out + 4 * c, d_model // 2)
        self.rh_hand_out = nn.Linear(21 * graph_out + 4 * c, d_model // 2)
        self.register_buffer("hand_parent", torch.tensor(HAND_PARENT), persistent=False)
        # Negates x of every coordinate group in the node features.
        node_x_sign = torch.ones(5 * c)
        node_x_sign[::c] = -1.0
        self.register_buffer("node_x_sign", node_x_sign, persistent=False)

    def _hand_streams(self, pos, vel1, vel2, vel5):
        B, T, _ = pos.shape
        c = COORDS_PER_LM
        node_feats, locations = [], []
        for hs in (LH_START, RH_START):
            s = slice(hs, hs + 21 * c)
            p, v1, v2, v5 = (t[:, :, s].reshape(B, T, 21, c) for t in (pos, vel1, vel2, vel5))
            locations.append(torch.cat([p[:, :, 0], v1[:, :, 0], v2[:, :, 0], v5[:, :, 0]], -1))
            # Wrist node = origin of the shape frame (its motion is location).
            shape = torch.cat(
                [torch.zeros_like(p[:, :, :1]), p[:, :, 1:]], dim=2
            )
            vels = [torch.cat([torch.zeros_like(v[:, :, :1]), v[:, :, 1:]], dim=2) for v in (v1, v2, v5)]
            bones = shape - shape[:, :, self.hand_parent]
            node_feats.append(torch.cat([shape, bones, *vels], dim=-1))  # (B, T, 21, 5c)
        node_feats[1] = node_feats[1] * self.node_x_sign  # non-dominant → same chirality

        nodes = torch.stack(node_feats, dim=2)  # (B, T, 2, 21, 5c)
        shape_emb = self.hand_graph(nodes.reshape(B * T * 2, 21, -1)).reshape(B, T, 2, -1)
        lh = self.lh_hand_out(torch.cat([shape_emb[:, :, 0], locations[0]], -1))
        rh = self.rh_hand_out(torch.cat([shape_emb[:, :, 1], locations[1]], -1))
        lh_p, lh_v = lh.chunk(2, dim=-1)
        rh_p, rh_v = rh.chunk(2, dim=-1)
        return lh_p, rh_p, lh_v, rh_v
