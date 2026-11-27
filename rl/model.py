"""Set transformer over entity tokens + pointer-style candidate scoring + value head.

Token embedding = type embedding + Linear(LN([bag_presence, bag_symlog, bag_lin])), where each bag is an
EmbeddingBag over the token's hashed entries (two rows per key, see encode.py) weighted by 1, symlog(x) and
clip(x/scale). No positional encoding: tokens form a set (positions that matter, like hand slot or map row,
are entries). Candidates are embedded with the same bag scheme, plus projections of the contextual
embeddings of the tokens they point at (primary / target / bundle members) and the global token, then go
through `cand_layers` blocks of (self-attention over candidates, cross-attention to tokens, MLP).
"""
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import HASH_ROWS, N_TOKEN_TYPES

NEG = -1e9
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")


@dataclass
class ModelCfg:
    d: int = 256
    layers: int = 4
    heads: int = 8
    ff_mult: int = 4
    cand_layers: int = 1
    d_emb: int = 0  # hash-table width; 0 -> d
    value_tower: bool = False
    value_layers: int = 0  # 0 -> layers
    mlp: str = "gelu"

    def __post_init__(self):
        if self.mlp not in ("gelu", "swiglu"):
            raise ValueError(f"unknown mlp: {self.mlp}")


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, mem, mask):
        """x [B,Q,d] attends to mem [B,K,d]; mask [B,K] bool (True = attend)."""
        B, Q, d = x.shape
        K = mem.shape[1]
        q = self.q(x).view(B, Q, self.h, -1).transpose(1, 2)
        k, v = self.kv(mem).view(B, K, 2, self.h, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        return self.o(y.transpose(1, 2).reshape(B, Q, d))


class SwiGLU(nn.Module):
    def forward(self, x):
        gate, value = x.chunk(2, -1)
        return F.silu(gate) * value


class MLP(nn.Sequential):
    def __init__(self, d, mult, kind="gelu"):
        if kind == "swiglu":
            width = (2 * mult * d // 3 + 7) // 8 * 8
            super().__init__(nn.Linear(d, 2 * width), SwiGLU(), nn.Linear(width, d))
        else:
            super().__init__(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))


class Block(nn.Module):
    def __init__(self, d, heads, mult, mlp="gelu"):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, heads)
        self.mlp = MLP(d, mult, mlp)

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, mask)
        return x + self.mlp(self.ln2(x))


class CandBlock(nn.Module):
    def __init__(self, d, heads, mult, mlp="gelu"):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = Attn(d, heads)
        self.cross = Attn(d, heads)
        self.mlp = MLP(d, mult, mlp)

    def forward(self, c, cmask, mem, tmask):
        h = self.ln1(c)
        c = c + self.self_attn(h, h, cmask)
        c = c + self.cross(self.ln2(c), mem, tmask)
        return c + self.mlp(self.ln3(c))


def _gather(h, idx):
    """h [B,T,d], idx [B,K] (>=0) -> [B,K,d]"""
    return torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))


class Policy(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        d, de = cfg.d, cfg.d_emb or cfg.d
        self.table = nn.Parameter(torch.randn(HASH_ROWS, de) * 0.02)
        with torch.no_grad():
            self.table[0].zero_()
        self.type_emb = nn.Embedding(N_TOKEN_TYPES, d)
        self.tok_in = nn.Sequential(nn.LayerNorm(3 * de), nn.Linear(3 * de, d))
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.ff_mult, cfg.mlp) for _ in range(cfg.layers))
        self.ln_f = nn.LayerNorm(d)
        self.cand_in = nn.Sequential(nn.LayerNorm(3 * de), nn.Linear(3 * de, d))
        self.ref_primary = nn.Linear(d, d)
        self.ref_target = nn.Linear(d, d)
        self.ref_member = nn.Linear(d, d)
        self.ref_global = nn.Linear(d, d)
        self.cand_blocks = nn.ModuleList(CandBlock(d, cfg.heads, cfg.ff_mult, cfg.mlp) for _ in range(cfg.cand_layers))
        self.logit = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        if cfg.value_tower:
            self.value_type_emb = nn.Embedding(N_TOKEN_TYPES, d)
            self.value_tok_in = nn.Sequential(nn.LayerNorm(3 * de), nn.Linear(3 * de, d))
            self.value_blocks = nn.ModuleList(Block(d, cfg.heads, cfg.ff_mult, cfg.mlp)
                                              for _ in range(cfg.value_layers or cfg.layers))
            self.value_ln_f = nn.LayerNorm(d)

    def bag(self, key, val):
        """key [..., E] int32 packed hashes, val [..., E, 2] -> [..., 3*d_emb] (computed in fp32).

        The three weighted bags (presence, symlog, linear) are one sparse [3N x 2^16] CSR matrix times the
        table. This is ~5x faster to backprop than F.embedding_bag, whose backward serializes on keys that
        occur in almost every token (e.g. "card.cost")."""
        lead = key.shape[:-1]
        key = key.reshape(-1, key.shape[-1]).long()
        val = val.reshape(-1, val.shape[-2], 2).float()
        idx = torch.stack(((key >> 16) & 0xFFFF, key & 0xFFFF), -1).flatten(1)  # [N, 2E]
        pres = (key != 0).float()
        w = torch.stack((pres, val[..., 0], val[..., 1]), 0).repeat_interleave(2, -1)  # [3, N, 2E]
        N, L = idx.shape
        de = self.table.shape[1]
        crow = torch.arange(0, 3 * N * L + 1, L, device=idx.device)
        W = torch.sparse_csr_tensor(crow, idx.flatten().repeat(3), w.flatten(), size=(3 * N, HASH_ROWS))
        with torch.autocast(device_type=key.device.type, enabled=False):
            out = W @ self.table
        return out.view(3, N, de).permute(1, 0, 2).reshape(*lead, 3 * de)

    def forward(self, b):
        """b: dict of tensors (encode.field_spec with a batch dim; T and C may be sliced).
        Returns masked logits [B,C] (fp32) and normalized value [B] (fp32)."""
        tok_type = b["tok_type"].long()
        B, T = tok_type.shape
        C = b["cand_key"].shape[1]
        dev = tok_type.device
        tmask = torch.arange(T, device=dev)[None] < b["n_tok"].long()[:, None]
        cmask = torch.arange(C, device=dev)[None] < b["n_cand"].long()[:, None]

        tok_bag = self.bag(b["tok_key"], b["tok_val"])
        x = self.type_emb(tok_type) + self.tok_in(tok_bag)
        for blk in self.blocks:
            x = blk(x, tmask)
        h = self.ln_f(x)
        g = h[:, 0]  # GLOBAL token is always first

        ref = b["cand_ref"].long()
        rmask = (ref >= 0) & cmask[..., None]
        ridx = ref.clamp(min=0)
        c = self.cand_in(self.bag(b["cand_key"], b["cand_val"])) + self.ref_global(g)[:, None]
        c = c + _gather(self.ref_primary(h), ridx[..., 0]) * rmask[..., 0:1]
        c = c + _gather(self.ref_target(h), ridx[..., 1]) * rmask[..., 1:2]
        R = ref.shape[-1]
        if R > 2:
            m = _gather(self.ref_member(h), ridx[..., 2:].reshape(B, -1)).view(B, C, R - 2, -1)
            mm = rmask[..., 2:, None].to(m.dtype)
            c = c + (m * mm).sum(2) / mm.sum(2).clamp(min=1)
        c = c.to(h.dtype)
        for blk in self.cand_blocks:
            c = blk(c, cmask, h, tmask)
        logits = self.logit(c).squeeze(-1).float().masked_fill(~cmask, NEG)

        if self.cfg.value_tower:
            x = self.value_type_emb(tok_type) + self.value_tok_in(tok_bag)
            for blk in self.value_blocks:
                x = blk(x, tmask)
            h = self.value_ln_f(x)
            g = h[:, 0]
        tm = tmask[..., None].to(h.dtype)
        pooled = (h * tm).sum(1) / tm.sum(1).clamp(min=1)
        value = self.value(torch.cat((g, pooled), -1)).squeeze(-1).float()
        return logits, value


def n_params(m):
    return sum(p.numel() for p in m.parameters())
