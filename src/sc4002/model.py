"""Joint intent + slot model: pre-LN Transformer encoder over (sub)word units with
frozen pretrained vectors, a trainable unit embedding and a char-CNN; attention+max
pooled intent head; intent-conditioned slot head with a BIO-constrained linear-chain CRF."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


# ---------------------------------------------------------------- CRF


def bio_illegal(tags: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """illegal[i, j]: tag i -> tag j is invalid BIO; start_illegal[j]: tag j cannot start."""
    n = len(tags)
    illegal = torch.zeros(n, n, dtype=torch.bool)
    start = torch.zeros(n, dtype=torch.bool)
    for j, t in enumerate(tags):
        if t.startswith("I-"):
            start[j] = True
            for i, s in enumerate(tags):
                illegal[i, j] = s[2:] != t[2:] or s == "O"
    return illegal, start


class CRF(nn.Module):
    def __init__(self, tags: list[str]):
        super().__init__()
        n = len(tags)
        self.trans = nn.Parameter(torch.empty(n, n).uniform_(-0.1, 0.1))
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))
        illegal, start_illegal = bio_illegal(tags)
        self.register_buffer("trans_pen", illegal.float() * NEG, persistent=False)
        self.register_buffer("start_pen", start_illegal.float() * NEG, persistent=False)

    def potentials(self):
        return self.trans + self.trans_pen, self.start + self.start_pen, self.end

    def nll(self, em: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sum over the batch of -log p(tags | em). mask is a left-aligned (B, W) bool."""
        T, S, E = self.potentials()
        B, W, _ = em.shape
        lengths = mask.sum(1)
        # gold path score
        score = S[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, W):
            step = T[tags[:, t - 1], tags[:, t]] + em[:, t].gather(1, tags[:, t : t + 1]).squeeze(1)
            score = score + step * mask[:, t]
        last = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        score = score + E[last]
        # log partition
        alpha = S + em[:, 0]
        for t in range(1, W):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + T.unsqueeze(0), dim=1) + em[:, t]
            alpha = torch.where(mask[:, t : t + 1], nxt, alpha)
        log_z = torch.logsumexp(alpha + E, dim=1)
        return (log_z - score).sum()

    @torch.no_grad()
    def viterbi(self, em: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        T, S, E = self.potentials()
        B, W, n = em.shape
        score = S + em[:, 0]
        back = []
        ident = torch.arange(n, device=em.device).expand(B, n)
        for t in range(1, W):
            best, idx = (score.unsqueeze(2) + T.unsqueeze(0)).max(dim=1)
            m = mask[:, t : t + 1]
            score = torch.where(m, best + em[:, t], score)
            back.append(torch.where(m, idx, ident))
        score = score + E
        best_last = score.argmax(1)
        back = torch.stack(back, 1).cpu() if back else None
        lengths = mask.sum(1).tolist()
        out = []
        for b in range(B):
            path = [best_last[b].item()]
            for t in range(W - 2, -1, -1):
                path.append(back[b, t, path[-1]].item())
            out.append(path[::-1][: lengths[b]])
        return out


# ---------------------------------------------------------------- encoder


def rel_bucket(rel: torch.Tensor, num_buckets: int = 32, max_distance: int = 64) -> torch.Tensor:
    """T5-style bidirectional relative-position buckets."""
    nb = num_buckets // 2
    ret = (rel > 0).long() * nb
    n = rel.abs()
    max_exact = nb // 2
    is_small = n < max_exact
    large = max_exact + (
        torch.log(n.float().clamp(min=1) / max_exact) / math.log(max_distance / max_exact) * (nb - max_exact)
    ).long()
    large = large.clamp(max=nb - 1)
    return ret + torch.where(is_small, n, large)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = attn_dropout

    def forward(self, x, bias):
        B, U, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, U, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.attn_dropout if self.training else 0.0
        )
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, U, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class CharCNN(nn.Module):
    def __init__(self, n_chars: int, dim: int, filters: int, width: int = 3):
        super().__init__()
        self.emb = nn.Embedding(n_chars, dim, padding_idx=0)
        self.conv = nn.Conv1d(dim, filters, width, padding=width // 2)

    def forward(self, chars):  # (B, U, C) -> (B, U, filters)
        B, U, C = chars.shape
        x = self.emb(chars.view(B * U, C)).transpose(1, 2)
        h = self.conv(x).masked_fill((chars.view(B * U, 1, C) == 0), -1e4).max(dim=2).values
        return torch.tanh(h).view(B, U, -1)


def masked_softmax_pool(h, scores, mask):
    w = scores.masked_fill(~mask, -1e4).softmax(dim=1).unsqueeze(2)
    return (w * h).sum(1)


class JointTransformer(nn.Module):
    def __init__(self, cfg, n_units: int, n_chars: int, tags: list[str], n_intents: int,
                 vectors=None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        in_dim = 0
        self.vec = None
        if vectors is not None and vectors.shape[1] > 0:
            self.vec = nn.Embedding.from_pretrained(torch.as_tensor(vectors), freeze=True, padding_idx=0)
            in_dim += vectors.shape[1]
        self.unit_emb = nn.Embedding(n_units, cfg.d_task, padding_idx=0)
        in_dim += cfg.d_task
        self.char = CharCNN(n_chars, cfg.char_dim, cfg.char_filters) if cfg.char_filters else None
        in_dim += cfg.char_filters
        self.emb_drop = nn.Dropout(cfg.emb_dropout)
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d))
        self.pos = cfg.positions
        assert self.pos in ("relative", "learned"), self.pos
        if self.pos == "learned":
            self.pos_emb = nn.Embedding(cfg.max_len, d)
        else:
            self.rel = nn.Embedding(32, cfg.heads)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(d, cfg.heads, cfg.ffn, cfg.dropout, cfg.attn_dropout) for _ in range(cfg.layers))
        self.ln_f = nn.LayerNorm(d)
        # intent head
        self.pool_score = nn.Linear(d, 1)
        self.intent_mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, n_intents))
        # slot head
        self.intent_ctx = nn.Embedding(n_intents, cfg.d_intent_ctx) if cfg.intent_cond else None
        self.slot_mlp = nn.Sequential(
            nn.Linear(d + (cfg.d_intent_ctx if cfg.intent_cond else 0), d), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(d, len(tags)))
        self.crf = CRF(tags) if cfg.decoder == "crf" else None

    def forward(self, b):
        umask = b["umask"]
        feats = []
        if self.vec is not None:
            rows = b["vec"]
            have = (rows > 0).float().unsqueeze(-1)
            v = (self.vec(rows) * have).sum(2) / have.sum(2).clamp(min=1)
            feats.append(v)
        feats.append(self.unit_emb(b["units"]))
        if self.char is not None:
            feats.append(self.char(b["chars"]))
        x = self.proj(self.emb_drop(torch.cat(feats, -1)))
        B, U, _ = x.shape
        if self.pos == "learned":
            x = x + self.pos_emb.weight[:U]
        x = self.drop(x)
        bias = torch.zeros(B, 1, U, U, device=x.device).masked_fill(~umask[:, None, None, :], NEG)
        if self.pos == "relative":
            pos = torch.arange(U, device=x.device)
            rb = self.rel(rel_bucket(pos[None, :] - pos[:, None])).permute(2, 0, 1)  # H,U,U
            bias = bias + rb.unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, bias)
        h = self.ln_f(x)
        # intent
        pooled = torch.cat([
            masked_softmax_pool(h, self.pool_score(h).squeeze(-1), umask),
            h.masked_fill(~umask.unsqueeze(-1), -1e4).max(1).values,
        ], -1)
        intent_logits = self.intent_mlp(pooled)
        # slots, read at each word's first unit
        hw = h.gather(1, b["first"].unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        if self.intent_ctx is not None:
            ctx = intent_logits.softmax(-1) @ self.intent_ctx.weight
            hw = torch.cat([hw, ctx.unsqueeze(1).expand(-1, hw.shape[1], -1)], -1)
        emissions = self.slot_mlp(hw)
        return intent_logits, emissions

    def loss(self, b, label_smoothing: float = 0.0):
        intent_logits, em = self(b)
        l_int = F.cross_entropy(intent_logits, b["intents"], label_smoothing=label_smoothing)
        n_words = b["wmask"].sum()
        if self.crf is not None:
            l_slot = self.crf.nll(em, b["tags"], b["wmask"]) / n_words
        else:
            l_slot = F.cross_entropy(em[b["wmask"]], b["tags"][b["wmask"]])
        return l_int + l_slot, l_int.detach(), l_slot.detach()

    @torch.no_grad()
    def predict(self, b):
        intent_logits, em = self(b)
        intents = intent_logits.argmax(-1).tolist()
        if self.crf is not None:
            tags = self.crf.viterbi(em, b["wmask"])
        else:
            lengths = b["wmask"].sum(1).tolist()
            tags = [row[:n] for row, n in zip(em.argmax(-1).tolist(), lengths)]
        return intents, tags


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
