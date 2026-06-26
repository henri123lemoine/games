"""The three Ataraxos Stratego transformers in MLX.

Faithful to ATARAXOS_SPEC.md §3 and the reference `pyengine/networks/*`:
pre-LN blocks, ReLU FF with ff_factor=4, learned absolute positional embeddings
(trunc_normal std 0.1), LayerNorm eps 1e-5, Linear bias=True. Standard SDPA
(numerically equivalent to the reference's forced EFFICIENT_ATTENTION backend).

  - MoveTransformer  : encoder-only; key-query move head -> 1800 actions; value head.
  - ArrangementTransformer (setup): decoder-only causal; placement / value / entropy heads.
  - BeliefTransformer: encoder-decoder, dropout 0.2; per-hidden-piece type decoding.

Each net consumes the already-tokenized obs the Rust sim emits (the
FeatureOrchestrator work — plane filtering, piece-id one-hot, lake-cell drop — is
done sim-side). See README / the obs contract: move net takes (B, 92, 643) f32.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .action_map import ActionLogitMap
from .spec import (
    ARRANGEMENT_SIZE,
    BELIEF_IN_DIM,
    FLAG_IDX,
    MOVE_IN_DIM,
    N_ARRANGEMENT_COL,
    N_ARRANGEMENT_ROW,
    N_OCCUPIABLE_CELL,
    N_PIECE_TYPE,
    N_VF_CAT,
)

LN_EPS = 1e-5
POS_EMB_STD = 0.1
FF_FACTOR = 4


def _trunc_normal(shape, std):
    """trunc_normal_(std) approximated by clipping a normal to +/-2 std (PyTorch default)."""
    x = mx.random.normal(shape) * std
    return mx.clip(x, -2 * std, 2 * std)


def _sdpa(q, k, v, mask=None):
    """Scaled dot-product attention. q,k,v: (B, H, T, d_head). mask: additive (T_q, T_k).

    MLX's fused kernel: one pass, never materializes the (B,H,T,T) score matrix or
    round-trips it through fp32 — it does the fp32 softmax internally. Verified
    numerically faithful to the naive impl end-to-end (argmax identical, logits/value
    diff < 1e-3) and a notable speedup on the bandwidth-bound collect forward."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def _split_heads(x, n_head):
    b, t, d = x.shape
    return x.reshape(b, t, n_head, d // n_head).transpose(0, 2, 1, 3)


def _merge_heads(x):
    b, h, t, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * d)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x, mask=None):
        q = _split_heads(self.q_proj(x), self.n_head)
        k = _split_heads(self.k_proj(x), self.n_head)
        v = _split_heads(self.v_proj(x), self.n_head)
        return self.out_proj(_merge_heads(_sdpa(q, k, v, mask)))


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x, mem):
        q = _split_heads(self.q_proj(x), self.n_head)
        k = _split_heads(self.k_proj(mem), self.n_head)
        v = _split_heads(self.v_proj(mem), self.n_head)
        return self.out_proj(_merge_heads(_sdpa(q, k, v)))


class _FF(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, FF_FACTOR * d_model)
        self.linear2 = nn.Linear(FF_FACTOR * d_model, d_model)

    def __call__(self, x):
        return self.linear2(nn.relu(self.linear1(x)))


class SelfAttentionLayer(nn.Module):
    """Pre-LN self-attention block: residual MHSA then residual ReLU FF."""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.mha = MultiHeadSelfAttention(d_model, n_head)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.ff = _FF(d_model)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        x = x + self.drop(self.mha(self.layer_norm1(x), mask))
        x = x + self.drop(self.ff(self.layer_norm2(x)))
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.mha = MultiHeadCrossAttention(d_model, n_head)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.ff = _FF(d_model)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x, mem):
        x = x + self.drop(self.mha(self.layer_norm1(x), mem))
        x = x + self.drop(self.ff(self.layer_norm2(x)))
        return x


class DecoderBlock(nn.Module):
    """Causal self-attention layer + cross-attention layer (4 residual sub-blocks)."""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, n_head, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, n_head, dropout)

    def __call__(self, x, mem, causal_mask):
        x = self.self_attn(x, causal_mask)
        x = self.cross_attn(x, mem)
        return x


def _causal_mask(t, dtype=mx.float32):
    """Additive (t, t) mask: 0 on/below diagonal, -inf above (block attending future)."""
    idx = mx.arange(t)
    block = idx[None, :] > idx[:, None]
    return mx.where(block, mx.array(-1e9, dtype=dtype), mx.array(0.0, dtype=dtype))


# ----------------------------------------------------------------------------- #
# MOVE NET (encoder-only)
# ----------------------------------------------------------------------------- #
class MoveTransformer(nn.Module):
    """Encoder-only move net. ~14.7M at depth 8 / d 384 (config below scales it).

    forward(obs) -> dict with:
      move_logits  (B, 1800) mapped to the env action space, lake slots = -inf
      value_logp   (B, 3)    log-softmax over CATEGORICAL_AGGREGATION [-1,0,1]

    obs: (B, 92, 643) f32 from the sim (already orchestrated + lakes dropped).
    Pass legal_mask (B, 1800) bool to mask illegal actions to -inf.
    """

    @classmethod
    def from_config(cls, cfg):
        return cls(depth=cfg.depth, embed_dim=cfg.embed_dim, n_head=cfg.n_head)

    def __init__(self, depth=8, embed_dim=384, n_head=8, in_dim=MOVE_IN_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedder = nn.Linear(in_dim, embed_dim)
        # 92 cell tokens + 1 value token prepended -> 93 positions.
        self.positional_encoding = _trunc_normal(
            (1, N_OCCUPIABLE_CELL + 1, embed_dim), POS_EMB_STD
        )
        self.value_token = _trunc_normal((1, 1, embed_dim), POS_EMB_STD)
        self.layers = [SelfAttentionLayer(embed_dim, n_head) for _ in range(depth)]
        self.norm_out = nn.LayerNorm(embed_dim, eps=LN_EPS)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.value_head = nn.Linear(embed_dim, N_VF_CAT)
        self._action_map = ActionLogitMap()

    def trunk(self, obs):
        x = self.embedder(obs)  # (B, 92, d)
        b = x.shape[0]
        vtok = mx.broadcast_to(self.value_token, (b, 1, self.embed_dim))
        x = mx.concatenate([vtok, x], axis=1)  # value token at index 0
        x = x + self.positional_encoding
        for layer in self.layers:
            x = layer(x)
        return self.norm_out(x)

    def __call__(self, obs, legal_mask=None):
        out = self.trunk(obs)
        value_tok = out[:, 0, :]
        cells = out[:, 1:, :]  # (B, 92, d)
        q = self.q_proj(cells)
        k = self.k_proj(cells)
        grid = (q @ mx.swapaxes(k, 1, 2)) / math.sqrt(self.embed_dim)  # (B, 92, 92)
        move_logits = self._action_map.apply(grid)  # (B, 1800)
        if legal_mask is not None:
            fill = mx.finfo(move_logits.dtype).min
            move_logits = mx.where(legal_mask, move_logits, fill)
        value_logits = self.value_head(value_tok)
        value_logp = value_logits.astype(mx.float32)
        value_logp = value_logp - mx.logsumexp(value_logp, axis=-1, keepdims=True)
        return {"move_logits": move_logits, "value_logp": value_logp}


# ----------------------------------------------------------------------------- #
# SETUP NET (decoder-only, causal)
# ----------------------------------------------------------------------------- #
class ArrangementTransformer(nn.Module):
    """Decoder-only causal setup net. ~12.6M at depth 4 / d 512.

    forward(seq, piece_counts) -> dict with, per placement slot:
      logits   (B, 40, 14)  next-placement type, legal-type masked
      value    (B, 40, 3)   W/L/D value
      ent_pred (B, 40, 1)   conditional-entropy prediction

    seq: (B, T<=40, 14) one-hot placements (the prefix). A learned zero start
    token is prepended and the sequence truncated to 40 (so causally each output
    position predicts the *next* placement). piece_counts: (14,) remaining-type
    budget used for the legal mask.
    """

    @classmethod
    def from_config(cls, cfg):
        return cls(
            depth=cfg.depth,
            embed_dim=cfg.embed_dim,
            n_head=cfg.n_head,
            force_handedness=cfg.force_handedness,
        )

    def __init__(self, depth=4, embed_dim=512, n_head=8, force_handedness=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.force_handedness = force_handedness
        self.embedder = nn.Linear(N_PIECE_TYPE, embed_dim)
        self.positional_encoding = _trunc_normal(
            (1, ARRANGEMENT_SIZE, embed_dim), POS_EMB_STD
        )
        self.start_token = mx.zeros((1, 1, N_PIECE_TYPE))  # learned-zero start (fixed)
        self.layers = [SelfAttentionLayer(embed_dim, n_head) for _ in range(depth)]
        self.norm_out = nn.LayerNorm(embed_dim, eps=LN_EPS)
        self.policy_out = nn.Linear(embed_dim, N_PIECE_TYPE)
        self.value_out = nn.Linear(embed_dim, N_VF_CAT)
        self.ent_out = nn.Linear(embed_dim, 1)
        # right-half columns get the flag under force_handedness (cols 5..9 per row).
        right = N_ARRANGEMENT_ROW * (
            [False] * (N_ARRANGEMENT_COL // 2) + [True] * (N_ARRANGEMENT_COL // 2)
        )
        self._right_side = mx.array(right)  # (40,) bool

    def _legal_mask(self, seq, piece_counts):
        """(B, T, 14) bool: types with remaining budget; flag restricted to right half."""
        cum = mx.cumsum(seq, axis=1)
        remaining = piece_counts.reshape(1, 1, -1) - cum
        legal = remaining > 0
        if self.force_handedness:
            t = legal.shape[1]
            not_right = ~self._right_side[:t]  # (T,)
            flag_block = mx.broadcast_to(not_right[None, :, None], legal.shape)
            type_is_flag = (mx.arange(N_PIECE_TYPE) == FLAG_IDX).reshape(1, 1, -1)
            legal = legal & ~(flag_block & type_is_flag)
        return legal

    def __call__(self, seq, piece_counts):
        b, t, _ = seq.shape
        start = mx.broadcast_to(self.start_token, (b, 1, N_PIECE_TYPE))
        full = mx.concatenate([start, seq], axis=1)[:, :ARRANGEMENT_SIZE]
        x = self.embedder(full) + self.positional_encoding[:, : full.shape[1]]
        mask = _causal_mask(x.shape[1])
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm_out(x)
        logits = self.policy_out(x)
        legal = self._legal_mask(full, piece_counts)
        fill = mx.finfo(logits.dtype).min
        logits = mx.where(legal, logits, fill)
        return {
            "logits": logits,
            "value": self.value_out(x),
            "ent_pred": self.ent_out(x),
        }


# ----------------------------------------------------------------------------- #
# BELIEF NET (encoder-decoder, dropout 0.2)
# ----------------------------------------------------------------------------- #
class BeliefTransformer(nn.Module):
    """Encoder-decoder belief net. ~57M at full size; scaled to ~8-12M here.

    forward(obs, unknown_pos_onehot, unknown_type_onehot) -> (B, n_piece, 14) logits.

    Encoder runs over the 92 cell tokens (697-wide obs). Encoder embeddings at the
    squares holding unknown opponent pieces are gathered (row-major) as decoder
    memory. The decoder autoregressively predicts each hidden piece's type with
    teacher forcing (the true type one-hots, right-shifted by a zero start row).

    obs: (B, 92, 697) f32 from the sim (belief variant, 86 history planes).
    unknown_pos_onehot: (B, n_piece, 92) one-hot of each unknown piece's cell.
    unknown_type_onehot: (B, n_piece, 14) teacher-forcing target one-hots.
    """

    @classmethod
    def from_config(cls, cfg):
        return cls(
            n_encoder_layer=cfg.n_encoder_layer,
            n_decoder_block=cfg.n_decoder_block,
            embed_dim=cfg.embed_dim,
            n_head=cfg.n_head,
            n_piece=cfg.n_piece,
            dropout=cfg.dropout,
        )

    def __init__(
        self,
        n_encoder_layer=6,
        n_decoder_block=6,
        embed_dim=512,
        n_head=8,
        n_piece=40,
        dropout=0.2,
        in_dim=BELIEF_IN_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedder = nn.Linear(in_dim, embed_dim)
        self.positional_embed_enc = _trunc_normal(
            (1, N_OCCUPIABLE_CELL, embed_dim), POS_EMB_STD
        )
        self.piece_embed = nn.Linear(N_PIECE_TYPE, embed_dim)
        self.positional_embed_dec = _trunc_normal((1, n_piece, embed_dim), POS_EMB_STD)
        self.encoder = [
            SelfAttentionLayer(embed_dim, n_head, dropout) for _ in range(n_encoder_layer)
        ]
        self.decoder = [
            DecoderBlock(embed_dim, n_head, dropout) for _ in range(n_decoder_block)
        ]
        self.final_ln_enc = nn.LayerNorm(embed_dim, eps=LN_EPS)
        self.final_ln_dec = nn.LayerNorm(embed_dim, eps=LN_EPS)
        self.final_linear = nn.Linear(embed_dim, N_PIECE_TYPE)

    def encode(self, obs, unknown_pos_onehot):
        x = self.embedder(obs) + self.positional_embed_enc
        for layer in self.encoder:
            x = layer(x)
        # Gather encoder embeddings at the cell each unknown piece occupies.
        cell_idx = mx.argmax(unknown_pos_onehot, axis=-1)  # (B, n_piece)
        mem = mx.take_along_axis(x, cell_idx[..., None], axis=1)  # (B, n_piece, d)
        is_active = mx.max(unknown_pos_onehot, axis=-1, keepdims=True) > 0
        mem = mem * is_active  # zero out padding rows (no unknown piece)
        return self.final_ln_enc(mem)

    def decode(self, unknown_type_onehot, mem):
        # right-shift the teacher-forcing one-hots by a zero start row.
        b, n, _ = unknown_type_onehot.shape
        zero = mx.zeros((b, 1, N_PIECE_TYPE))
        shifted = mx.concatenate([zero, unknown_type_onehot[:, :-1, :]], axis=1)
        x = self.piece_embed(shifted) + self.positional_embed_dec[:, :n]
        mask = _causal_mask(n)
        for block in self.decoder:
            x = block(x, mem, mask)
        return self.final_linear(self.final_ln_dec(x))

    def __call__(self, obs, unknown_pos_onehot, unknown_type_onehot):
        mem = self.encode(obs, unknown_pos_onehot)
        return self.decode(unknown_type_onehot, mem)
