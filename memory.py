import argparse
import csv
import datetime
import json
import math
import os
import random
import time
import wandb
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


# =============================================================================
# Tokenizer and synthetic dataset
# =============================================================================
class IdentityTokenizer:
    """Synthetic data tokenizer: integers are already token ids."""

    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str):
        raise NotImplementedError("IdentityTokenizer encodes pre-tokenized ids only.")

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(f"id_{int(i)}" for i in ids)


class UniformRandomSequenceDataset(IterableDataset):
    """
    Fixed-length uniform random token sequences.
    Data are regenerated deterministically each epoch using the same seed.
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        for _ in range(self.num_samples):
            yield torch.randint(
                low=0,
                high=self.vocab_size,
                size=(self.seq_len,),
                generator=g,
                dtype=torch.long,
            )

    def __len__(self):
        return self.num_samples


# =============================================================================
# GPT-style LM (minimal GPT-2-like)
# =============================================================================
class GPT2Config:
    def __init__(
        self,
        vocab_size: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_every_two(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(y)
        return self.c_proj(y)


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.LongTensor, labels: torch.LongTensor | None = None):
        x = self.wte(input_ids)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.view(-1)
            )
        return logits, loss


# =============================================================================
# ESN language model
# =============================================================================
class ESNLanguageModel(nn.Module):
    """
    ESN_mlr variant with shared token embedding.
    Only A, B, and bias are learnable; reservoir weights stay fixed.
    """

    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        reservoir_size: int,
        d: int = 32,
        spectral_radius: float = 0.99,
        sigma_in: float = 1.0,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        activation: str = "tanh",
        dropout: float = 0.1,
        r_out: int = 512,
        max_frobenius_norm: float = 150.0,
        max_retries: int = 20,
        device: torch.device | None = None,
    ):
        super().__init__()
        device = device or torch.device("cpu")
        self.vocab_size = vocab_size
        self.reservoir_size = reservoir_size
        self.d = d
        self.gamma = d / float(self.reservoir_size)
        self.sigma_in = sigma_in
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.activation = activation
        self.dropout = dropout
        self.r_out = r_out
        self.spectral_radius = spectral_radius
        self.max_frobenius_norm = max_frobenius_norm
        self.max_retries = max_retries

        self.tok_emb = nn.Embedding(vocab_size, d_embed, device=device)
        a = torch.empty(self.reservoir_size, device=device).uniform_(alpha_min, alpha_max)
        self.register_buffer("a", a)

        W_in = self._rand_sparse((d_embed, self.reservoir_size), self.gamma, self.sigma_in, device)
        self.register_buffer("W_in_T", W_in.transpose(0, 1).coalesce())

        W_rec, fro_norm = self._init_reservoir(
            (self.reservoir_size, self.reservoir_size),
            self.gamma,
            spectral_radius,
            max_frobenius_norm,
            max_retries,
            device,
        )
        self.register_buffer("W_rec", W_rec.coalesce())
        self.W_rec_fro_norm = float(fro_norm)

        self.B = nn.Linear(self.reservoir_size, r_out, bias=False, device=device)
        self.A = nn.Linear(r_out, vocab_size, bias=True, device=device)
        self.drop = nn.Dropout(dropout)
        if activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError("activation must be tanh")

        self.register_buffer("h0", torch.zeros(self.reservoir_size, device=device))

    @staticmethod
    def _rand_sparse(shape, density, scale, device):
        rows, cols = shape
        nnz = int(round(rows * cols * density))
        row_idx = torch.randint(rows, (nnz,), device=device)
        col_idx = torch.randint(cols, (nnz,), device=device)
        vals = torch.randn(nnz, device=device) * scale
        idx = torch.stack([row_idx, col_idx])
        return torch.sparse_coo_tensor(idx, vals, shape, device=device)

    @staticmethod
    @torch.no_grad()
    def _scale_spectral_radius(mat, target_rho, iters=500):
        v = torch.randn(mat.size(0), 1, device=mat.device)
        v /= v.norm() + 1e-9
        for _ in range(iters):
            v = torch.sparse.mm(mat, v)
            v /= v.norm() + 1e-9
        cur_rho = torch.dot(v.squeeze(), torch.sparse.mm(mat, v).squeeze()).abs()
        mat = mat.coalesce()
        new_vals = mat.values() * (target_rho / (cur_rho + 1e-9))
        return torch.sparse_coo_tensor(mat.indices(), new_vals, mat.size(), device=mat.device)

    @torch.no_grad()
    def _init_reservoir(
        self,
        shape: Tuple[int, int],
        density: float,
        spectral_radius: float,
        max_frobenius_norm: float,
        max_retries: int,
        device: torch.device,
    ):
        for attempt in range(max_retries):
            W = self._rand_sparse(shape, density, 1.0, device)
            W = self._scale_spectral_radius(W, spectral_radius)
            fro = torch.linalg.norm(W.to_dense(), ord="fro").item()
            #fro = torch.linalg.norm(W.values(), ord=2).item()
            if fro < max_frobenius_norm:
                return W, fro
        raise RuntimeError(
            f"Failed to sample reservoir within Frobenius norm budget {max_frobenius_norm}"
        )

    def forward(self, x: torch.LongTensor, labels: torch.LongTensor | None = None):
        B, T = x.shape
        h = self.h0.expand(B, -1)
        a = self.a.unsqueeze(0)
        outs = []

        for t in range(T):
            emb_t = self.tok_emb(x[:, t])  # (B, d_embed)
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                emb_proj = torch.sparse.mm(self.W_in_T.float(), emb_t.float().t()).t()
                rec = torch.sparse.mm(self.W_rec.float(), h.float().t()).t()
            emb_proj = emb_proj.to(h.dtype)
            rec = rec.to(h.dtype)
            pre = (emb_proj + rec).clamp_(-10.0, 10.0)
            h = (1 - a) * h + a * self.act(pre)
            out = self.A(self.drop(self.B(h)))
            outs.append(out)

        logits = torch.stack(outs, dim=1)
        if labels is None:
            return logits

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.vocab_size),
            labels[:, 1:].reshape(-1),
        )
        return logits, loss


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_gpt_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# def count_esn_params(model: ESNLanguageModel) -> int:
#     p_tok_emb = model.tok_emb.weight.numel()
#     nnz_in = model.W_in_T._nnz()
#     nnz_rec = model.W_rec._nnz()
#     p_B = model.B.weight.numel()
#     p_A = model.A.weight.numel() + (model.A.bias.numel() if model.A.bias is not None else 0)
#     return p_tok_emb + nnz_in + nnz_rec + p_A + p_B


def count_esn_params(vocab_size, d_embed, reservoir_size, r_out, d):
    gamma = d / float(reservoir_size)
    # _rand_sparse の nnz 計算と揃える
    nnz_in  = int(round(d_embed * reservoir_size * gamma))
    nnz_rec = int(round(reservoir_size * reservoir_size * gamma))
    p_tok_emb = vocab_size * d_embed
    p_B = reservoir_size * r_out
    p_A = r_out * vocab_size + vocab_size  # bias
    return p_tok_emb + nnz_in + nnz_rec + p_A + p_B

def make_dataloader(num_samples: int, vocab_size: int, seq_len: int, batch_size: int, seed: int):
    dataset = UniformRandomSequenceDataset(vocab_size, seq_len, num_samples, seed)

    def collate(batch):
        data = torch.stack(batch)
        return data, data

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate_loss(model: nn.Module, dataloader: DataLoader, device, use_amp: bool = False):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(inputs, labels=labels)
        B, T = inputs.shape
        total_loss += loss.item() * (B * (T - 1))
        total_tokens += B * (T - 1)
    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    max_steps: int,
    max_epochs: int,
    patience: int,
    use_amp: bool,
    wandb_run = None,
):
    model.train()
    best_val = float("inf")
    best_val_epoch = -1
    steps = 0
    validation_epochs_every = 0
    
    steps_per_epoch = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
    loss_history = []
    for epoch in range(max_epochs):
        model.train()
        for inputs, labels in train_loader:
            if steps >= max_steps:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                _, train_loss = model(inputs, labels=labels)
            loss_history.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss_step": train_loss.item(),
                        "train/step": steps,
                        "train/epoch_float": epoch + steps / max(steps_per_epoch, 1),
                    }
                )
            if steps % 500 == 0:
                print(f"[log] Step {steps} Loss: {train_loss.item()}")
        validation_epochs_every += 1
        if epoch % validation_epochs_every == 0:
            val_loss = evaluate_loss(model, val_loader, device, use_amp)
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_val_epoch = epoch
            elif patience > 0 and (epoch - best_val_epoch) >= patience:
                break
        if steps >= max_steps:
            break
        if steps_per_epoch == 0:
            break
        
    train_loss_full = evaluate_loss(model, train_loader, device, use_amp)
    if wandb_run is not None:
        wandb_run.log(
            {
                "train/loss_final": train_loss_full,
                "val/loss_best": best_val,
                "train/steps_total": steps,
            }
        )
    print(f"train_loss_final: {train_loss_full}",
        f"val_loss_best: {best_val}",
        f"steps: {steps}",
        f"loss_history: {loss_history}")
    
    return {
        "train_loss_final": train_loss_full,
        "val_loss_best": best_val,
        "steps": steps,
        "loss_history": loss_history,
    }


# =============================================================================
# Experiment driver
# =============================================================================
GPT_SPECS = {
    "S": {"n_layer": 1, "n_embd": 64, "n_head": 4},
    "M": {"n_layer": 2, "n_embd": 128, "n_head": 4},
    "L": {"n_layer": 4, "n_embd": 128, "n_head": 4},
    "XL": {"n_layer": 8, "n_embd": 256, "n_head": 8},
}

ESN_SPECS = {
    "S":  {"d_embed": 64,  "reservoir_size":  276,  "r_out":  16,  "d_nonzero": 32},
    "M":  {"d_embed": 128, "reservoir_size": 7456,  "r_out":  16,  "d_nonzero": 32},
    "L":  {"d_embed": 128, "reservoir_size": 11276, "r_out":  32,  "d_nonzero": 32},
    "XL": {"d_embed": 256, "reservoir_size": 20084, "r_out": 256,  "d_nonzero": 32},
}

DATASET_SIZES = [
    16_000,
    32_000,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
    2_097_152,
    4_194_304,
    8_388_608,
]


@dataclass
class ESNConfig:
    reservoir_size: int
    r_out: int
    d_embed: int
    d: int
    spectral_radius: float
    sigma_in: float
    alpha_min: float
    alpha_max: float
    dropout: float

def choose_esn_config(
    target_params: int,
    vocab_size: int,
    d_embed: int,
    reservoir_candidates: List[int],
    r_out_candidates: List[int],
    d: int,
    spectral_radius: float,
    sigma_in: float,
    alpha_min: float,
    alpha_max: float,
    dropout: float,
    device,
):
    best = None
    best_gap = float("inf")
    best_res = None
    best_r_out = None
    best_p_esn = None

    for res in reservoir_candidates:
        for r_out in r_out_candidates:
            p_esn = count_esn_params(
                vocab_size=vocab_size,
                d_embed=d_embed,
                reservoir_size=res,
                r_out=r_out,
                d=d,
            )
            gap = abs(p_esn - target_params)
            if gap < best_gap:
                best_gap = gap
                best = (res, r_out, p_esn)
                best_res, best_r_out, best_p_esn = res, r_out, p_esn

    if best is None:
        raise RuntimeError("No ESN configuration found.")

    # Frobenius ノルムは「ベスト構成で 1 回だけ」サンプルしておく
    dummy = ESNLanguageModel(
        vocab_size=vocab_size,
        d_embed=d_embed,
        reservoir_size=best_res,
        d=d,
        spectral_radius=spectral_radius,
        sigma_in=sigma_in,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        dropout=dropout,
        r_out=best_r_out,
        device=device,
    )
    fro = dummy.W_rec_fro_norm
    del dummy

    rel_gap = abs(best_p_esn - target_params) / max(target_params, 1)
    return {
        "reservoir_size": best_res,
        "r_out": best_r_out,
        "num_params": best_p_esn,
        "relative_error": rel_gap,
        "W_rec_fro_norm": fro,
    }

def compute_mem_bits(train_loss: float, num_tokens: int, vocab_size: int):
    total_data_bits = num_tokens * math.log2(vocab_size)
    total_code_bits = (train_loss * num_tokens) / math.log(2.0)
    total_mem_bits = max(total_data_bits - total_code_bits, 0.0)
    return total_data_bits, total_code_bits, total_mem_bits


def aggregate_capacity(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in rows:
        key = (r["arch"], r["model_size"], r["seed"])
        grouped[key].append(float(r["total_mem_bits"]))
    per_run = {(k): max(v) for k, v in grouped.items()}
    agg = []
    by_model = defaultdict(list)
    for (arch, size, _), cap in per_run.items():
        by_model[(arch, size)].append(cap)
    for (arch, size), caps in by_model.items():
        caps_arr = np.array(caps, dtype=np.float64)
        agg.append(
            {
                "arch": arch,
                "model_size": size,
                "capacity_mean": float(caps_arr.mean()),
                "capacity_std": float(caps_arr.std(ddof=0)),
            }
        )
    return agg


def aggregate_data_efficiency(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["arch"], r["model_size"])].append(
            (math.log10(int(r["N"])), float(r["test_loss_final"]))
        )
    results = []
    for (arch, size), pts in grouped.items():
        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
        else:
            slope, intercept = float("nan"), float("nan")
        results.append({"arch": arch, "model_size": size, "slope": float(slope), "intercept": float(intercept)})
    return results


def plot_results(rows: List[Dict], capacity: List[Dict], output_dir: Path):
    colors = {"S": "tab:blue", "M": "tab:orange", "L": "tab:green", "XL": "tab:red"}
    fig1, ax1 = plt.subplots()
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["arch"], r["model_size"], int(r["N"]))].append(float(r["total_mem_bits"]))
    for key, vals in grouped.items():
        arch, size, N = key
        grouped[key] = np.mean(vals)
    for arch in ["GPT", "ESN"]:
        for size in ["S", "M", "L", "XL"]:
            xs, ys = [], []
            for N in DATASET_SIZES:
                val = grouped.get((arch, size, N))
                if val is not None:
                    xs.append(N)
                    ys.append(val)
            if xs:
                ax1.plot(
                    xs,
                    ys,
                    label=f"{arch}-{size}",
                    color=colors.get(size, None),
                    linestyle="-" if arch == "GPT" else "--",
                    marker="o",
                )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("N (train samples)")
    ax1.set_ylabel("total_mem_bits (mean over seeds)")
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    fig1.tight_layout()
    fig1.savefig(output_dir / "total_mem_bits_vs_N.png", dpi=200)

    fig2, ax2 = plt.subplots()
    for cap in capacity:
        arch = cap["arch"]
        size = cap["model_size"]
        num_params = cap.get("num_params")
        if num_params is None:
            continue
        ax2.scatter(
            [num_params],
            [cap["capacity_mean"]],
            label=f"{arch}-{size}",
            color=colors.get(size, None),
            marker="o" if arch == "GPT" else "x",
        )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("num_params")
    ax2.set_ylabel("capacity_mean")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    handles, labels = ax2.get_legend_handles_labels()
    if labels:
        ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / "capacity_vs_num_params.png", dpi=200)

    fig3, ax3 = plt.subplots()
    grouped_loss = defaultdict(list)
    for r in rows:
        grouped_loss[(r["arch"], r["model_size"], int(r["N"]))].append(float(r["test_loss_final"]))
    grouped_loss = {k: np.mean(v) for k, v in grouped_loss.items()}
    for arch in ["GPT", "ESN"]:
        for size in ["S", "M", "L", "XL"]:
            xs, ys = [], []
            for N in DATASET_SIZES:
                val = grouped_loss.get((arch, size, N))
                if val is not None:
                    xs.append(N)
                    ys.append(val)
            if xs:
                ax3.plot(
                    xs,
                    ys,
                    label=f"{arch}-{size}",
                    color=colors.get(size, None),
                    linestyle="-" if arch == "GPT" else "--",
                    marker="o",
                )
    ax3.set_xscale("log")
    ax3.set_xlabel("N (train samples)")
    ax3.set_ylabel("test_loss (nats/token)")
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.4)
    fig3.tight_layout()
    fig3.savefig(output_dir / "test_loss_vs_N.png", dpi=200)


def run_experiments(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"

    vocab_size = args.vocab_size
    seq_len = args.seq_len
    S = seq_len - 1
    tokenizer = IdentityTokenizer(vocab_size)

    gpt_param_counts = {}
    esn_param_counts = {}
    esn_choices = {}

    for size_name, cfg in GPT_SPECS.items():
        gpt_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=seq_len,
            n_embd=cfg["n_embd"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
        )
        gpt_model = GPT2LMHeadModel(gpt_config)
        gpt_param_counts[size_name] = count_gpt_params(gpt_model)

        esn_spec = ESN_SPECS[size_name]
        p_esn = count_esn_params(
            vocab_size=vocab_size,
            d_embed=esn_spec["d_embed"],
            reservoir_size=esn_spec["reservoir_size"],
            r_out=esn_spec["r_out"],
            d=esn_spec["d_nonzero"],
        )
        esn_cfg = {
            "d_embed": esn_spec["d_embed"],
            "reservoir_size": esn_spec["reservoir_size"],
            "r_out": esn_spec["r_out"],
            "d_nonzero": esn_spec["d_nonzero"],
            "num_params": p_esn,
        }
        esn_choices[size_name] = esn_cfg
        esn_param_counts[size_name] = p_esn

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "model_size",
                "num_params",
                "N",
                "seed",
                "train_loss_final",
                "val_loss_best",
                "test_loss_final",
                "total_tokens",
                "total_data_bits",
                "total_code_bits",
                "total_mem_bits",
                "bpp",
                "W_rec_fro_norm",
                "steps",
            ],
        )
        writer.writeheader()

    results_rows = []
    start_time = time.time()
    seeds = list(range(args.num_seeds))
    selected_sizes = args.sizes              # 例: ["S", "M"]
    selected_archs = args.archs              # 例: ["GPT"] or ["ESN"] or ["GPT", "ESN"]
    for arch in selected_archs:
        for size_name in selected_sizes:
            cfg = GPT_SPECS[size_name]
            for N in DATASET_SIZES:
                for seed in seeds:
                    set_seed(seed)
                    train_loader = make_dataloader(
                        num_samples=N,
                        vocab_size=vocab_size,
                        seq_len=seq_len,
                        batch_size=args.batch_size,
                        seed=seed,
                    )
                    val_loader = make_dataloader(
                        num_samples=args.eval_samples,
                        vocab_size=vocab_size,
                        seq_len=seq_len,
                        batch_size=args.batch_size,
                        seed=10_000 + seed,
                    )
                    test_loader = make_dataloader(
                        num_samples=args.eval_samples,
                        vocab_size=vocab_size,
                        seq_len=seq_len,
                        batch_size=args.batch_size,
                        seed=20_000 + seed,
                    )

                    if arch == "GPT":
                        model = GPT2LMHeadModel(
                            GPT2Config(
                                vocab_size=vocab_size,
                                n_positions=seq_len,
                                n_embd=cfg["n_embd"],
                                n_layer=cfg["n_layer"],
                                n_head=cfg["n_head"],
                            )
                        ).to(device)
                        num_params = gpt_param_counts[size_name]
                        w_rec_fro_norm = None
                        print(f"{cfg}")
                    else:
                        esn_cfg = esn_choices[size_name]
                        model = ESNLanguageModel(
                            vocab_size=vocab_size,
                            d_embed=esn_cfg["d_embed"],
                            reservoir_size=esn_cfg["reservoir_size"],
                            d=esn_cfg["d_nonzero"],
                            spectral_radius=args.spectral_radius,
                            sigma_in=args.sigma_in,
                            alpha_min=args.alpha_min,
                            alpha_max=args.alpha_max,
                            dropout=args.dropout,
                            r_out=esn_cfg["r_out"],
                            device=device,
                        )
                        num_params = esn_cfg["num_params"]
                        w_rec_fro_norm = model.W_rec_fro_norm
                        print(
                            f"{esn_cfg}"
                        )
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=args.lr,
                        betas=(0.9, 0.95),
                        weight_decay=args.weight_decay,
                    )
                    if args.wandb:
                        wandb_run = wandb.init(
                            project="esn-data-efficiency_abci", 
                            name=f"{arch}-{size_name}-N{N}-seed{seed}",
                            config={
                                    "arch": arch,
                                    "model_size": size_name,
                                    "N": N,
                                    "seed": seed,
                                    "vocab_size": vocab_size,
                                    "seq_len": seq_len,
                                    "batch_size": args.batch_size,
                                    "lr": args.lr,
                                    "weight_decay": args.weight_decay,
                                    "max_steps": args.max_steps,
                                    "max_epochs": args.max_epochs,
                                    "patience": args.patience,
                                    "spectral_radius": args.spectral_radius,
                                    "sigma_in": args.sigma_in,
                                    "alpha_min": args.alpha_min,
                                    "alpha_max": args.alpha_max,
                                    "dropout": args.dropout,
                                    "num_params": num_params,
                                },
                            )
                    train_stats = train_model(
                        model=model,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        max_steps=args.max_steps,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        use_amp=use_amp,
                        wandb_run=wandb_run,
                    )
                    test_loss = evaluate_loss(model, test_loader, device, use_amp)

                    total_tokens = N * S
                    total_data_bits, total_code_bits, total_mem_bits = compute_mem_bits(
                        train_stats["train_loss_final"], total_tokens, vocab_size
                    )
                    bpp = total_mem_bits / num_params if num_params > 0 else float("nan")
                    if args.wandb:
                        wandb_run.log(
                            {
                                "test/loss": test_loss,
                                "metrics/total_tokens": total_tokens,
                                "metrics/total_data_bits": total_data_bits,
                                "metrics/total_code_bits": total_code_bits,
                                "metrics/total_mem_bits": total_mem_bits,
                                "metrics/bpp": bpp,
                            }
                        )
                        wandb_run.finish()
                    row = {
                        "arch": arch,
                        "model_size": size_name,
                        "num_params": num_params,
                        "N": N,
                        "seed": seed,
                        "train_loss_final": train_stats["train_loss_final"],
                        "val_loss_best": train_stats["val_loss_best"],
                        "test_loss_final": test_loss,
                        "total_tokens": total_tokens,
                        "total_data_bits": total_data_bits,
                        "total_code_bits": total_code_bits,
                        "total_mem_bits": total_mem_bits,
                        "bpp": bpp,
                        "W_rec_fro_norm": w_rec_fro_norm,
                        "steps": train_stats["steps"],
                    }
                    results_rows.append(row)
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=row.keys())
                        writer.writerow(row)
                    elapsed = time.time() - start_time
                    print(
                        f"[{elapsed/60:.1f}m] {arch}-{size_name} N={N} seed={seed} "
                        f"train_loss={train_stats['train_loss_final']:.4f} "
                        f"test_loss={test_loss:.4f} bpp={bpp:.4f}"
                    )

    capacity = aggregate_capacity(results_rows)
    for entry in capacity:
        size = entry["model_size"]
        if entry["arch"] == "GPT":
            entry["num_params"] = gpt_param_counts[size]
        else:
            entry["num_params"] = esn_param_counts[size]

    data_eff = aggregate_data_efficiency(results_rows)

    summary = {
        "capacity": capacity,
        "data_efficiency": data_eff,
        "gpt_param_counts": gpt_param_counts,
        "esn_param_counts": esn_param_counts,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_results(results_rows, capacity, output_dir)


def parse_args():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Synthetic memorization capacity: GPT vs ESN (single-script runner)"
    )
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=9192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--max-epochs", type=int, default=5_000_000)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--eval-samples", type=int, default=10_000)
    parser.add_argument("--d-nonzero", type=int, default=32)
    parser.add_argument("--spectral-radius", type=float, default=0.99)
    parser.add_argument("--sigma-in", type=float, default=1.0)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        choices=["S", "M", "L", "XL"],
        default=["S", "M"],
        help="どのモデルサイズを走らせるか（複数指定可）",
    )
    parser.add_argument(
        "--archs",
        type=str,
        nargs="+",
        choices=["GPT", "ESN"],
        default=["ESN"],
        help="実行するアーキテクチャ: GPT, ESN, または両方を空白区切りで指定",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("experiments", f"{ts}_synthetic_memorization_full"),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("=== Parsed args ===")
    for k, v in vars(args).items():
        print(f"{k:15s}: {v}")
    print("====================")

    run_experiments(args)

