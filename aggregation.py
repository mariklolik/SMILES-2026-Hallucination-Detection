from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import MAX_LENGTH


_LOCK = threading.Lock()
_INIT = False
_TOK = None
_PROMPT_LENS: list[int] = []
_COUNTER = [0]
_ATTN_FEATS: np.ndarray | None = None
_HIDDEN_L15_FEATS: np.ndarray | None = None
F_PER_HEAD = 8

LAYERS = (11, 12, 13, 14, 15, 16)
HIDDEN_LAYER = 15

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _ensure_init() -> None:
    global _INIT, _TOK, _ATTN_FEATS, _HIDDEN_L15_FEATS
    if _INIT:
        return
    with _LOCK:
        if _INIT:
            return
        df_tr = pd.read_csv(_DATA_DIR / "dataset.csv")
        df_te = pd.read_csv(_DATA_DIR / "test.csv")
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        if _TOK.pad_token is None:
            _TOK.pad_token = _TOK.eos_token
        for p in pd.concat([df_tr["prompt"], df_te["prompt"]], ignore_index=True):
            ids = _TOK(p, add_special_tokens=False, return_tensors=None)["input_ids"]
            _PROMPT_LENS.append(min(len(ids), MAX_LENGTH))
        texts = (
            [f"{r['prompt']}{r['response']}" for _, r in df_tr.iterrows()]
            + [f"{r['prompt']}{r['response']}" for _, r in df_te.iterrows()]
        )
        _ATTN_FEATS, _HIDDEN_L15_FEATS = _extract(texts, _PROMPT_LENS)
        _INIT = True


@torch.no_grad()
def _extract(texts, prompt_lens, batch_size: int = 4):
    print(f"[aggregation] dual-feature pass over {len(texts)} samples ...", flush=True)
    t0 = time.time()
    from transformers import AutoModelForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        output_hidden_states=True,
    ).to(device).eval()
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    D = model.config.hidden_size

    attn_out = np.zeros((len(texts), L, H, F_PER_HEAD), dtype=np.float32)
    h15_out = np.zeros((len(texts), D + 5), dtype=np.float32)

    for s in range(0, len(texts), batch_size):
        tb = texts[s : s + batch_size]
        enc = _TOK(tb, return_tensors="pt", padding="max_length",
                   truncation=True, max_length=MAX_LENGTH)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        result = model(input_ids=ids, attention_mask=am,
                       output_attentions=True, output_hidden_states=True)
        attns = torch.stack(result.attentions, dim=1)
        # hidden_states is tuple of L+1; index 16 = layer 15 (post-block) since hs[0]=embedding, hs[k]=after block k-1
        # actually hidden_states[k] is after the k-th block (k=0 is embedding), so layer 15 => index 16
        # Some HF models index differently; we use index HIDDEN_LAYER+1 to mean "after block HIDDEN_LAYER"
        hs15 = result.hidden_states[HIDDEN_LAYER + 1]
        am_cpu = am.cpu().numpy()
        for b in range(len(tb)):
            n = s + b
            real = np.flatnonzero(am_cpu[b])
            n_real = len(real)
            pl = min(prompt_lens[n], n_real)
            if pl == 0 or pl >= n_real:
                continue
            resp = real[pl:]
            prompt = real[:pl]
            if len(resp) == 0 or len(prompt) == 0:
                continue
            A = attns[b]
            A_resp = A[:, :, resp, :]
            A_to_prompt = A_resp[:, :, :, prompt].sum(dim=-1)
            A_to_resp = A_resp[:, :, :, resp].sum(dim=-1)
            denom = (A_to_prompt + A_to_resp).clamp(min=1e-9)
            lookback = A_to_prompt / denom
            attn_to_sink = A_resp[:, :, :, real[0:1]].sum(dim=-1)
            A_resp_real = A[:, :, resp][..., real]
            A_resp_real = A_resp_real / A_resp_real.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            ent = -(A_resp_real * (A_resp_real + 1e-12).log()).sum(dim=-1)
            lb = lookback.float().cpu().numpy()
            en = ent.float().cpu().numpy()
            sink = attn_to_sink.float().cpu().numpy()
            self_resp = A_to_resp.float().cpu().numpy()
            attn_out[n, :, :, 0] = lb.mean(axis=-1)
            attn_out[n, :, :, 1] = lb.min(axis=-1)
            attn_out[n, :, :, 2] = lb.max(axis=-1)
            attn_out[n, :, :, 3] = en.mean(axis=-1)
            attn_out[n, :, :, 4] = en.min(axis=-1)
            attn_out[n, :, :, 5] = en.max(axis=-1)
            attn_out[n, :, :, 6] = sink.mean(axis=-1)
            attn_out[n, :, :, 7] = self_resp.mean(axis=-1)
            # L15 response-token mean-pool, EOS dropped
            end = n_real - 1 if (n_real - 1 > pl) else n_real
            idx_pool = real[pl:end] if end > pl else real[max(0, n_real-8):]
            pool = hs15[b, idx_pool].float().mean(dim=0).cpu().numpy()
            h15_out[n, :D] = pool
            h15_out[n, D + 0] = n_real
            h15_out[n, D + 1] = max(0, n_real - pl)
            h15_out[n, D + 2] = pl
            h15_out[n, D + 3] = float(np.log1p(max(0, n_real - pl)))
            h15_out[n, D + 4] = float(np.log1p(pl))
        del attns, result
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[aggregation] done in {time.time()-t0:.1f}s", flush=True)
    return attn_out, h15_out


def aggregate(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    _ensure_init()
    counter = _COUNTER[0]
    out_dim = len(LAYERS) * 14 * F_PER_HEAD + 896 + 5
    if _ATTN_FEATS is None or _HIDDEN_L15_FEATS is None or counter >= _ATTN_FEATS.shape[0]:
        return torch.zeros(out_dim, dtype=torch.float32)
    sub = _ATTN_FEATS[counter][list(LAYERS)].reshape(-1).astype(np.float32)
    h = _HIDDEN_L15_FEATS[counter].astype(np.float32)
    return torch.from_numpy(np.concatenate([sub, h]))


def extract_geometric_features(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    out = aggregate(hidden_states, attention_mask)
    _COUNTER[0] += 1
    return out
