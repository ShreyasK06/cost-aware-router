"""Optional CPU-sized fine-tuning of a small sentence encoder (torch + transformers, lazy imports)."""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np


class FineTunedRouter:
    name = "Fine-tuned MiniLM"
    is_probabilistic = True

    def __init__(self, tokenizer, encoder, head, max_len: int = 128, name: str | None = None):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.head = head
        self.max_len = max_len
        if name:
            self.name = name

    def score_prompts(self, prompts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        import torch
        prompts = list(prompts)
        self.encoder.eval()
        self.head.eval()
        out = np.zeros(len(prompts), dtype=np.float64)
        order = np.argsort([len(p) for p in prompts])
        with torch.no_grad():
            for s in range(0, len(prompts), batch_size):
                ids = order[s:s + batch_size]
                logits = _forward(self.tokenizer, self.encoder, self.head,
                                  [prompts[i] for i in ids], self.max_len)
                out[ids] = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        return out

    def score(self, prompts, batch_size: int = 64) -> np.ndarray:
        return self.score_prompts(prompts, batch_size)


def _forward(tokenizer, encoder, head, texts, max_len):
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt")
    hidden = encoder(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return head(pooled).squeeze(-1)


def finetune_encoder(train_prompts, train_labels, val_prompts, val_labels,
                     model_name: str = "sentence-transformers/all-MiniLM-L6-v2", epochs: int = 1,
                     max_len: int = 128, batch_size: int = 32, lr: float = 5e-5, seed: int = 0,
                     max_train: int | None = None, log_every: int = 50) -> FineTunedRouter:
    import torch
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tp = list(train_prompts)
    ty = np.asarray(train_labels, dtype=np.float32)
    if max_train is not None and len(tp) > max_train:
        sel = np.sort(rng.choice(len(tp), size=max_train, replace=False))
        tp = [tp[i] for i in sel]
        ty = ty[sel]
    vp = list(val_prompts)
    vy = torch.tensor(np.asarray(val_labels, dtype=np.float32))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    head = torch.nn.Linear(encoder.config.hidden_size, 1)
    n_pos = float(ty.sum())
    n_neg = float(len(ty) - n_pos)
    pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    steps_per_epoch = max(1, (len(tp) + batch_size - 1) // batch_size)
    total = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    best_val, best_state = float("inf"), None
    step = 0
    for ep in range(epochs):
        encoder.train()
        head.train()
        perm = rng.permutation(len(tp))
        for s in range(0, len(tp), batch_size):
            ids = perm[s:s + batch_size]
            logits = _forward(tokenizer, encoder, head, [tp[i] for i in ids], max_len)
            loss = loss_fn(logits, torch.tensor(ty[ids]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if log_every and step % log_every == 0:
                print(f"[finetune] epoch {ep + 1}/{epochs} step {step}/{total} loss {loss.item():.4f}",
                      flush=True)
        encoder.eval()
        head.eval()
        vl = 0.0
        if len(vp):
            with torch.no_grad():
                order = np.argsort([len(p) for p in vp])
                tot = 0.0
                for s in range(0, len(vp), 64):
                    ids = order[s:s + 64]
                    lg = _forward(tokenizer, encoder, head, [vp[i] for i in ids], max_len)
                    tot += loss_fn(lg, vy[ids]).item() * len(ids)
                vl = tot / len(vp)
        print(f"[finetune] epoch {ep + 1} val_loss {vl:.4f}", flush=True)
        if vl < best_val:
            best_val = vl
            best_state = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(head.state_dict()))
    if best_state is not None:
        encoder.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    encoder.eval()
    head.eval()
    return FineTunedRouter(tokenizer, encoder, head, max_len)
