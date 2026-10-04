import copy
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16,
          "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}


def _load_model(path, dtype, device):
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    m = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, **{key: dtype})
    m = m.to(device).eval()
    got = next(m.parameters()).dtype
    if got != dtype:
        raise RuntimeError(f"model loaded as {got}, expected {dtype}")
    return m


def _expand_cache(cache, n):
    if n == 1:
        return copy.deepcopy(cache)
    if hasattr(cache, "batch_repeat_interleave"):
        c = copy.deepcopy(cache)
        c.batch_repeat_interleave(n)
        return c
    return tuple(tuple(t.repeat_interleave(n, dim=0) for t in layer) for layer in cache)


class Scorer:
    def __init__(self, model_path, device="cpu", dtype="float32", batch_size=8,
                 max_len=4096, threads=None, prefix_cache=True):
        if threads:
            torch.set_num_threads(int(threads))
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = _load_model(model_path, self.dtype, self.device)
        self.batch_size = batch_size
        self.max_len = max_len
        self.prefix_cache = prefix_cache
        self.name = str(model_path)
        self.n_forward = 0

    def chat_prefix(self, system, user, assistant_prefix=""):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if getattr(self.tok, "chat_template", None):
            # enable_thinking=False: Qwen3 se mo <think> neu khong tat; template khac bo qua bien nay
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            p = f"{system}\n\n{user}\n\n"
        return p + assistant_prefix

    def _ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _split(self, prefix, conts):
        p = self._ids(prefix)
        full = [self._ids(prefix + c) for c in conts]
        k = len(p)
        for f in full:
            j = 0
            lim = min(len(p), len(f))
            while j < lim and p[j] == f[j]:
                j += 1
            k = min(k, j)
        k = max(1, k)
        return full[0][:k] if full else p[:k], [f[k:] for f in full]

    @torch.inference_mode()
    def score(self, prefix, conts):
        if not conts:
            return []
        head, tails = self._split(prefix, conts)
        if len(head) + max(len(t) for t in tails) > self.max_len:
            raise ValueError(f"sequence too long: {len(head)}+{max(len(t) for t in tails)} > {self.max_len}")
        if self.prefix_cache:
            try:
                return self._score_cached(head, tails)
            except Exception:
                self.prefix_cache = False
        return self._score_full(head, tails)

    def _score_cached(self, head, tails):
        ids = torch.tensor([head], device=self.device)
        out = self.model(input_ids=ids, use_cache=True)
        self.n_forward += 1
        cache = out.past_key_values
        last = torch.log_softmax(out.logits[0, -1].float(), -1)
        res = [0.0] * len(tails)
        order = [i for i in range(len(tails)) if tails[i]]
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            b = len(idx)
            L = max(len(tails[i]) for i in idx)
            pad = self.tok.pad_token_id
            x = torch.full((b, L), pad, dtype=torch.long, device=self.device)
            m = torch.zeros((b, L), dtype=torch.long, device=self.device)
            for r, i in enumerate(idx):
                t = tails[i]
                x[r, :len(t)] = torch.tensor(t, device=self.device)
                m[r, :len(t)] = 1
            att = torch.cat([torch.ones((b, len(head)), dtype=torch.long, device=self.device), m], 1)
            pos = torch.arange(len(head), len(head) + L, device=self.device).unsqueeze(0).expand(b, L)
            o = self.model(input_ids=x, attention_mask=att, past_key_values=_expand_cache(cache, b),
                           position_ids=pos, use_cache=True)
            self.n_forward += 1
            lp = torch.log_softmax(o.logits.float(), -1)
            for r, i in enumerate(idx):
                t = tails[i]
                total = last[t[0]].item()
                if len(t) > 1:
                    tgt = torch.tensor(t[1:], device=self.device)
                    total += lp[r, torch.arange(len(t) - 1, device=self.device), tgt].sum().item()
                res[i] = total
        return res

    def _score_full(self, head, tails):
        res = [0.0] * len(tails)
        seqs = [head + t for t in tails]
        for s in range(0, len(seqs), self.batch_size):
            chunk = list(range(s, min(s + self.batch_size, len(seqs))))
            L = max(len(seqs[i]) for i in chunk)
            b = len(chunk)
            x = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long, device=self.device)
            m = torch.zeros((b, L), dtype=torch.long, device=self.device)
            for r, i in enumerate(chunk):
                x[r, :len(seqs[i])] = torch.tensor(seqs[i], device=self.device)
                m[r, :len(seqs[i])] = 1
            o = self.model(input_ids=x, attention_mask=m)
            self.n_forward += 1
            lp = torch.log_softmax(o.logits.float(), -1)
            for r, i in enumerate(chunk):
                t = tails[i]
                if not t:
                    continue
                pos = torch.arange(len(head) - 1, len(head) - 1 + len(t), device=self.device)
                tgt = torch.tensor(t, device=self.device)
                res[i] = lp[r, pos, tgt].sum().item()
        return res

    @torch.inference_mode()
    def generate(self, prefix, max_new_tokens=192):
        ids = torch.tensor([self._ids(prefix)], device=self.device)
        t0 = time.time()
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        self.n_forward += 1
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return text, time.time() - t0
