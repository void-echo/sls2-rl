"""Reusable pinned uploads and bucketed CUDA graphs for rollout inference."""
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from .encode import field_spec
from .ppo import TORCH_DT


@dataclass
class GraphBatch:
    obs: dict
    graph: object
    logits: torch.Tensor
    value: torch.Tensor


class FastPolicy:
    def __init__(self, model, caps, n_envs, device, amp):
        self.model, self.caps, self.n_envs, self.device, self.amp = model, caps, n_envs, device, amp
        self.host = {k: torch.empty(n_envs * int(np.prod(shape)), dtype=TORCH_DT[np.dtype(dt).type],
                                   pin_memory=True)
                     for k, (shape, dt) in field_spec(caps).items()}
        self.cache = OrderedDict()
        self.captures = 0
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=device)

    def _shape(self, batch):
        b, t = batch["tok_type"].shape
        c = batch["cand_key"].shape[1]
        if not self.amp:
            return b, t, c
        return (min((b + 31) // 32 * 32, self.n_envs),
                min((t + 31) // 32 * 32, self.caps.max_tok),
                min((c + 15) // 16 * 16, self.caps.max_cand))

    @torch.no_grad()
    def _capture(self, shape):
        b, t, c = shape
        obs = {}
        for k, (tail, dt) in field_spec(self.caps).items():
            tail = ((t,) + tail[1:] if k.startswith("tok_") else
                    (c,) + tail[1:] if k.startswith("cand_") else tail)
            obs[k] = torch.zeros((b,) + tail, dtype=TORCH_DT[np.dtype(dt).type], device=self.device)
        obs["n_tok"].fill_(1)
        obs["n_cand"].fill_(1)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream), torch.autocast("cuda", dtype=torch.bfloat16,
                                                          enabled=self.amp, cache_enabled=False):
            for _ in range(3):
                self.model(obs)
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, stream=self.stream,
                              capture_error_mode="thread_local"), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.amp, cache_enabled=False):
            logits, value = self.model(obs)
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        return GraphBatch(obs, graph, logits, value)

    def prepare(self, batch):
        shape = self._shape(batch)
        if shape not in self.cache:
            if len(self.cache) == 64:
                self.cache.popitem(last=False)
            self.cache[shape] = self._capture(shape)
            self.captures += 1
        self.cache.move_to_end(shape)
        self.current = self.cache[shape]
        self.b, self.c = len(batch["n_tok"]), batch["cand_key"].shape[1]
        for k, dst in self.current.obs.items():
            host = self.host[k][:dst.numel()].view(dst.shape)
            arr = host.numpy()
            arr.fill(1 if k in ("n_tok", "n_cand") else 0)
            src = batch[k]
            arr[tuple(slice(0, n) for n in src.shape)] = src
            dst.copy_(host, non_blocking=True)
        return {k: v[:self.b, :batch[k].shape[1]] if v.ndim >= 2 else v[:self.b]
                for k, v in self.current.obs.items()}

    def __call__(self, obs):
        self.current.graph.replay()
        return self.current.logits[:self.b, :self.c], self.current.value[:self.b]
