"""PPO training entry point.

    python -m rl.train --name ironclad_a0 --engine real --n_workers 32 --envs_per_worker 8
    python -m rl.train --preset tiny --name smoke            # CPU, FakeEngine

Rollouts are collected asynchronously per env (each env steps as soon as its engine answers; inference is
batched over whichever envs are ready), but the update itself is synchronous: once `rollout_steps`
actions are taken the learner stops issuing actions, waits for their results, bootstraps, and runs PPO.
Engines that are booting or resetting keep working during the update, so slow resets never stall a step.
"""
import contextlib
import copy
import csv
import glob
import gzip
import io
import json
import os
import signal
import statistics
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .actors import ActorPool, TASTE_CARD_KINDS
from .config import Config, parse_args, save_config
from .diskfull import append_all, quiet_stdio, remove_quietly, write_atomic
from .encode import TT
from .force import force_choice
from .model import Policy, n_params
from .ppo import Rollout, ValueNorm, ppo_update, to_torch
from .protocol import N_DECISION_IDS, decision_id, decision_name


TASTE_DECISIONS = {decision_id(name) for name in TASTE_CARD_KINDS}


class CSVLog:
    """Appends rows to a CSV file. Rows that cannot be written (the disk is full) stay in memory and go out before the
    next row; a row is appended whole or not at all, so the file never holds a broken line."""
    MAX_PENDING = 200_000       # rows kept while the disk is full (about a thousand updates of episodes)

    def __init__(self, path, expand=False):
        self.path, self.fields = str(path), None
        self.warned = False
        self.expand = expand
        self.pending, self.stalled, self.dropped = [], False, 0

    def write(self, row):
        self.write_many([row])

    def write_many(self, rows):
        self.pending.extend(rows)
        if not self.pending:
            return
        try:
            self._flush()
        except OSError as e:
            if not self.stalled:
                print(f"CSVLog {self.path}: not written ({e}); rows stay in memory until the disk takes them", flush=True)
                self.stalled = True
            over = len(self.pending) - self.MAX_PENDING
            if over > 0:
                del self.pending[:over]
                self.dropped += over
            return
        if self.stalled:
            print(f"CSVLog {self.path}: writing again" + (f" ({self.dropped} rows were dropped)" if self.dropped else ""), flush=True)
            self.stalled, self.dropped = False, 0

    def _flush(self):
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        if self.fields is None:
            if exists:
                with open(self.path, newline="") as f:
                    self.fields = next(csv.reader(f))
            else:
                self.fields = list(self.pending[0])
        extra = set().union(*(row.keys() for row in self.pending)) - set(self.fields)
        if extra and self.expand:
            fields = self.fields + sorted(extra)
            if exists:      # rewrite the file with the wider header; the old file stays until the new one is complete
                with open(self.path, newline="") as f:
                    rows = list(csv.DictReader(f))
                write_atomic(self.path, self._text(fields, rows, header=True))
            self.fields = fields
        elif extra and not self.warned:
            print(f"CSVLog {self.path}: dropping fields {', '.join(sorted(extra))}", flush=True)
            self.warned = True
        append_all(self.path, self._text(self.fields, self.pending, header=not exists))
        self.pending = []

    @staticmethod
    def _text(fields, rows, header):
        out = io.StringIO(newline="")
        w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        if header:
            w.writeheader()
        w.writerows(rows)
        return out.getvalue().encode()

    def close(self):
        self.write_many([])


def amp_context(cfg, device):
    if cfg.amp and device.type == "cuda":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext


@torch.no_grad()
def act_batch(model, obs, amp_ctx, greedy=False, n_actions=None, return_logits=False, logit_bias=None):
    with amp_ctx():
        logits, v = model(obs)
    logits = logits.float()
    all_logits = logits
    if n_actions is not None:
        logits = logits[:n_actions]
    if greedy:
        a = logits.argmax(-1)
    else:
        u = torch.rand_like(logits).clamp_(1e-10, 1 - 1e-7)
        if logit_bias is None:
            a = (logits - torch.log(-torch.log(u))).argmax(-1)
        else:
            noise = torch.log(-torch.log(u))
            scores = logits - noise
            rows, bias = logit_bias
            scores[rows] = logits[rows] + bias - noise[rows]
            a = scores.argmax(-1)
    logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
    result = a, logp, v.float()
    return (*result, all_logits) if return_logits else result


class AscCurriculum:
    def __init__(self, cfg, frontier):
        self.cfg, self.frontier = cfg, frontier
        self.windows = {}

    def record(self, ep):
        if ep.get("midrun") or ep.get("force") or ep.get("taste"):
            return False
        level = ep["asc"]
        window = self.windows.setdefault(level, deque(maxlen=self.cfg.asc_window))
        if ep.get("aborted"):
            return False
        window.append(bool(ep["victory"]))
        if (level == self.frontier < self.cfg.asc_max and len(window) == self.cfg.asc_window
                and sum(window) / len(window) >= self.cfg.asc_threshold):
            self.frontier += 1
            return True
        return False

    def stats(self):
        return {"asc_frontier": self.frontier,
                **{f"w_win_a{k}": sum(w) / len(w) if w else "" for k, w in sorted(self.windows.items())}}


class Collector:
    """Drives the ActorPool: batches ready envs for inference, records transitions and episode stats."""

    def __init__(self, cfg, pool, model, vnorm, device, amp_ctx, err_log=True):
        self.cfg, self.pool, self.model, self.vnorm = cfg, pool, model, vnorm
        self.device, self.amp_ctx = device, amp_ctx
        self.fast = None
        if cfg.collect_fast and device.type == "cuda":
            from .collect_fast import FastPolicy
            self.fast = FastPolicy(model, cfg.caps, pool.n_envs, device, cfg.amp)
        self.ready = set()
        self.pending = {}
        self.episodes = []
        self.errors = Counter()
        self.err_samples = []
        self.trunc = Counter()
        self.groups, self.value_requests, self.result_idx = {}, {}, {}
        self.branches, self.branch_offers, self.branch_first = {}, {}, {}
        self.explore = {}
        if cfg.explore_eps > 0:
            self.explore_rng = np.random.default_rng(cfg.seed)
        self.force_on = cfg.force_enabled
        if self.force_on:
            self.force_rng = np.random.default_rng(cfg.seed)
            self.force, self.n_forced = {}, {}
        self.taste_on = cfg.taste_enabled
        if self.taste_on:
            self.taste_rng = np.random.default_rng([cfg.seed, 1])
            self.taste = {}
        if cfg.asc_curriculum:
            self.curriculum = AscCurriculum(cfg, pool.asc_frontier)
        self.err_file = open(os.path.join(cfg.resolved_run_dir(), "engine_errors.log"), "a") if err_log else None

    def handle(self, events, buf):
        for ev in events:
            kind = ev[0]
            if kind == "res":
                _, e, r, done, tr, has_obs = ev
                i = self.pending.pop(e, None)
                if i is not None and buf is not None:
                    buf.set_result(i, r, done, tr)
                    if self.cfg.group_k > 1 or self.cfg.branch_decisions:
                        self.result_idx[e] = i
                if has_obs:
                    self.ready.add(e)
            elif kind == "obs":
                self.ready.add(ev[1])
            elif kind == "ep_start":
                if self.cfg.explore_eps > 0:
                    self.explore[ev[1]] = self.explore_rng.random() < self.cfg.explore_frac
                if self.force_on:
                    self.force[ev[1]] = self.force_rng.random() < self.cfg.force_frac
                    self.n_forced[ev[1]] = 0
                if self.taste_on and self.taste_rng.random() < self.cfg.taste_frac:
                    direction = self.taste_rng.standard_normal(self.model.table.shape[1], dtype=np.float32)
                    self.taste[ev[1]] = direction / np.linalg.norm(direction)
            elif kind == "group_start":
                self.groups[ev[1]] = dict(trials=[[]], wins=[], boots=[])
            elif kind == "group_end":
                _, e, win, repeat, terminal = ev
                self.groups[e]["wins"].append(win)
                i = self.result_idx[e]
                buf.cut[i] = repeat
                if terminal:
                    self.finish_trial(e, 0.0, repeat, buf)
                else:
                    self.value_requests[e] = (i, repeat)
            elif kind == "group_cancel":
                e = ev[1]
                g = self.groups.pop(e, None)
                if g is not None and e in self.branches:
                    self.branches[e]["trials"][-1].extend(g["trials"][-1])
                self.value_requests.pop(e, None)
            elif kind == "branch_offer":
                _, e, options, keys = ev
                self.branch_offers[e] = dict(options=options, keys=keys)
            elif kind == "branch_trial":
                _, e, action, main = ev
                self.branch_first[e] = (action, main)
            elif kind == "branch_end":
                _, e, main, terminal = ev
                if terminal:
                    self.finish_branch(e, 0.0, main, buf)
                else:
                    self.value_requests[e] = ("branch", main)
            elif kind == "branch_cancel":
                e = ev[1]
                if len(ev) > 2 and ev[2]:
                    self.branches[e]["cancelled"] = True
                    if self.branches[e]["trials"][-1]:
                        self.finish_branch(e, 0.0, False, buf)
                else:
                    self.branches.pop(e, None)
                self.branch_offers.pop(e, None)
                self.branch_first.pop(e, None)
                self.value_requests.pop(e, None)
            elif kind == "ep":
                if self.cfg.explore_eps > 0:
                    ev[1]["explore"] = int(self.explore.pop(ev[1]["env"], False))
                if self.force_on:
                    ev[1]["force"] = int(self.force.pop(ev[1]["env"], False))
                    ev[1]["n_forced"] = self.n_forced.pop(ev[1]["env"], 0)
                if self.taste_on:
                    ev[1]["taste"] = int(self.taste.pop(ev[1]["env"], None) is not None)
                self.episodes.append(ev[1])
                if self.cfg.asc_curriculum and self.curriculum.record(ev[1]):
                    self.pool.set_asc_frontier(self.curriculum.frontier)
            elif kind == "err":
                self.errors["engine_errors"] += 1
                if len(self.err_samples) < 20:
                    self.err_samples.append(f"env{ev[1]}: {ev[2]}")
                if self.err_file is not None:
                    try:
                        self.err_file.write(f"{time.strftime('%H:%M:%S')} env{ev[1]}: {ev[2]}\n")
                        self.err_file.flush()
                    except OSError:     # disk full: the count and the samples in the progress line remain
                        pass
            elif kind == "trunc":
                self.trunc.update(ev[1])

    def finish_trial(self, e, value, repeat, buf):
        g = self.groups[e]
        buf.bootstrap[g["trials"][-1][-1]] = value
        g["boots"].append(float(value))
        if repeat:
            g["trials"].append([])
        else:
            if e in self.branches:
                self.branches[e]["trials"][-1].extend(g["trials"][-1])
            buf.groups.append(self.groups.pop(e))

    def finish_branch(self, e, value, main, buf):
        b = self.branches[e]
        ix = b["trials"][-1]
        if main and e in self.groups:
            ix.extend(self.groups[e]["trials"][-1])
        b["boots"].append(float(value))
        if main:
            self.branches.pop(e)
            if not b.get("cancelled"):
                buf.branches.append(b)
        else:
            buf.cut[ix[-1]], buf.bootstrap[ix[-1]] = True, value
            buf.branch_extra += len(ix)
            # Keep the main trajectory linked across the side trials, each of which ends at a cut.
            buf.env_idx[e] = ix + buf.env_idx[e][:-len(ix)]
            b["trials"].append([])

    def values(self, envs):
        out = []
        for s in range(0, len(envs), 512):
            obs_np = self.pool.shm.gather(envs[s:s + 512])
            if self.force_on:
                obs_np.pop("cand_force")
            if self.taste_on:
                obs_np.pop("cand_card_rows")
            obs = to_torch(obs_np, self.device)
            _, _, v = act_batch(self.model, obs, self.amp_ctx)
            out.append(v.cpu().numpy())
        v = np.concatenate(out) if out else np.zeros(0, np.float32)
        return self.vnorm.denormalize(v) if self.cfg.value_norm else v

    def collect(self, buf):
        # one autocast context for the whole rollout: weights are cast to bf16 once, not on every call
        with self.amp_ctx():
            if self.cfg.group_k > 1 or self.cfg.branch_decisions:
                return self._collect_groups(buf)
            return self._collect(buf)

    def _collect_groups(self, buf):
        cfg, pool = self.cfg, self.pool
        buf.reset()
        self.model.eval()
        budget = cfg.rollout_steps
        replay_limit = budget + max(budget, pool.n_envs * 256)
        if buf.cap < replay_limit:
            buf.grow(replay_limit)
        min_batch = max(1, int(cfg.min_infer_batch * pool.n_envs))
        max_wait = cfg.max_infer_wait_ms / 1000
        t0 = time.time()
        sent, n_calls, infer_t = 0, 0, 0.0
        t_parts = np.zeros(3)
        dec_counts = np.zeros(N_DECISION_IDS, np.int64)
        wait_start = None
        while True:
            self.handle(pool.poll(0), buf)
            active = {e for e, g in self.groups.items() if g["trials"][0]}
            if cfg.branch_decisions:
                active.update(self.branches)
            envs = sorted(self.ready if sent < budget else self.ready & active)
            if sent < budget:
                envs = envs[:budget - sent]
            normal_envs = [e for e in envs if e not in self.branch_first] if cfg.branch_decisions else envs
            envs = normal_envs + [e for e in envs if e in self.branch_first]
            value_envs = sorted(self.value_requests)
            if not envs and not value_envs:
                if sent >= budget and not self.pending and not active:
                    break
                self.handle(pool.poll(0.05), buf)
                continue
            now = time.time()
            wait_start = wait_start or now
            if not value_envs and len(envs) < min_batch and now - wait_start < max_wait:
                self.handle(pool.poll(max_wait - (now - wait_start)), buf)
                continue
            wait_start = None
            if buf.n + len(envs) > buf.cap:
                buf.grow(buf.cap + max(budget, len(envs)))
            self.ready.difference_update(envs)
            ti = time.time()
            obs_np = pool.shm.gather(envs + value_envs)
            t1 = time.time()
            obs = to_torch(obs_np, self.device)
            result = act_batch(self.model, obs, self.amp_ctx, n_actions=len(normal_envs),
                               return_logits=bool(cfg.branch_decisions))
            a, logp, v = result[:3]
            a_np, v_np = a.cpu().numpy(), v.cpu().numpy()
            if cfg.value_norm:
                v_np = self.vnorm.denormalize(v_np)
            t2 = time.time()
            if envs:
                keep, actions, lps, starts, forced = [], [], [], [], []
                for row, e in enumerate(envs):
                    if cfg.branch_decisions and e in self.branch_offers:
                        b = self.branch_offers.pop(e)
                        action = int(a_np[row])
                        b.update(logits=result[3][row].cpu().numpy(), trials=[[]], boots=[],
                                 actions=[j for j in b["options"] if j != action] + [action])
                        self.branches[e] = b
                        starts.append((e, action))
                        continue
                    first = self.branch_first.pop(e, None) if cfg.branch_decisions else None
                    action = first[0] if first else int(a_np[row])
                    lp = float(torch.log_softmax(result[3][row], -1)[action]) if first else float(logp[row])
                    keep.append(row)
                    actions.append(action)
                    lps.append(lp)
                    forced.append(first is not None and not first[1])
                action_envs = [envs[j] for j in keep]
                action_obs = {k: x[keep] for k, x in obs.items()}
                idx = buf.add(action_envs, action_obs, actions, lps, v_np[keep])
                buf.forced[idx] = forced
                for e, i in zip(action_envs, idx):
                    self.pending[e] = i
                    if e in self.groups:
                        self.groups[e]["trials"][-1].append(int(i))
                    if e in self.branches and e not in self.groups:
                        self.branches[e]["trials"][-1].append(int(i))
                pool.act(action_envs, actions, allow_replay=sent + len(action_envs) < replay_limit - pool.n_envs)
                if starts:
                    pool.start_branches(starts)
                dec_counts += np.bincount(obs_np["decision"][keep].astype(np.int64),
                                          minlength=N_DECISION_IDS)
            for e, value in zip(value_envs, v_np[len(envs):]):
                kind, repeat = self.value_requests.pop(e)
                if kind == "branch":
                    self.finish_branch(e, value, repeat, buf)
                else:
                    self.finish_trial(e, value, repeat, buf)
            if value_envs:
                pool.value_done(value_envs)
            t3 = time.time()
            infer_t += t3 - ti
            t_parts += (t1 - ti, t2 - t1, t3 - t2)
            n_calls += 1
            sent = buf.n
        collect_t = time.time() - t0
        boot = np.zeros(pool.n_envs, np.float32)
        need = [e for e, ix in enumerate(buf.env_idx)
                if ix and not buf.done[ix[-1]] and not buf.trunc[ix[-1]]]
        if need:
            with torch.no_grad():
                obs = to_torch(pool.shm.gather(need), self.device)
                _, v = self.model(obs)
            v = v.float().cpu().numpy()
            boot[need] = self.vnorm.denormalize(v) if cfg.value_norm else v
        return boot, {"collect_time": collect_t, "env_sps": sent / collect_t, "infer_calls": n_calls,
                      "infer_batch": sent / max(n_calls, 1), "infer_time": infer_t,
                      "t_gather": t_parts[0], "t_model": t_parts[1], "t_store": t_parts[2],
                      "dec_counts": dec_counts}

    def explore_actions(self, envs, obs, actions, logp, logits, skip=()):
        """Replace some build actions in exploration episodes; returns the replaced rows."""
        combat = np.any(obs["tok_type"] == TT["COMBAT"], axis=1)
        replaced = []
        for j, e in enumerate(envs):
            if combat[j] or not self.explore.get(e, False) or j in skip:
                continue
            self.explore_build_decisions += 1
            if self.explore_rng.random() < self.cfg.explore_eps:
                kinds = obs["cand_type"][j, :int(obs["n_cand"][j])]
                kind = self.explore_rng.choice(np.unique(kinds))
                action = int(self.explore_rng.choice(np.flatnonzero(kinds == kind)))
                actions[j] = action
                logp[j] = torch.log_softmax(logits[j], -1)[action]
                self.explore_replaced += 1
                replaced.append(j)
        return replaced

    def force_actions(self, envs, matches, actions, logp, logits):
        replaced, locked = [], []
        for j, e in enumerate(envs):
            if not self.force.get(e, False) or not matches[j].any():
                continue
            if self.force_rng.random() >= self.cfg.force_prob:
                continue
            action = force_choice(matches[j], torch.softmax(logits[j], -1))
            locked.append(j)
            if action == int(actions[j]):
                continue
            actions[j] = action
            logp[j] = torch.log_softmax(logits[j], -1)[action]
            self.force_replaced += 1
            self.n_forced[e] += 1
            replaced.append(j)
        return replaced, locked

    @torch.no_grad()
    def taste_bias(self, envs, decisions, card_rows):
        selected = [j for j, e in enumerate(envs) if e in self.taste and decisions[j] in TASTE_DECISIONS]
        if not selected:
            return selected, None
        rows = torch.as_tensor(card_rows[selected], device=self.device)
        directions = torch.as_tensor(np.stack([self.taste[envs[j]] for j in selected]), device=self.device)
        table = self.model.table
        valid = rows[..., 0] >= 0
        rows = rows.clamp_min(0)
        embeds = table[rows[..., 0]].float() + table[rows[..., 1]].float()
        embeds = torch.nn.functional.normalize(embeds, dim=-1)
        bias = self.cfg.taste_beta * (embeds * directions[:, None, :]).sum(-1)
        return selected, bias.masked_fill(~valid, 0.0)

    @torch.no_grad()
    def record_taste(self, logits, bias):
        biased = logits + bias
        logp = logits.log_softmax(-1)
        biased_logp = biased.log_softmax(-1)
        kl = (biased_logp.exp() * (biased_logp - logp)).masked_fill(~logits.isfinite(), 0.0).sum(-1)
        self.taste_steps += len(logits)
        self.taste_kl += float(kl.clamp_min(0.0).sum())
        self.taste_changed += int((biased.argmax(-1) != logits.argmax(-1)).sum())

    def _collect(self, buf):
        cfg, pool = self.cfg, self.pool
        buf.reset()
        if cfg.explore_eps > 0:
            self.explore_replaced = self.explore_build_decisions = 0
        if self.force_on:
            self.force_replaced = 0
        if self.taste_on:
            self.taste_steps, self.taste_kl, self.taste_changed = 0, 0.0, 0
        self.model.eval()
        budget = buf.cap
        min_batch = max(1, int(cfg.min_infer_batch * pool.n_envs))
        max_wait = cfg.max_infer_wait_ms / 1000
        t0 = time.time()
        sent, n_calls, infer_t = 0, 0, 0.0
        t_parts = np.zeros(3)  # gather from shm, H2D + model + D2H, store + send
        dec_counts = np.zeros(N_DECISION_IDS, np.int64)
        wait_start = None
        while True:
            if sent >= budget:
                if not self.pending:
                    break
                self.handle(pool.poll(0.05), buf)
                continue
            self.handle(pool.poll(0), buf)
            if not self.ready:
                self.handle(pool.poll(0.05), buf)
                continue
            now = time.time()
            wait_start = wait_start or now
            if len(self.ready) < min_batch and now - wait_start < max_wait:
                self.handle(pool.poll(max_wait - (now - wait_start)), buf)
                continue
            wait_start = None
            envs = sorted(self.ready)[: budget - sent]
            self.ready.difference_update(envs)
            ti = time.time()
            obs_np = pool.shm.gather(envs)
            if self.force_on:
                force_matches = obs_np.pop("cand_force")
            if self.taste_on:
                card_rows = obs_np.pop("cand_card_rows")
            t1 = time.time()
            obs = self.fast.prepare(obs_np) if self.fast is not None else to_torch(obs_np, self.device)
            infer_model = self.fast if self.fast is not None else self.model
            if self.force_on:
                a, logp, v, logits = act_batch(infer_model, obs, self.amp_ctx, return_logits=True)
                replaced, locked = self.force_actions(envs, force_matches, a, logp, logits)
                if cfg.explore_eps > 0:
                    replaced.extend(self.explore_actions(envs, obs_np, a, logp, logits, skip=locked))
            elif self.taste_on:
                replaced, bias = self.taste_bias(envs, obs_np["decision"], card_rows)
                a, logp, v, logits = act_batch(infer_model, obs, self.amp_ctx, return_logits=True,
                                             logit_bias=(replaced, bias) if replaced else None)
                if replaced:
                    self.record_taste(logits[replaced], bias)
                if cfg.explore_eps > 0:
                    replaced.extend(self.explore_actions(envs, obs_np, a, logp, logits, skip=replaced))
            elif cfg.explore_eps > 0:
                a, logp, v, logits = act_batch(infer_model, obs, self.amp_ctx, return_logits=True)
                replaced = self.explore_actions(envs, obs_np, a, logp, logits)
            else:
                a, logp, v = act_batch(infer_model, obs, self.amp_ctx)
            a_np = a.cpu().numpy()
            t2 = time.time()
            pool.act(envs, a_np)
            v_np = v.cpu().numpy()
            if cfg.value_norm:
                v_np = self.vnorm.denormalize(v_np)
            # two devices (infer_device): upload the observations to the buffer's device from the host; a GPU-to-GPU copy
            # waits on the device that is busy with the update (5 s per 32768 steps in the smoke test of 11-11)
            store = obs if torch.device(buf.device) == self.device else to_torch(obs_np, buf.device)
            idx = buf.add(envs, store, a_np, logp.cpu().numpy(), v_np)
            if (cfg.explore_eps > 0 or self.force_on or self.taste_on) and replaced:
                buf.explored[np.asarray(idx)[replaced]] = True
            t3 = time.time()
            infer_t += t3 - ti
            t_parts += (t1 - ti, t2 - t1, t3 - t2)
            for e, i in zip(envs, idx):
                self.pending[e] = i
            dec_counts += np.bincount(obs_np["decision"].astype(np.int64), minlength=N_DECISION_IDS)
            n_calls += 1
            sent += len(envs)
        collect_t = time.time() - t0
        # bootstrap values for envs whose last transition did not end the episode
        boot = np.zeros(pool.n_envs, np.float32)
        need = []
        for e, ix in enumerate(buf.env_idx):
            if ix and not buf.done[ix[-1]] and not buf.trunc[ix[-1]]:
                if e in self.ready:
                    need.append(e)
                else:
                    buf.trunc[ix[-1]] = True  # should not happen; be safe
        if need:
            boot[need] = self.values(need)
        stats = {}
        if cfg.explore_eps > 0:
            stats.update(explore_replaced=self.explore_replaced, explore_build_decisions=self.explore_build_decisions)
        if self.force_on:
            stats["force_replaced"] = self.force_replaced
        if self.taste_on:
            stats.update(taste_steps=self.taste_steps, taste_kl=self.taste_kl / max(self.taste_steps, 1),
                         taste_changed=self.taste_changed / max(self.taste_steps, 1))
        return boot, {"collect_time": collect_t, "env_sps": sent / collect_t, "infer_calls": n_calls,
                      "infer_batch": sent / max(n_calls, 1), "infer_time": infer_t,
                      "t_gather": t_parts[0], "t_model": t_parts[1], "t_store": t_parts[2],
                      "dec_counts": dec_counts, **stats}


def episode_stats(eps):
    done = [e for e in eps if not e.get("aborted")]
    out = {"episodes": len(done), "aborted": len(eps) - len(done)}
    if done:
        fl = [e["floor"] for e in done]
        out.update(win_rate=float(np.mean([bool(e["victory"]) for e in done])), floor_mean=float(np.mean(fl)),
                   floor_median=float(statistics.median(fl)), floor_max=max(fl),
                   return_mean=float(np.mean([e["ret"] for e in done])),
                   ep_len_mean=float(np.mean([e["steps"] for e in done])))
    else:
        out.update(win_rate="", floor_mean="", floor_median="", floor_max="", return_mean="", ep_len_mean="")
    return out


def save_ckpt(path_dir, model, opt, vnorm, cfg, state, keep, rng=None):
    """Write ckpt_<update>.pt and point latest.pt at it; returns the path. When the disk does not take it (full, quota),
    returns False with the earlier checkpoints and latest.pt as they were: the caller keeps training and tries again."""
    p = os.path.join(path_dir, f"ckpt_{state['update']:06d}.pt")
    random_state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        random_state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    if rng is not None:
        random_state["numpy_rng_state"] = rng.bit_generator.state
    data = io.BytesIO()         # serialized in memory first: a failed write is then a plain OSError, not a torch error
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "vnorm": vnorm.state_dict(), **random_state,
                "cfg": cfg.to_dict(), **state}, data)
    latest = os.path.join(path_dir, "latest.pt")
    try:
        os.makedirs(path_dir, exist_ok=True)
        write_atomic(p, data.getbuffer())
        os.symlink(os.path.basename(p), latest + ".new")    # swapped in by rename: latest.pt is never missing
        os.replace(latest + ".new", latest)
    except OSError as e:
        remove_quietly(latest + ".new")
        print(f"checkpoint {p} not written ({e}); training goes on and tries again after the next update", flush=True)
        return False
    for f in sorted(glob.glob(os.path.join(path_dir, "ckpt_*.pt")))[:-keep]:
        remove_quietly(f)
    return p


def load_policy(path, device="cpu"):
    """Load a checkpoint for evaluation: returns (model, vnorm, cfg)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ck["cfg"])
    model = Policy(cfg.model).to(device)
    model.load_state_dict(ck["model"])
    vn = ValueNorm()
    vn.load_state_dict(ck["vnorm"])
    return model, vn, cfg, ck


def _decays(name, p):
    """Weight decay applies to the matrices of the trunk, candidate blocks and heads; the hash table, the token-type
    embeddings, LayerNorm parameters and biases do not decay (rare keys would shrink while unused)."""
    return p.ndim >= 2 and name != "table" and not name.endswith("type_emb.weight")


def kl_lr_step(lr, kl_ema, kl, cfg):
    """One step of the KL-tracking learning rate: update the EMA of the per-update approx KL, then scale lr by
    (target / ema) ** gain, at most lr_kl_step per update, within [lr_kl_min, lr_kl_max]. Returns (lr, ema)."""
    ema = kl if kl_ema is None else cfg.lr_kl_ema * kl_ema + (1 - cfg.lr_kl_ema) * kl
    factor = (cfg.lr_kl_target / max(ema, 1e-12)) ** cfg.lr_kl_gain
    factor = min(max(factor, 1 / cfg.lr_kl_step), cfg.lr_kl_step)
    return min(max(lr * factor, cfg.lr_kl_min), cfg.lr_kl_max), ema


def make_optimizer(model, cfg):
    """AdamW with optional growth or weight-decay groups."""
    if cfg.grow_params and cfg.grow_lr_mult != 1:
        if cfg.weight_decay > 0:
            raise ValueError("grow parameter groups cannot be combined with weight_decay > 0")
        prefixes = tuple(p.strip() for p in cfg.grow_params.split(",") if p.strip())
        params = list(model.named_parameters())
        groups = [{"params": [p for n, p in params if not n.startswith(prefixes)], "lr_mult": 1.0},
                  {"params": [p for n, p in params if n.startswith(prefixes)], "lr_mult": cfg.grow_lr_mult,
                   "lr": cfg.lr * cfg.grow_lr_mult}]
        return torch.optim.AdamW(groups, lr=cfg.lr, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    if cfg.weight_decay <= 0:
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    params = list(model.named_parameters())
    groups = [{"params": [p for n, p in params if _decays(n, p)], "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in params if not _decays(n, p)], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, eps=cfg.adam_eps)


def load_opt_state(opt, model, sd, cfg, saved_cfg=None):
    """Restore Adam moments using the saving run's parameter groups."""
    named = list(model.named_parameters())
    saved_cfg = saved_cfg or {}
    if len(sd["param_groups"]) == 1:
        saved_names = [[n for n, _ in named]]
    elif saved_cfg.get("grow_params") and saved_cfg.get("grow_lr_mult", 1.0) != 1:
        prefixes = tuple(p.strip() for p in saved_cfg["grow_params"].split(",") if p.strip())
        saved_names = [[n for n, _ in named if not n.startswith(prefixes)],
                       [n for n, _ in named if n.startswith(prefixes)]]
    else:
        saved_names = [[n for n, p in named if _decays(n, p)],
                       [n for n, p in named if not _decays(n, p)]]
    names = {id(p): n for n, p in named}
    current_names = [[names[id(p)] for p in g["params"]] for g in opt.param_groups]
    if saved_names == current_names:
        opt.load_state_dict(sd)
    else:
        params = dict(named)
        opt.state.clear()
        for group, group_names in zip(sd["param_groups"], saved_names):
            for i, name in zip(group["params"], group_names):
                if i in sd["state"]:
                    p = params[name]
                    opt.state[p] = {k: v.to(p.device) if torch.is_tensor(v) and k != "step" else v
                                    for k, v in sd["state"][i].items()}
    for g, wd in zip(opt.param_groups, [max(cfg.weight_decay, 0.0), 0.0]):
        g["weight_decay"] = wd
    if cfg.grow_params and cfg.grow_lr_mult != 1:
        for g, mult in zip(opt.param_groups, [1.0, cfg.grow_lr_mult]):
            g["lr_mult"] = mult


def policy_advantages(buf, boot, adv, cfg):
    """Advantages for the policy loss when lam_pi is set: a second GAE pass with the policy's lambda. The value targets
    (ret) always come from lam / lam_build."""
    if cfg.lam_pi is None:
        return adv, {}
    lam_b = cfg.lam_pi if cfg.lam_pi_build is None else cfg.lam_pi_build
    if cfg.lam_build is None and cfg.lam_pi_build is None:
        a = buf.gae(boot, cfg.gamma, cfg.lam_pi)[0]
    else:
        a = buf.gae_build(boot, cfg.gamma, cfg.lam_pi, lam_b)[0]
    return a, {"adv_std_pi": float(a.std())}


class ActionLog:
    """<run_dir>/actions/u<first update of the block of 1000>.jsonl.gz: one line per finished full game with its action
    indices. Aborted games and mid-run starts are left out (they cannot be replayed from the seed)."""

    def __init__(self, run_dir, cfg):
        self.dir, self.cfg = os.path.join(run_dir, "actions"), cfg
        os.makedirs(self.dir, exist_ok=True)

    def write(self, update, eps):
        rows = [e for e in eps if e.get("actions") and not e.get("aborted") and not e.get("midrun")]
        if not rows:
            return
        text = "".join(json.dumps({"u": update, "seed": e["seed"], "char": e["char"], "asc": e.get("asc", self.cfg.ascension),
                                   "victory": bool(e["victory"]), "floor": e["floor"], "a": e["actions"]},
                                  separators=(",", ":")) + "\n" for e in rows)
        try:    # one gzip member per update, appended whole or not at all; an update that does not fit is left out
            append_all(os.path.join(self.dir, f"u{(update - 1) // 1000 * 1000:07d}.jsonl.gz"), gzip.compress(text.encode()))
        except OSError as e:
            print(f"action log of update {update} not written ({e})", flush=True)


def log_episodes(ep_log, action_log, state, eps):
    if action_log is not None:
        action_log.write(state["update"], eps)
    ep_log.write_many([{"update": state["update"], "env_steps": state["env_steps"], **{k: v for k, v in e.items() if k != "actions"}}
                       for e in eps])


class _AsyncUpdater:
    def __init__(self, model, opt, vnorm, cfg, rng, amp_ctx, device):
        self.model, self.opt, self.vnorm = model, opt, vnorm
        self.cfg, self.rng, self.amp_ctx, self.device = cfg, rng, amp_ctx, device
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.future = None
        self.refreshed = True

    def start(self, buf, adv, ret, mask, distill):
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.refreshed = False
        self.future = self.executor.submit(self._update, buf, adv, ret, mask, distill)

    def _update(self, buf, adv, ret, mask, distill):
        ctx = torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()
        with ctx:
            self.model.train()
            return ppo_update(self.model, self.opt, buf, adv, ret, mask, self.vnorm,
                              self.cfg, self.rng, self.amp_ctx, **distill)

    def refresh(self, model, vnorm):
        if not self.refreshed and self.future.done():
            self.future.result()
            model.load_state_dict(self.model.state_dict())
            vnorm.load_state_dict(self.vnorm.state_dict())
            self.refreshed = True

    def wait(self):
        if self.future.done():
            return self.future.result(), 0.0
        t0 = time.time()
        stats = self.future.result()
        return stats, time.time() - t0


class _StopCollection(Exception):
    pass


class _AsyncCollector(Collector):
    def __init__(self, cfg, pool, model, vnorm, device, amp_ctx, updater, stop):
        self.updater, self.stop, self.infer_amp_ctx = updater, stop, amp_ctx
        super().__init__(cfg, pool, copy.deepcopy(model).to(device).eval().requires_grad_(False),
                         copy.deepcopy(vnorm), device, self.inference_context)

    @contextlib.contextmanager
    def inference_context(self):
        if self.stop["flag"]:
            raise _StopCollection
        self.updater.refresh(self.model, self.vnorm)
        with self.infer_amp_ctx():
            yield

    def collect(self, buf):
        return self._collect(buf)

    def values(self, envs):
        out = []
        for s in range(0, len(envs), 512):
            obs = to_torch(self.pool.shm.gather(envs[s:s + 512]), self.device)
            _, _, v = act_batch(self.model, obs, self.amp_ctx)
            v = v.cpu().numpy()
            out.append(self.vnorm.denormalize(v) if self.cfg.value_norm else v)
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def _split_buffers(cfg, device, infer_device):
    def canonical(dev):
        if dev.type == "cuda":
            return torch.device("cuda", torch.cuda.current_device() if dev.index is None else dev.index)
        return torch.device(dev.type)
    return cfg.async_update and cfg.async_split_buffers and canonical(device) != canonical(infer_device)


def _train_async(cfg, device, model, opt, vnorm, state, teacher, rng, amp_ctx, pool, log, ep_log, tb, ckpt_dir, action_log=None):
    infer_device = torch.device(cfg.infer_device) if cfg.infer_device else device
    split = _split_buffers(cfg, device, infer_device)
    buffers = [Rollout(cfg.caps, cfg.rollout_steps, cfg.n_envs, d)
               for d in ((infer_device, device) if split else (device, device))]
    for buf in buffers:
        buf.track_combat = cfg.lam_build is not None
    updater = _AsyncUpdater(model, opt, vnorm, cfg, rng, amp_ctx, device)
    stop = {"flag": False}
    handlers = {sig: signal.signal(sig, lambda *a: stop.update(flag=True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    col = _AsyncCollector(cfg, pool, model, vnorm, infer_device, amp_context(cfg, infer_device), updater, stop)
    window = []
    t_start = time.time() - state["wall"]
    pending = None

    def collect(buf):
        boot, cstats = col.collect(buf)
        episodes, col.episodes = col.episodes, []
        samples, col.err_samples = col.err_samples, []
        trunc, col.trunc = col.trunc, Counter()
        return dict(buf=buf, boot=boot, cstats=cstats, episodes=episodes, errors=col.errors.copy(),
                    samples=samples, trunc=trunc,
                    curriculum=copy.deepcopy(col.curriculum) if cfg.asc_curriculum else None)

    def start(batch):
        if split:
            t = time.time()
            buffers[1].copy_from(batch["buf"])
            batch["buf"] = buffers[1]
            batch["cstats"]["buffer_copy_time"] = time.time() - t
        frac = state["update"] / max(cfg.total_updates, 1)
        for gparam in opt.param_groups:
            gparam["lr"] = (state["lr_now"] if cfg.lr_kl_target > 0 else
                            cfg.lr * (1 - (1 - cfg.lr_final_frac) * frac)) * gparam.get("lr_mult", 1.0)
        buf, boot = batch["buf"], batch["boot"]
        if cfg.lam_build is None:
            adv, ret, mask = buf.gae(boot, cfg.gamma, cfg.lam)
        else:
            adv, ret, mask, bstats = buf.gae_build(boot, cfg.gamma, cfg.lam, cfg.lam_build)
            batch["cstats"].update(bstats)
        adv, pstats_pi = policy_advantages(buf, boot, adv, cfg)
        batch["cstats"].update(pstats_pi)
        distill = {}
        if teacher is not None:
            rel = state["distill_update"] / cfg.distill_updates if cfg.distill_updates > 0 else 1.0
            distill = dict(teacher=teacher, distill_coef_now=cfg.distill_coef * max(0.0, 1 - rel))
        updater.start(buf, adv, ret, mask, distill)

    def finish(batch):
        nonlocal window
        pstats, update_wait = updater.wait()
        updater.refresh(col.model, col.vnorm)
        pstats.update(update_wait=update_wait, kl_ctrl=pstats["step_kl"] if cfg.lr_kl_signal == "step"
                      else max(pstats["approx_kl"] - pstats["approx_kl_first"], 1e-8))
        if teacher is not None:
            state["distill_update"] += 1
        if cfg.lr_kl_target > 0:
            state["lr_now"], state["kl_ema"] = kl_lr_step(state["lr_now"], state["kl_ema"], pstats["kl_ctrl"], cfg)
            pstats["kl_ema"] = state["kl_ema"]
        state["update"] += 1
        state["env_steps"] += batch["buf"].n
        pool.env_steps.value = state["env_steps"]
        all_eps = batch["episodes"]
        eps = [e for e in all_eps if not e.get("midrun")]
        midrun = [e for e in all_eps if e.get("midrun") and not e.get("aborted")]
        state["episodes"] += sum(1 for e in eps if not e.get("aborted"))
        if cfg.asc_curriculum:
            state["asc_frontier"] = batch["curriculum"].frontier
        log_episodes(ep_log, action_log, state, all_eps)
        window = (window + eps)[-200:]
        state["wall"] = time.time() - t_start
        cstats = batch["cstats"]
        dec = cstats.pop("dec_counts")
        dec = dec / max(dec.sum(), 1)
        row = {"update": state["update"], "env_steps": state["env_steps"], "wall": round(state["wall"], 1),
               "episodes_total": state["episodes"], "lr": opt.param_groups[0]["lr"],
               **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in cstats.items()},
               **{k: v if k in ("kl_ctrl", "step_kl") else round(v, 5) for k, v in pstats.items()},
               **{f"u_{k}": v for k, v in episode_stats(eps).items()},
               **{f"w_{k}": v for k, v in episode_stats(window).items()},
               "engine_errors": batch["errors"]["engine_errors"],
               "vnorm_mean": round(vnorm.mean_std[0], 4), "vnorm_std": round(vnorm.mean_std[1], 4),
               **{f"dec_{decision_name(i)}": round(float(x), 4) for i, x in enumerate(dec)}}
        if cfg.asc_curriculum:
            row.update(batch["curriculum"].stats())
        if cfg.midrun_frac > 0:
            row.update(midrun_eps=len(midrun), midrun_steps=sum(e["steps"] for e in midrun),
                       midrun_win=float(np.mean([bool(e["victory"]) for e in midrun])) if midrun else "")
        log.write(row)
        if tb is not None:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    tb.add_scalar(k, v, state["env_steps"])
        if state["update"] % cfg.log_every == 0:
            print(f"[{state['update']:5d}] steps={state['env_steps']:>10d} env_sps={cstats['env_sps']:7.0f} "
                  f"learn_sps={pstats['learner_sps']:7.0f} eps={row['w_episodes']:3d} "
                  f"win={_fmt(row['w_win_rate'])} floor={_fmt(row['w_floor_mean'])} ret={_fmt(row['w_return_mean'])} "
                  f"ent={pstats['entropy']:.3f} kl={pstats['approx_kl']:.4f} vl={pstats['v_loss']:.4f} "
                  f"ev={pstats['explained_var']:.3f} err={batch['errors']['engine_errors']}", flush=True)
            if batch["samples"]:
                print("  engine errors (sample; all in engine_errors.log):", *[x[:200] for x in batch["samples"][:3]],
                      sep="\n    ", flush=True)
            if batch["trunc"]:
                print("  encoder truncations:", dict(batch["trunc"]), flush=True)
        if state["update"] % cfg.ckpt_every == 0 or stop.get("ckpt_owed"):   # owed: the last attempt did not fit on the disk
            stop["ckpt_owed"] = save_ckpt(ckpt_dir, model, opt, vnorm, cfg, state, cfg.keep_ckpts, rng) is False

    try:
        if state["update"] < cfg.total_updates and not stop["flag"]:
            batch = collect(buffers[0])
            while not stop["flag"]:
                start(batch)
                pending = batch
                next_buf = buffers[0] if split or batch["buf"] is buffers[1] else buffers[1]
                batch = collect(next_buf)
                finish(pending)
                pending = None
                if state["update"] >= cfg.total_updates:
                    break
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    except _StopCollection:
        pass
    finally:
        try:
            if pending is not None:
                finish(pending)
            if state["update"] > 0:
                save_ckpt(ckpt_dir, model, opt, vnorm, cfg, state, cfg.keep_ckpts, rng)
        finally:
            updater.executor.shutdown(wait=True)
            pool.close()
            col.err_file.close()
            if tb is not None:
                tb.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return state


def train(cfg: Config):
    run_dir = cfg.resolved_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = Policy(cfg.model).to(device)
    opt = make_optimizer(model, cfg)
    vnorm = ValueNorm()
    state = {"update": 0, "env_steps": 0, "episodes": 0, "wall": 0.0}
    teacher = None
    if cfg.teacher:
        teacher_model, teacher_vnorm, teacher_cfg, _ = load_policy(cfg.teacher, device)
        if teacher_cfg.caps != cfg.caps:
            raise ValueError("teacher caps must match student caps exactly")
        teacher_model.eval().requires_grad_(False)
        teacher = teacher_model, teacher_vnorm, teacher_cfg
        vnorm.load_state_dict(teacher_vnorm.state_dict())
    asc_frontier = cfg.asc_start
    ckpt_dir = os.path.join(run_dir, "ckpt")
    resumed = {}
    if cfg.resume:
        path = os.path.join(ckpt_dir, "latest.pt") if cfg.resume == "auto" else cfg.resume
        if cfg.resume == "auto" and not os.path.exists(path):
            # latest.pt lost (older code removed it before writing the new link): the newest checkpoint, never a silent restart
            found = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_[0-9]*.pt")))
            path = found[-1] if found else path
        if os.path.exists(path):
            ck = torch.load(path, map_location=device, weights_only=False)
            resumed = ck
            model.load_state_dict(ck["model"])
            load_opt_state(opt, model, ck["opt"], cfg, ck["cfg"])
            vnorm.load_state_dict(ck["vnorm"])
            state = {k: ck[k] for k in state}
            if teacher is not None:
                state["distill_update"] = ck.get("distill_update", 0)
            if cfg.asc_curriculum:
                asc_frontier = ck.get("asc_frontier", cfg.asc_start)
            if "torch_rng_state" in ck:
                torch.set_rng_state(ck["torch_rng_state"].cpu())
            if "cuda_rng_state" in ck and torch.cuda.is_available():
                # a checkpoint saved with more visible GPUs (infer_device) carries more states than this process has devices
                torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng_state"]][:torch.cuda.device_count()])
            if "numpy_rng_state" in ck:
                rng.bit_generator.state = ck["numpy_rng_state"]
            print(f"resumed from {path} at update {state['update']}", flush=True)
        elif cfg.resume != "auto":
            raise FileNotFoundError(path)
    if teacher is not None:
        state.setdefault("distill_update", 0)
    if cfg.lr_kl_target > 0:  # the adapted lr and the KL average continue on resume; otherwise start at cfg.lr
        state["lr_now"] = resumed.get("lr_now", cfg.lr)
        state["kl_ema"] = resumed.get("kl_ema")
    save_config(cfg, os.path.join(run_dir, "config.json"))
    print(f"run_dir={run_dir} device={device} params={n_params(model) / 1e6:.2f}M envs={cfg.n_envs}", flush=True)

    amp_ctx = amp_context(cfg, device)
    log = CSVLog(os.path.join(run_dir, "progress.csv"),
                 expand=bool(cfg.branch_decisions) or cfg.lam_build is not None or cfg.explore_eps > 0 or cfg.asc_curriculum
                        or bool(cfg.teacher) or cfg.lr_kl_target > 0 or cfg.midrun_frac > 0 or cfg.async_update
                        or cfg.force_enabled or cfg.taste_enabled)
    ep_log = CSVLog(os.path.join(run_dir, "episodes.csv"),
                    expand=cfg.explore_eps > 0 or cfg.asc_curriculum or cfg.midrun_frac > 0 or cfg.force_enabled or cfg.taste_enabled)
    action_log = ActionLog(run_dir, cfg) if cfg.log_actions else None
    tb = None
    if cfg.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(os.path.join(run_dir, "tb"))
        except Exception as ex:
            print("tensorboard unavailable:", ex)

    if cfg.asc_curriculum:
        state["asc_frontier"] = asc_frontier
        pool = ActorPool(cfg, resume_steps=state["env_steps"], asc_frontier=asc_frontier)
    else:
        pool = ActorPool(cfg, resume_steps=state["env_steps"])
    if cfg.async_update:
        return _train_async(cfg, device, model, opt, vnorm, state, teacher, rng, amp_ctx,
                            pool, log, ep_log, tb, ckpt_dir, action_log)
    buf = Rollout(cfg.caps, cfg.rollout_steps, cfg.n_envs, device)
    if cfg.lam_build is not None:
        buf.track_combat = True
    col = Collector(cfg, pool, model, vnorm, device, amp_ctx)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))
    window = []
    t_start = time.time() - state["wall"]
    try:
        while state["update"] < cfg.total_updates and not stop["flag"]:
            frac = state["update"] / max(cfg.total_updates, 1)
            for gparam in opt.param_groups:
                gparam["lr"] = (state["lr_now"] if cfg.lr_kl_target > 0 else
                                cfg.lr * (1 - (1 - cfg.lr_final_frac) * frac)) * gparam.get("lr_mult", 1.0)
            boot, cstats = col.collect(buf)
            if cfg.lam_build is None:
                adv, ret, mask = buf.gae(boot, cfg.gamma, cfg.lam)
            else:
                adv, ret, mask, bstats = buf.gae_build(boot, cfg.gamma, cfg.lam, cfg.lam_build)
                cstats.update(bstats)
            if cfg.lam_pi is not None:
                adv, pstats_pi = policy_advantages(buf, boot, adv, cfg)
                cstats.update(pstats_pi)
            if cfg.group_k > 1:
                adv, gstats = buf.group_advantages(adv, cfg.gamma, cfg.group_alpha)
                cstats.update(gstats)
            model.train()
            distill = {}
            if teacher is not None:
                rel = state["distill_update"] / cfg.distill_updates if cfg.distill_updates > 0 else 1.0
                distill = dict(teacher=teacher, distill_coef_now=cfg.distill_coef * max(0.0, 1 - rel))
            pstats = ppo_update(model, opt, buf, adv, ret, mask, vnorm, cfg, rng, amp_ctx, **distill)
            if cfg.lr_kl_target <= 0:
                pstats.pop("approx_kl_first", None)
            if teacher is not None:
                state["distill_update"] += 1
            if cfg.lr_kl_target > 0:
                state["lr_now"], state["kl_ema"] = kl_lr_step(
                    state["lr_now"], state["kl_ema"], pstats["step_kl" if cfg.lr_kl_signal == "step" else "approx_kl"], cfg)
                pstats["kl_ema"] = state["kl_ema"]
            state["update"] += 1
            state["env_steps"] += buf.n
            pool.env_steps.value = state["env_steps"]
            all_eps, col.episodes = col.episodes, []
            eps = [e for e in all_eps if not e.get("midrun")]
            midrun = [e for e in all_eps if e.get("midrun") and not e.get("aborted")]
            state["episodes"] += sum(1 for e in eps if not e.get("aborted"))
            if cfg.asc_curriculum:
                state["asc_frontier"] = col.curriculum.frontier
            log_episodes(ep_log, action_log, state, all_eps)
            window = (window + eps)[-200:]
            state["wall"] = time.time() - t_start
            if state["update"] % cfg.log_every == 0:
                dec = cstats.pop("dec_counts")
                dec = dec / max(dec.sum(), 1)
                row = {"update": state["update"], "env_steps": state["env_steps"], "wall": round(state["wall"], 1),
                       "episodes_total": state["episodes"], "lr": opt.param_groups[0]["lr"],
                       **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in cstats.items()},
                       **{k: v if k == "step_kl" else round(v, 5) for k, v in pstats.items()},
                       **{f"u_{k}": v for k, v in episode_stats(eps).items()},
                       **{f"w_{k}": v for k, v in episode_stats(window).items()},
                       "engine_errors": col.errors["engine_errors"],
                       "vnorm_mean": round(vnorm.mean_std[0], 4), "vnorm_std": round(vnorm.mean_std[1], 4),
                       **{f"dec_{decision_name(i)}": round(float(x), 4) for i, x in enumerate(dec)}}
                if cfg.explore_eps > 0:
                    done = [e for e in eps if not e.get("aborted")]
                    main = episode_stats([e for e in window if not e["explore"]])
                    row.update(explore_ep_frac=float(np.mean([e["explore"] for e in done])) if done else "",
                               w_win_rate_main=main["win_rate"], w_floor_mean_main=main["floor_mean"])
                if col.force_on or col.taste_on:
                    flag = "taste" if col.taste_on else "force"
                    done = [e for e in eps if not e.get("aborted")]
                    row.update({f"{flag}_ep_frac": float(np.mean([e[flag] for e in done])) if done else "",
                                f"{flag}_win": episode_stats([e for e in eps if e[flag]])["win_rate"],
                                "main_win": episode_stats([e for e in eps if not e[flag]])["win_rate"]})
                if col.taste_on:
                    row["taste_kl"] = cstats["taste_kl"]
                if cfg.asc_curriculum:
                    row.update(col.curriculum.stats())
                if cfg.midrun_frac > 0:
                    row.update(midrun_eps=len(midrun), midrun_steps=sum(e["steps"] for e in midrun),
                               midrun_win=float(np.mean([bool(e["victory"]) for e in midrun])) if midrun else "")
                log.write(row)
                if tb is not None:
                    for k, v in row.items():
                        if isinstance(v, (int, float)):
                            tb.add_scalar(k, v, state["env_steps"])
                print(f"[{state['update']:5d}] steps={state['env_steps']:>10d} env_sps={cstats['env_sps']:7.0f} "
                      f"learn_sps={pstats['learner_sps']:7.0f} eps={row['w_episodes']:3d} "
                      f"win={_fmt(row['w_win_rate'])} floor={_fmt(row['w_floor_mean'])} ret={_fmt(row['w_return_mean'])} "
                      f"ent={pstats['entropy']:.3f} kl={pstats['approx_kl']:.4f} vl={pstats['v_loss']:.4f} "
                      f"ev={pstats['explained_var']:.3f} err={col.errors['engine_errors']}", flush=True)
                if col.err_samples:
                    print("  engine errors (sample; all in engine_errors.log):", *[x[:200] for x in col.err_samples[:3]],
                          sep="\n    ", flush=True)
                    col.err_samples = []
                if col.trunc:
                    print("  encoder truncations:", dict(col.trunc), flush=True)
                    col.trunc.clear()
            if state["update"] % cfg.ckpt_every == 0 or stop.get("ckpt_owed"):   # owed: the last attempt did not fit on the disk
                stop["ckpt_owed"] = save_ckpt(ckpt_dir, model, opt, vnorm, cfg, state, cfg.keep_ckpts, rng) is False
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        if state["update"] > 0:
            save_ckpt(ckpt_dir, model, opt, vnorm, cfg, state, cfg.keep_ckpts, rng)
        pool.close()
        if tb is not None:
            tb.close()
    return state


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "  -  "


def main(argv=None):
    quiet_stdio()
    cfg = parse_args(argv)
    train(cfg)


if __name__ == "__main__":
    main()
