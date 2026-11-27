"""Rollout storage, GAE over per-env transition sequences, value normalization and the PPO update."""
import copy
import math
import time

import numpy as np
import torch

from .encode import TT, field_spec

TORCH_DT = {np.uint8: torch.uint8, np.int8: torch.int8, np.int16: torch.int16, np.int32: torch.int32,
            np.float16: torch.float16, np.float32: torch.float32}


def to_torch(batch, device):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in batch.items()}


class Rollout:
    """Flat buffer of `cap` transitions. Observations live on the configured device, scalars on the CPU.
    Transitions of one env are linked in collection order through `env_idx[e]`."""

    def __init__(self, caps, cap, n_envs, device, track_combat=False):
        self.cap, self.n_envs, self.device = cap, n_envs, device
        self.track_combat = track_combat
        self.obs = {k: torch.zeros((cap,) + shape, dtype=TORCH_DT[np.dtype(dt).type], device=device)
                    for k, (shape, dt) in field_spec(caps).items()}
        self.act = np.zeros(cap, np.int64)
        self.logp = np.zeros(cap, np.float32)
        self.val = np.zeros(cap, np.float32)      # denormalized value estimates
        self.rew = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, bool)
        self.trunc = np.zeros(cap, bool)
        self.cut = np.zeros(cap, bool)
        self.bootstrap = np.zeros(cap, np.float32)
        self.has_res = np.zeros(cap, bool)
        self.forced = np.zeros(cap, bool)
        self.combat = np.zeros(cap, bool)
        self.explored = np.zeros(cap, bool)   # action replaced by build exploration (rl/train.py)
        self.reset()

    def reset(self):
        for v in self.obs.values():  # zero once per rollout so add() can write only the used prefix
            v.zero_()
        self.n = 0
        self.env_idx = [[] for _ in range(self.n_envs)]
        self.has_res[:] = False
        self.cut[:] = False
        self.explored[:] = False
        self.groups = []
        self.branches = []
        self.branch_extra = 0
        self.forced[:] = False

    def grow(self, cap):
        for k, v in self.obs.items():
            dst = torch.zeros((cap,) + v.shape[1:], dtype=v.dtype, device=self.device)
            dst[:self.cap] = v
            self.obs[k] = dst
        for k in ("act", "logp", "val", "rew", "done", "trunc", "cut", "bootstrap", "has_res", "forced", "combat", "explored"):
            v = getattr(self, k)
            dst = np.zeros(cap, v.dtype)
            dst[:self.cap] = v
            setattr(self, k, dst)
        self.cap = cap

    def copy_from(self, source):
        n = source.n
        assert n <= self.cap
        for k, dst in self.obs.items():
            dst[:n].copy_(source.obs[k][:n])
        for k in ("act", "logp", "val", "rew", "done", "trunc", "cut", "bootstrap", "has_res", "forced", "combat", "explored"):
            getattr(self, k)[:n] = getattr(source, k)[:n]
        self.n = n
        self.env_idx = copy.deepcopy(source.env_idx)
        self.groups = copy.deepcopy(source.groups)
        self.branches = copy.deepcopy(source.branches)
        self.branch_extra = source.branch_extra
        if torch.device(self.device).type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def add(self, envs, obs, act, logp, val):
        b = len(envs)
        i0 = self.n
        assert i0 + b <= self.cap
        for k, v in obs.items():
            dst = self.obs[k][i0:i0 + b]
            if v.dim() >= 2 and v.shape[1] != dst.shape[1]:
                dst[:, : v.shape[1]] = v
            else:
                dst.copy_(v)
        self.act[i0:i0 + b] = act
        self.logp[i0:i0 + b] = logp
        self.val[i0:i0 + b] = val
        if self.track_combat:
            self.combat[i0:i0 + b] = (obs["tok_type"] == TT["COMBAT"]).any(dim=1).cpu().numpy()
        for j, e in enumerate(envs):
            self.env_idx[int(e)].append(i0 + j)
        self.n += b
        return np.arange(i0, i0 + b)

    def set_result(self, i, r, done, trunc):
        self.rew[i], self.done[i], self.trunc[i], self.has_res[i] = r, done, trunc, True

    def gae(self, boot_val, gamma, lam):
        """boot_val[e]: value of the obs following env e's last transition (ignored if it ended the episode).
        lam: scalar or one lambda per flat transition.
        Returns advantages, returns, and the loss mask (False for truncated or forced transitions)."""
        lens = [len(ix) for ix in self.env_idx]
        L = max(lens) if lens else 0
        E = self.n_envs
        M = np.full((E, L), -1, np.int64)
        for e, ix in enumerate(self.env_idx):
            M[e, : len(ix)] = ix
        valid = M >= 0
        Mi = np.where(valid, M, 0)
        lam_seq = None if np.ndim(lam) == 0 else np.asarray(lam, np.float64)[Mi]
        rew, val = self.rew[Mi], self.val[Mi]
        done, trunc = self.done[Mi], self.trunc[Mi]
        cut, bootstrap = self.cut[Mi], self.bootstrap[Mi]
        adv = np.zeros((E, L), np.float32)
        next_v = np.asarray(boot_val, np.float32).copy()
        next_a = np.zeros(E, np.float32)
        for t in range(L - 1, -1, -1):
            ok, tr = valid[:, t], trunc[:, t]
            nonterm = 1.0 - done[:, t]
            nv = np.where(cut[:, t], bootstrap[:, t], next_v)
            na = np.where(cut[:, t], 0.0, next_a)
            delta = rew[:, t] + gamma * nv * nonterm - val[:, t]
            lt = lam if lam_seq is None else lam_seq[:, t]
            a = np.where(tr, 0.0, delta + gamma * lt * nonterm * na)
            adv[:, t] = a
            next_v = np.where(ok, val[:, t], next_v)
            next_a = np.where(ok, a, next_a)
        flat_adv = np.zeros(self.n, np.float32)
        flat_adv[M[valid]] = adv[valid]
        mask = ~self.trunc[: self.n] & ~self.forced[: self.n]
        ret = flat_adv + self.val[: self.n]
        return flat_adv, ret, mask

    def gae_build(self, boot_val, gamma, lam, lam_build):
        adv, ret, mask = self.gae(boot_val, gamma, lam)
        combat = self.combat[:self.n]
        build = ~combat
        lams = np.ones(self.n, np.float64)
        for ix in self.env_idx:
            prev, nxt = np.asarray(ix[:-1], np.int64), np.asarray(ix[1:], np.int64)
            lams[prev[build[nxt]]] = lam_build
        adv_build, ret_build, _ = self.gae(boot_val, gamma, lams)
        std_build = float(adv[build].std()) if build.any() else 0.0
        adv[build], ret[build] = adv_build[build], ret_build[build]
        stats = dict(build_frac=float(build.mean()) if self.n else 0.0,
                     adv_std_combat=float(adv[combat].std()) if combat.any() else 0.0,
                     adv_std_build=float(adv[build].std()) if build.any() else 0.0,
                     adv_std_build_std=std_build)
        return adv, ret, mask, stats

    def group_advantages(self, adv, gamma, alpha):
        out = adv.copy()
        win_rates, outcome_std = [], []
        extra = 0
        for group in self.groups:
            trials = group["trials"]
            outcomes = []
            for ix, boot in zip(trials, group["boots"]):
                outcome = float(boot)
                for i in reversed(ix):
                    outcome = float(self.rew[i]) + gamma * outcome
                outcomes.append(outcome)
            outcomes = np.asarray(outcomes, np.float64)
            if len(trials) > 1 and alpha != 1:
                loo = (len(trials) * outcomes - outcomes.sum()) / (len(trials) - 1)
                for ix, g in zip(trials, loo):
                    out[ix] = alpha * adv[ix] + (1 - alpha) * g
            win_rates.append(np.mean(group["wins"]))
            outcome_std.append(outcomes.std())
            extra += sum(map(len, trials[:-1]))
        return out, dict(group_count=len(self.groups), group_win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
                         group_outcome_std=float(np.mean(outcome_std)) if outcome_std else 0.0,
                         group_extra_frac=extra / max(self.n, 1))

    def branch_targets(self, gamma, std, temp):
        targets, choices, indices = [], [], []
        outcomes, probs, policies = {}, {}, {}
        for b in self.branches:
            q = {}
            for action, ix, boot in zip(b["actions"], b["trials"], b["boots"]):
                value = float(boot)
                for i in reversed(ix):
                    value = float(self.rew[i]) + gamma * value
                q[action] = value
            opts = b["options"]
            values = np.array([q[j] for j in opts])
            old = b["logits"][opts].astype(np.float64)
            scores = old + np.clip((values - values.mean()) / std, -5, 5) / temp
            p = np.exp(scores - scores.max())
            p /= p.sum()
            policy = np.exp(old - old.max())
            policy /= policy.sum()
            target = np.zeros(self.obs["cand_key"].shape[1], np.float32)
            choice = np.zeros_like(target, bool)
            target[opts], choice[opts] = p, True
            targets.append(target)
            choices.append(choice)
            indices.append(b["trials"][-1][0])
            for key, value, prob, prior in zip(b["keys"], values, p, policy):
                outcomes.setdefault(key, []).append(value)
                probs.setdefault(key, []).append(prob)
                policies.setdefault(key, []).append(prior)
        stats = dict(branch_count=len(self.branches), branch_extra_frac=self.branch_extra / max(self.n, 1))
        for key in outcomes:
            stats.update({f"branch_n_{key}": len(outcomes[key]), f"branch_q_{key}": float(np.mean(outcomes[key])),
                          f"branch_target_{key}": float(np.mean(probs[key])),
                          f"branch_policy_{key}": float(np.mean(policies[key]))})
        return indices, targets, choices, stats


class ValueNorm:
    """Running mean/std of value targets (EMA with debiasing, as in MAPPO)."""

    def __init__(self, beta=0.995, eps=1e-5):
        self.beta, self.eps = beta, eps
        self.m = self.m2 = self.w = 0.0

    def update(self, x):
        x = np.asarray(x, np.float64)
        if x.size == 0:
            return
        b = self.beta
        self.m = b * self.m + (1 - b) * x.mean()
        self.m2 = b * self.m2 + (1 - b) * (x ** 2).mean()
        self.w = b * self.w + (1 - b)

    @property
    def mean_std(self):
        if self.w == 0:
            return 0.0, 1.0
        m = self.m / self.w
        var = max(self.m2 / self.w - m * m, 1e-4)
        return m, math.sqrt(var)

    def normalize(self, x):
        m, s = self.mean_std
        return (x - m) / s

    def denormalize(self, x):
        m, s = self.mean_std
        return x * s + m

    def state_dict(self):
        return {"m": self.m, "m2": self.m2, "w": self.w}

    def load_state_dict(self, d):
        self.m, self.m2, self.w = d["m"], d["m2"], d["w"]


def sliced(obs, idx, T=None, C=None):
    """Gather a minibatch and trim padding to its largest token / candidate count.
    Pass T and C (computed on the CPU) to avoid a device sync."""
    if T is None:
        T = max(int(obs["n_tok"][idx].max()), 1)
    if C is None:
        C = max(int(obs["n_cand"][idx].max()), 1)
    out = {}
    for k, v in obs.items():
        if k.startswith("tok_"):
            out[k] = v[idx, :T]
        elif k.startswith("cand_"):
            out[k] = v[idx, :C]
        else:
            out[k] = v[idx]
    return out


def bucketed_minibatches(n_tok, idx, mb, rng, bucket=8):
    """Shuffle, then sort chunks of `bucket` minibatches by token count so padding is small."""
    perm = idx[rng.permutation(len(idx))]
    out = []
    chunk = mb * bucket
    for s in range(0, len(perm), chunk):
        c = perm[s:s + chunk]
        c = c[np.argsort(n_tok[c], kind="stable")]
        out += [c[j:j + mb] for j in range(0, len(c), mb)]
    rng.shuffle(out)
    return out


def masked_logp_entropy(logits, act, cmask):
    logp_all = torch.log_softmax(logits, -1)
    logp = logp_all.gather(1, act[:, None]).squeeze(1)
    p = logp_all.exp()
    ent = -(p * logp_all).masked_fill(~cmask, 0.0).sum(-1)
    return logp, ent


def branch_cross_entropy(logits, target, choices):
    logp = torch.log_softmax(logits.masked_fill(~choices, -1e9), -1)
    return -(target * logp).sum(-1)


@torch.no_grad()
def probe_logp(model, buf, order, n_tok, n_cand, micro_batch, amp_ctx):
    """Masked log-probabilities of every candidate for the transitions `order`, in chunks of micro_batch."""
    out = []
    for s in range(0, len(order), micro_batch):
        ix = order[s:s + micro_batch]
        ti = torch.as_tensor(ix, device=buf.device)
        T = min(-(-max(int(n_tok[ix].max()), 1) // 8) * 8, buf.obs["tok_type"].shape[1])
        C = min(-(-max(int(n_cand[ix].max()), 1) // 8) * 8, buf.obs["cand_key"].shape[1])
        batch = sliced(buf.obs, ti, T, C)
        with amp_ctx():
            logits, _ = model(batch)
        cmask = torch.arange(logits.shape[1], device=buf.device)[None] < batch["n_cand"].long()[:, None]
        out.append((torch.log_softmax(logits.float(), -1), cmask))
    return out


def ppo_update(model, opt, buf, adv, ret, mask, vnorm, cfg, rng, amp_ctx, teacher=None, distill_coef_now=0.0):
    """One PPO update over the buffer. Returns a dict of statistics."""
    n = buf.n
    taste_on = cfg.taste_enabled
    replaced_actions = cfg.explore_eps > 0 or cfg.force_enabled or taste_on
    drop_replaced = cfg.force_enabled and not cfg.force_learn or taste_on
    dev = buf.device
    t0 = time.time()
    idx_all = np.nonzero(mask)[0]
    a = adv[idx_all]
    adv_n = np.zeros_like(adv)
    adv_n[idx_all] = (a - a.mean()) / (a.std() + 1e-8)
    if cfg.value_norm:
        vnorm.update(ret[idx_all])
        tgt = vnorm.normalize(ret).astype(np.float32)
    else:
        tgt = ret.astype(np.float32)
    g = lambda x, dt=torch.float32: torch.as_tensor(x[:n], dtype=dt, device=dev)
    T_adv, T_tgt, T_logp, T_act = g(adv_n), g(tgt), g(buf.logp), g(buf.act, torch.int64)
    n_tok_cpu = buf.obs["n_tok"][:n].cpu().numpy().astype(np.int64)
    n_cand_cpu = buf.obs["n_cand"][:n].cpu().numpy().astype(np.int64)
    branch_stats, branch_acc = {}, 0.0
    branch_row = np.full(n, -1, np.int64)
    if cfg.branch_decisions:
        indices, targets, choices, branch_stats = buf.branch_targets(cfg.gamma, vnorm.mean_std[1], cfg.branch_temp)
        branch_row[indices] = np.arange(len(indices))
        if indices:
            branch_target = torch.as_tensor(np.asarray(targets), device=dev)
            branch_choice = torch.as_tensor(np.asarray(choices), device=dev)

    probe = None
    if cfg.kl_probe > 0:  # evenly spaced in collection order: no draw from rng
        probe_rows = idx_all[~buf.explored[idx_all]] if taste_on else idx_all
        if len(probe_rows):
            probe = probe_rows[:: max(1, len(probe_rows) // cfg.kl_probe)][:cfg.kl_probe]
            probe = probe[np.argsort(n_tok_cpu[probe], kind="stable")]
            probe_old = probe_logp(model, buf, probe, n_tok_cpu, n_cand_cpu, cfg.micro_batch, amp_ctx)
    keys = ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "grad_norm")
    acc = torch.zeros(len(keys), device=dev)  # accumulated on device: no per-step host syncs
    if teacher is not None:
        distill_acc = torch.zeros(2, device=dev)
    n_mb = n_samples = 0
    kl_first = torch.zeros((), device=dev)
    n_first = 0
    stop = False
    ex_ratio, ex_n = 0.0, 0
    skipped = 0
    for epoch in range(cfg.epochs):
        kl_epoch = torch.zeros((), device=dev)
        n_epoch = 0
        for mb in bucketed_minibatches(n_tok_cpu, idx_all, cfg.minibatch, rng, cfg.mb_bucket):
            opt.zero_grad(set_to_none=True)
            for s in range(0, len(mb), cfg.micro_batch):
                ix = mb[s:s + cfg.micro_batch]
                frac = len(ix) / len(mb)
                ti = torch.as_tensor(ix, device=dev)
                T = min(-(-max(int(n_tok_cpu[ix].max()), 1) // 8) * 8, buf.obs["tok_type"].shape[1])
                C = min(-(-max(int(n_cand_cpu[ix].max()), 1) // 8) * 8, buf.obs["cand_key"].shape[1])
                batch = sliced(buf.obs, ti, T, C)
                with amp_ctx():
                    logits, v = model(batch)
                cmask = torch.arange(logits.shape[1], device=dev)[None] < batch["n_cand"].long()[:, None]
                logp, ent = masked_logp_entropy(logits, T_act[ti], cmask)
                lr_new = logp - T_logp[ti]
                if replaced_actions or cfg.dual_clip:  # replaced actions can have tiny old probabilities: keep exp finite
                    lr_new = lr_new.clamp(-20.0, 20.0)
                ratio = torch.exp(lr_new)
                A = T_adv[ti]
                if replaced_actions:  # self-imitation: learn only from replaced actions that did better than expected
                    ex = torch.as_tensor(buf.explored[ix], device=dev)
                    A = torch.where(ex if drop_replaced else ex & (A < 0), torch.zeros_like(A), A)
                surr = torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A)
                if cfg.dual_clip:  # dual-clip PPO: a negative advantage cannot be scaled by a ratio above dual_clip
                    surr = torch.where(A < 0, torch.maximum(surr, cfg.dual_clip * A), surr)
                pg = -surr.mean()
                vl = 0.5 * ((v - T_tgt[ti]) ** 2).mean()
                el = ent.mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * el
                if teacher is not None and distill_coef_now > 0:
                    teacher_model, teacher_vnorm, teacher_cfg = teacher
                    with torch.no_grad():
                        with amp_ctx():
                            teacher_logits, teacher_v = teacher_model(batch)
                        teacher_logp = torch.log_softmax(teacher_logits.float().masked_fill(~cmask, -1e9), -1)
                        target_v = teacher_vnorm.denormalize(teacher_v) if teacher_cfg.value_norm else teacher_v
                        if cfg.value_norm:
                            target_v = vnorm.normalize(target_v)
                    student_logp = torch.log_softmax(logits.masked_fill(~cmask, -1e9), -1)
                    dkl = (teacher_logp.exp() * (teacher_logp - student_logp)).masked_fill(~cmask, 0.0).sum(-1).mean()
                    vmse = ((v - target_v) ** 2).mean()
                    loss = loss + distill_coef_now * (dkl + cfg.distill_value_coef * vmse)
                    distill_acc += torch.stack((dkl.detach(), vmse.detach())) * len(ix)
                if cfg.branch_decisions and np.any(branch_row[ix] >= 0):
                    sel = torch.as_tensor(np.nonzero(branch_row[ix] >= 0)[0], device=dev)
                    rows = torch.as_tensor(branch_row[ix][branch_row[ix] >= 0], device=dev)
                    ce = branch_cross_entropy(logits[sel], branch_target[rows, :logits.shape[1]],
                                              branch_choice[rows, :logits.shape[1]]).sum()
                    loss = loss + cfg.branch_coef * ce * len(idx_all) / (len(indices) * len(ix))
                    branch_acc += float(ce.detach())
                (loss * frac).backward()
                with torch.no_grad():
                    lr_ = lr_new.detach()
                    if replaced_actions:  # statistics over sampled actions only; replaced ones are tracked apart
                        keep = (~ex).float()
                        kl = (((torch.exp(lr_) - 1) - lr_) * keep).sum() / keep.sum().clamp(min=1)
                        cf = (((ratio - 1).abs() > cfg.clip).float() * keep).sum() / keep.sum().clamp(min=1)
                        ex_ratio += float((ratio.clamp(max=1e6) * ex.float()).sum())
                        ex_n += int(ex.sum())
                    else:
                        kl = ((torch.exp(lr_) - 1) - lr_).mean()
                        cf = ((ratio - 1).abs() > cfg.clip).float().mean()
                    acc[:5] += torch.stack((pg.detach(), vl.detach(), el.detach(), kl, cf)) * len(ix)
                    kl_epoch += kl * len(ix)
                    if n_mb == 0:
                        kl_first += kl * len(ix)
                        n_first += len(ix)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if torch.isfinite(gn):
                opt.step()
            else:  # never apply a non-finite step: one bad minibatch must not corrupt the weights
                opt.zero_grad(set_to_none=True)
                gn = torch.zeros_like(gn)
                skipped += 1
            acc[5] += gn.detach()
            n_mb += 1
            n_samples += len(mb)
            n_epoch += len(mb)
        if cfg.target_kl > 0 and float(kl_epoch) / max(n_epoch, 1) > cfg.target_kl:
            stop = True
            break
    if dev.type == "cuda":
        if cfg.async_update:
            torch.cuda.current_stream(dev).synchronize()
        else:
            torch.cuda.synchronize()
    dt = time.time() - t0
    out = {k: float(x) / max(n_mb if k == "grad_norm" else n_samples, 1) for k, x in zip(keys, acc.cpu().numpy())}
    out["approx_kl_first"] = float(kl_first) / max(n_first, 1)
    if probe is not None:
        new = probe_logp(model, buf, probe, n_tok_cpu, n_cand_cpu, cfg.micro_batch, amp_ctx)
        kl_sum = sum(float((lo.exp() * (lo - ln)).masked_fill(~cm, 0.0).sum()) for (lo, cm), (ln, _) in zip(probe_old, new))
        out["step_kl"] = kl_sum / len(probe)
    elif cfg.kl_probe > 0 and taste_on:
        out["step_kl"] = 0.0
    v_pred = buf.val[:n][idx_all]
    var_r = np.var(ret[idx_all])
    out.update(learner_sps=n_samples / dt, update_time=dt, epochs_done=epoch + 1, kl_stop=float(stop),
               explained_var=float(1 - np.var(ret[idx_all] - v_pred) / var_r) if var_r > 0 else 0.0,
               adv_mean=float(a.mean()), ret_mean=float(ret[idx_all].mean()), masked_frac=float(1 - mask.mean()))
    if skipped:
        out["skipped_steps"] = skipped
    if replaced_actions:
        out.update(explored_ratio_mean=ex_ratio / max(ex_n, 1), explored_frac=float(buf.explored[:n][idx_all].mean()))
    if cfg.branch_decisions:
        out.update(branch_stats, branch_ce=branch_acc / max(len(buf.branches) * (epoch + 1), 1))
    if teacher is not None:
        dkl, vmse = (distill_acc / max(n_samples, 1)).cpu().tolist()
        out.update(distill_coef_now=distill_coef_now, distill_kl=dkl, distill_vmse=vmse)
    return out
