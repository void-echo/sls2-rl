"""Training configuration. Every field is a CLI flag (`--n_envs 256`, `--characters Ironclad,Silent`);
`--preset tiny` gives a CPU-sized config for smoke tests."""
import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field

from .encode import Caps
from .force import parse_force_list
from .model import ModelCfg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    name: str = "run"
    run_dir: str = ""                  # default: <repo>/runs/<name>
    seed: int = 0
    device: str = "cuda"
    # environment
    engine: str = "real"               # real | fake | fakeproc
    engine_dir: str = os.path.join(REPO, "engine")
    engine_cmd: str = ""               # JSON list to override the launch command
    engine_log: bool = False           # keep per-engine stderr logs in <run_dir>/engine_logs
    restart_every_episode: bool = False
    log_actions: bool = False          # keep every episode's action indices: seed + character + ascension + actions replays a game
                                       # exactly (the engine is deterministic). Training writes them to <run_dir>/actions/*.jsonl.gz,
                                       # evaluation always stores them in its JSON ("actions" of each episode)
    restart_every_n: int = 1           # with restart_every_episode: a fresh engine process every n-th episode only (1 = every
                                       # episode); a warm process spends about half the CPU per step (tools/bench_engine_cpu.py)
    fake_p_error: float = 0.0
    fake_args: str = ""                # extra flags for the fakeproc engine, e.g. "--p_crash 0.001 --step_ms 3"
    characters: str = "Ironclad"       # comma-separated; each episode samples one uniformly
    ascension: int = 0
    asc_curriculum: bool = False
    asc_start: int = 1
    asc_max: int = 10
    asc_threshold: float = 0.6
    asc_window: int = 300
    asc_frontier_frac: float = 0.5
    group_k: int = 1                  # room-entry combat replays; 1 disables grouping
    group_rooms: str = "Elite,Boss"
    group_alpha: float = 0.5          # GAE weight; the remainder uses leave-one-out combat outcomes
    branch_decisions: str = ""       # comma-separated decision types; currently rest_site
    branch_prob: float = 1.0
    branch_max_steps: int = 600
    branch_temp: float = 0.1
    branch_coef: float = 1.0
    # Mid-run starts (training only): a fraction of new episodes load a room-entry save kept from this policy's own games at the
    # frontier level, so that the states where games are decided get more experience. They do not count in the win statistics.
    midrun_frac: float = 0.0
    midrun_rooms: str = "Boss"
    midrun_min_floor: int = 0          # only saves at this total floor or later
    midrun_pool: int = 128             # saves kept per worker (newest first out of a ring)
    midrun_reuse: int = 16             # consecutive loads in one engine process before it is restarted
    # Safety valves, deliberately loose: an infinite combo that is actually winning needs far fewer steps.
    max_episode_steps: int = 0         # whole run; 0 = unlimited
    max_combat_steps: int = 100_000    # decisions within one combat
    max_turn_steps: int = 20_000       # decisions within one combat turn
    timeout_start: float = 300.0       # seconds for boot + rl_start
    timeout_step: float = 120.0        # wall clock for one action (covers engine-side automatic loops)
    # actors
    n_workers: int = 32
    envs_per_worker: int = 8
    min_infer_batch: float = 0.25      # fraction of envs to wait for before an inference call...
    max_infer_wait_ms: float = 4.0     # ...but never wait longer than this
    # PPO
    async_update: bool = False
    async_split_buffers: bool = False
    collect_fast: bool = False
    infer_device: str = ""             # with async_update: run rollout inference on this device (e.g. cuda:1) while the update keeps
                                       # `device`; async_split_buffers also moves collection storage here. Empty = same device
    rollout_steps: int = 32768         # transitions per update (e.g. 256 envs x 128)
    epochs: int = 3
    minibatch: int = 4096
    micro_batch: int = 1024            # gradient accumulation chunk (memory bound)
    mb_bucket: int = 8                 # minibatches sorted together by token count to cut padding. When mb_bucket * minibatch >=
                                       # rollout_steps (every run so far) each minibatch is one token-count stratum, from small
                                       # combat states to build decisions (audit of 11-10); 1 = i.i.d. minibatches, sorted inside
    lr: float = 2.5e-4
    lr_final_frac: float = 1.0         # linear decay to lr*frac over total_updates (1.0 = constant)
    lr_kl_target: float = 0.0          # >0: adapt lr after every update so the EMA of approx KL tracks this value
    lr_kl_min: float = 1e-6
    lr_kl_max: float = 1e-3
    lr_kl_gain: float = 0.5            # factor (target / kl_ema) ** gain, clipped to [1/lr_kl_step, lr_kl_step]
    lr_kl_step: float = 1.1
    lr_kl_ema: float = 0.9
    kl_probe: int = 0                  # >0: exact KL(policy before the update || after it) on this many evenly spaced transitions,
                                       # logged as step_kl (two extra forward passes over the probe)
    lr_kl_signal: str = "approx"       # what lr_kl_target controls: approx (epoch mean of approx_kl; with async_update minus the
                                       # first minibatch's, a difference of two noisy numbers) | step (step_kl, needs kl_probe)
    weight_decay: float = 0.0
    adam_eps: float = 1e-5
    grow_params: str = ""
    grow_lr_mult: float = 1.0
    gamma: float = 0.999
    lam: float = 0.95
    lam_build: float | None = None
    lam_pi: float | None = None        # lambda of the policy's advantages only; value targets keep lam / lam_build. None = same as lam.
                                       # Measured on a frozen policy (tools/critic_gram.py, 11-10): 0.5 gives gradients about 3x the
                                       # sample-size equivalent of 0.95, with the critic trained on lambda 0.95 targets
    lam_pi_build: float | None = None  # the same for build decisions (None = lam_pi)
    explore_eps: float = 0.0
    explore_frac: float = 0.5
    force_ids: str = ""
    force_vars: str = ""
    force_frac: float = 0.0
    force_prob: float = 1.0
    force_subset: float = 0.0          # > 0: each game draws its own subset of force_ids (every id with this probability)
    force_learn: bool = True           # False: forced steps give no policy gradient at all (exposure only); True: the explored rule
    taste_frac: float = 0.0
    taste_beta: float = 0.0
    clip: float = 0.2
    dual_clip: float | None = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.0             # >0: stop the epoch loop early when approx KL exceeds it
    value_norm: bool = True
    teacher: str = ""
    distill_coef: float = 1.0
    distill_updates: int = 300
    distill_value_coef: float = 1.0
    amp: bool = True                   # bf16 autocast on CUDA
    total_updates: int = 100000
    # reward
    win_reward: float = 1.0
    loss_reward: float = 0.0
    floor_reward: float = 0.01         # per increase of info.total_floor
    boss_reward: float = 0.0           # per act boss beaten (info.act increases); the last act's boss is the win
    floor_anneal_steps: int = 0        # >0: floor_reward falls linearly to 0 over this many env steps (per episode)
    shaping_fade_start: int = 0        # env steps at which floor_reward and boss_reward start to fade (per episode)
    shaping_fade_steps: int = 0        # >0: both fall linearly to shaping_fade_to times their value over this many env steps
    shaping_fade_to: float = 0.0
    shaping: str = "per_floor"         # per_floor: floors are rewarded (changes the objective); potential: policy-invariant;
                                       # switch: per_floor until shaping_switch_steps, then potential (per episode)
    shaping_switch_steps: int = 15_000_000
    # logging / checkpoints
    log_every: int = 1
    ckpt_every: int = 25
    keep_ckpts: int = 5
    tensorboard: bool = False
    resume: str = ""                   # path to checkpoint, or "auto" (latest in run_dir)
    # sub-configs
    model: ModelCfg = field(default_factory=ModelCfg)
    caps: Caps = field(default_factory=Caps)

    def __post_init__(self):
        if self.force_enabled and (self.async_update or self.group_k > 1 or self.branch_decisions):
            raise ValueError("force cannot be combined with async_update, group_k > 1 or branch_decisions")
        if self.taste_enabled and (self.async_update or self.group_k > 1 or self.branch_decisions or self.force_enabled):
            raise ValueError("taste cannot be combined with async_update, group_k > 1, branch_decisions or force")
        if self.infer_device and not self.async_update:
            raise ValueError("infer_device needs async_update")
        if self.lr_kl_signal not in ("approx", "step") or (self.lr_kl_signal == "step" and self.kl_probe <= 0):
            raise ValueError("lr_kl_signal must be approx or step, and step needs kl_probe > 0")
        if self.async_update and (self.group_k > 1 or self.branch_decisions or self.explore_eps > 0):
            raise ValueError("async_update cannot be combined with group_k > 1, branch_decisions or explore_eps > 0")
        if self.asc_curriculum and (self.group_k > 1 or self.branch_decisions):
            raise ValueError("asc_curriculum cannot be combined with group_k > 1 or branch_decisions")
        if self.lam_build is not None and (self.group_k > 1 or self.branch_decisions):
            raise ValueError("lam_build cannot be combined with group_k > 1 or branch_decisions")
        if self.explore_eps > 0 and (self.group_k > 1 or self.branch_decisions):
            raise ValueError("explore_eps cannot be combined with group_k > 1 or branch_decisions")
        if self.midrun_frac > 0 and (self.group_k > 1 or self.branch_decisions):
            raise ValueError("midrun_frac cannot be combined with group_k > 1 or branch_decisions")

    @property
    def force_enabled(self):
        return self.force_frac > 0 and bool(parse_force_list(self.force_ids) or parse_force_list(self.force_vars))

    @property
    def taste_enabled(self):
        return self.taste_frac > 0 and self.taste_beta > 0

    @property
    def n_envs(self):
        return self.n_workers * self.envs_per_worker

    def resolved_run_dir(self):
        return self.run_dir or os.path.join(REPO, "runs", self.name)

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        m = ModelCfg(**d.pop("model", {}))
        c = Caps.from_dict(d.pop("caps", {}))
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(model=m, caps=c, **{k: v for k, v in d.items() if k in names})


PRESETS = {
    "tiny": dict(device="cpu", engine="fake", n_workers=2, envs_per_worker=4, rollout_steps=512, epochs=2,
                 minibatch=256, micro_batch=256, ckpt_every=5, model=dict(d=64, layers=1, heads=4, cand_layers=1)),
    "fake": dict(engine="fake"),
}


def _add_fields(ap, dc, prefix=""):
    for f in dataclasses.fields(dc):
        if dataclasses.is_dataclass(f.type) or f.name in ("model", "caps"):
            continue
        name = f"--{prefix}{f.name}"
        default = getattr(dc, f.name) if not isinstance(dc, type) else f.default
        if isinstance(default, bool):
            kw = dict(nargs="?", const=True) if f.name == "async_update" else {}
            ap.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=None, **kw)
        else:
            ap.add_argument(name, type=float if default is None else type(default), default=None)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="PPO training for StS2")
    ap.add_argument("--preset", default="")
    _add_fields(ap, Config())
    _add_fields(ap, ModelCfg(), "model.")
    _add_fields(ap, Caps(), "caps.")
    a = vars(ap.parse_args(argv))
    d = Config().to_dict()
    for p in filter(None, (a.pop("preset") or "").split(",")):
        pd = PRESETS[p]
        for k, v in pd.items():
            if isinstance(v, dict):
                d[k].update(v)
            else:
                d[k] = v
    for k, v in a.items():
        if v is None:
            continue
        if "." in k:
            sub, kk = k.split(".", 1)
            d[sub][kk] = v
        else:
            d[k] = v
    return Config.from_dict(d)


def save_config(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=1)
