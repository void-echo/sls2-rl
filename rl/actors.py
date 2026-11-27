"""Actor processes: each worker owns several engines, steps them concurrently (non-blocking IO, select),
parses + encodes responses into a shared-memory observation slot per env, and exchanges small event
messages with the learner over a pipe.

Worker -> learner events (a list per message):
    ("res", env, reward, done, trunc, has_obs)  result of the last action; if has_obs, slot `env` holds the next obs
    ("obs", env)                                 first obs of a new episode is in slot `env`
    ("ep_start", env)                            new training episode, when exploration is enabled
    ("group_start", env)                         eligible room entry; sampling starts on its first action
    ("group_end", env, win, repeat, terminal)     trial ended; nonterminal ends wait for a value-only reply
    ("group_cancel", env)                        discard an incomplete group after an engine failure
    ("branch_offer", env, options, keys)          first rest-site decision at a room-entry save
    ("branch_trial", env, action, main)           restored entry; execute the supplied first action
    ("branch_end", env, main, terminal)           trial ended; nonterminal ends wait for a value-only reply
    ("branch_cancel", env[, recover])            discard an incomplete comparison; optionally restore main
    ("ep", {...})                                episode summary
    ("err", env, text)                           engine error / restart (for logging)
Learner -> worker:  ("act", [(env, cand_index[, allow_replay]), ...]) | ("value", [env, ...]) |
                   ("branch", [(env, main_action), ...]) | ("asc_frontier", k) | ("stop",)

`trunc` marks a transition whose outcome is unknown (engine error, crash, timeout, step limit): the learner
excludes it from the loss and bootstraps the preceding transitions from its value estimate.
Engine starts/resets are asynchronous, so a slow `rl_start` never blocks the other engines of the worker.
"""
import multiprocessing as mp
from collections import deque
import os
import random
import time
import traceback
from multiprocessing.connection import wait

import numpy as np

from .encode import Caps, Encoder, field_spec, k2
from .engine import REPO, EngineDead, make_engine
from .force import force_matches, parse_force_list


TASTE_CARD_KINDS = {"card_reward": "reward_card", "shop": "buy_card"}


def _round8(n):
    """Round sequence lengths up to a multiple of 8 (aligned attention masks, fewer distinct shapes)."""
    return max(8, (n + 7) // 8 * 8)


class SharedObs:
    """One memory-mapped array per encoded field, shape [n_envs, ...]; created by the learner."""

    def __init__(self, caps, n_envs, path=None, create=True, force=False, taste=False):
        self.caps, self.n = caps, n_envs
        self.spec = field_spec(caps)
        if force:
            self.spec["cand_force"] = ((caps.max_cand,), np.bool_)
        if taste:
            self.spec["cand_card_rows"] = ((caps.max_cand, 2), np.int32)
        if path is None:
            base = "/dev/shm" if os.path.isdir("/dev/shm") else os.path.join(REPO, "tmp", "shm")
            os.makedirs(base, exist_ok=True)
            path = os.path.join(base, f"sls2rl_{os.getpid()}_{random.getrandbits(32):08x}")
        self.path = path
        self.arrays = {}
        for k, (shape, dt) in self.spec.items():
            fn = f"{path}_{k}.bin"
            mode = "w+" if create else "r+"
            self.arrays[k] = np.memmap(fn, dtype=dt, mode=mode, shape=(n_envs,) + shape)
        self.owner = create

    def slot(self, e):
        return {k: a[e, ...] for k, a in self.arrays.items()}  # [e, ...] keeps 0-d views writable

    def gather(self, envs):
        """Copy the obs of `envs`, sliced to the largest token / candidate count among them."""
        envs = np.asarray(envs)
        a = self.arrays
        T = min(_round8(int(a["n_tok"][envs].max())), self.caps.max_tok)
        C = min(_round8(int(a["n_cand"][envs].max())), self.caps.max_cand)
        out = {}
        for k, arr in a.items():
            if k.startswith("tok_"):
                out[k] = arr[envs, :T]
            elif k.startswith("cand_"):
                out[k] = arr[envs, :C]
            else:
                out[k] = arr[envs]
        return out

    def close(self):
        for k, a in self.arrays.items():
            fn = a.filename
            del a
            if self.owner:
                try:
                    os.unlink(fn)
                except OSError:
                    pass
        self.arrays = {}


class SeedSource:
    """Training: endless fresh seeds. Evaluation: a fixed list shared by all workers (mp counter)."""

    def __init__(self, prefix, seed_list=None, counter=None):
        self.prefix, self.list, self.counter, self.n = prefix, seed_list, counter, 0

    def next(self):
        if self.list is None:
            self.n += 1
            return f"{self.prefix}_{self.n}"
        with self.counter.get_lock():
            i = self.counter.value
            if i >= len(self.list):
                return None
            self.counter.value = i + 1
        return self.list[i]


def progress_reward(cfg, ep, info, done):
    """Reward of one transition before the terminal win/loss term; advances ep["floor"] and ep["act"]."""
    floor = info.get("total_floor") or ep["floor"]
    new_floor = max(ep["floor"], floor)
    c = ep.get("floor_c", cfg["floor_reward"])
    if ep.get("shaping") == "potential":
        # Potential-based shaping (Ng et al. 1999) with potential c*floor and 0 at termination: the per-step
        # signal is the same as before, but it telescopes to zero over an episode, so the optimal policy is
        # that of the win/loss reward alone (no incentive to merely survive more floors).
        r = c * ((0.0 if done else cfg["gamma"] * new_floor) - ep["floor"])
    else:
        r = c * (new_floor - ep["floor"])
    # A new act is entered only after its predecessor's boss fell (the last act's boss ends the run as a win).
    act = info.get("act") or ep["act"]
    if ep["act"] and act > ep["act"]:
        r += ep.get("boss_c", cfg.get("boss_reward", 0.0)) * (act - ep["act"])
    ep["floor"] = new_floor
    ep["act"] = max(ep["act"], act)
    return r


def worker_main(wid, env_ids, conn, shm_path, cfg, env_steps, seed_list=None, counter=None):
    try:
        _Worker(wid, env_ids, conn, shm_path, cfg, env_steps, seed_list, counter).run()
    except Exception:
        try:
            conn.send([("fatal", wid, traceback.format_exc())])
        except Exception:
            pass
        raise


class _Worker:
    midrun_on = False  # tests build workers without __init__
    force_on = False
    taste_on = False

    BOOT, RESET, ACT, IDLE, HALT, LOAD, VALUE = range(7)

    def __init__(self, wid, env_ids, conn, shm_path, cfg, env_steps, seed_list, counter):
        self.cfg, self.conn, self.ids = cfg, conn, list(env_ids)
        self.env_steps = env_steps
        caps = Caps.from_dict(cfg["caps"])
        self.force_ids = parse_force_list(cfg.get("force_ids", ""))
        self.force_vars = parse_force_list(cfg.get("force_vars", ""))
        self.force_on = cfg.get("force_frac", 0.0) > 0 and bool(self.force_ids or self.force_vars)
        self.force_subset = cfg.get("force_subset", 0.0) if self.force_on else 0.0
        self.force_active = {}  # env -> the ids forced in its current game (force_subset > 0)
        self.taste_on = cfg.get("taste_frac", 0.0) > 0 and cfg.get("taste_beta", 0.0) > 0
        self.shm = SharedObs(caps, cfg["n_envs"], path=shm_path, create=False, force=self.force_on, taste=self.taste_on)
        self.enc = Encoder(caps)
        self.rng = random.Random(f"{cfg['seed']}-{wid}")
        if cfg.get("asc_curriculum"):
            self.asc_rng = random.Random(f"asc-{cfg['seed']}-{wid}")
        self.chars = [c.strip() for c in cfg["characters"].split(",") if c.strip()]
        log_dir = os.path.join(cfg["run_dir"], "engine_logs") if cfg.get("engine_log") else None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        cmd = None
        if cfg.get("engine_cmd"):
            import json
            cmd = json.loads(cfg["engine_cmd"])
        self.eng, self.state, self.deadline, self.ep = {}, {}, {}, {}
        prefix = "eval" if seed_list is not None else f"s{cfg['seed']}{cfg.get('seed_nonce', '')}"
        self.seeds = {e: SeedSource(f"{prefix}_e{e}", seed_list, counter) for e in self.ids}
        self.fails = {e: 0 for e in self.ids}
        self.runs_in_proc = {e: 0 for e in self.ids}  # full games finished by the current engine process
        self.last = {}  # env -> (decision, chosen candidate, ctx) of the last action, for error reports
        self.groups, self.seen_save = {}, {}
        self.branches, self.branch_offers = {}, {}
        self.branch_decisions = {s.strip() for s in cfg.get("branch_decisions", "").split(",") if s.strip()}
        self.branch_rng = random.Random(f"branch-{cfg['seed']}-{wid}")
        self.group_rooms = {s.strip() for s in cfg.get("group_rooms", "Elite,Boss").split(",")}
        self.midrun_on = cfg.get("midrun_frac", 0.0) > 0 and seed_list is None
        self.midrun_rooms = {s.strip() for s in cfg.get("midrun_rooms", "Boss").split(",")}
        self.midrun_saves = deque(maxlen=cfg.get("midrun_pool", 128))
        self.midrun_rng = random.Random(f"midrun-{cfg['seed']}-{wid}")
        self.midrun_seen, self.midrun_wait, self.midrun_next, self.midrun_loads = {}, {}, {}, {}
        for e in self.ids:
            self.eng[e] = make_engine(cfg["engine"], engine_dir=cfg.get("engine_dir"), engine_cmd=cmd,
                                      log_path=os.path.join(log_dir, f"engine_{e}.log") if log_dir else None,
                                      name=f"engine{e}", fake_p_error=cfg.get("fake_p_error", 0.0),
                                      fake_args=(cfg.get("fake_args") or "").split())
        self.events = []

    # ------------------------------------------------------------ engine state machine
    def boot(self, e):
        try:
            self.eng[e].start()
        except Exception as ex:  # e.g. missing binary: retry with backoff, report
            self.fail(e, f"start failed: {ex}")
            return
        self.state[e] = self.BOOT
        self.runs_in_proc[e] = 0
        if self.midrun_on:
            self.midrun_loads[e] = 0
        self.deadline[e] = time.monotonic() + self.cfg["timeout_start"]

    def midrun_choice(self, e):
        """Does the next episode of env e start from a saved room entry? Decided once per episode."""
        if not self.midrun_on:
            return False
        if e not in self.midrun_next:
            self.midrun_next[e] = bool(self.midrun_saves) and self.midrun_rng.random() < self.cfg["midrun_frac"]
        return self.midrun_next[e]

    def draw_force_subset(self, e):
        self.force_active[e] = {i for i in sorted(self.force_ids) if self.rng.random() < self.force_subset}

    def new_episode(self, e):
        self.seen_save.pop(e, None)
        midrun = self.midrun_choice(e)
        if self.midrun_on:
            self.midrun_seen.pop(e, None)
            self.midrun_wait.pop(e, None)
            del self.midrun_next[e]
        if self.cfg.get("branch_decisions"):
            self.branch_offers.pop(e, None)
        seed = self.seeds[e].next()
        if seed is None:
            self.state[e] = self.HALT
            self.eng[e].close()
            return
        if isinstance(seed, (tuple, list)):  # evaluation lists carry (character, seed)
            char, seed = seed
        else:
            char = self.rng.choice(self.chars)
        self.ep[e] = {"env": e, "seed": seed, "char": char, "ret": 0.0, "steps": 0, "floor": 0, "act": 0,
                      "shaping": self.episode_shaping(), "floor_c": self.episode_floor_reward() * self.shaping_fade()}
        if self.cfg.get("shaping_fade_steps", 0) > 0:
            self.ep[e]["boss_c"] = self.cfg.get("boss_reward", 0.0) * self.shaping_fade()
        if getattr(self, "force_subset", 0.0) > 0:
            # one subset per game: within a game forcing is deterministic, so a draw at the rewards screen and another
            # at the pick cannot be dodged by leaving card rewards unopened (seen with force_prob 0.2, 11-20)
            self.draw_force_subset(e)
        if self.cfg.get("explore_eps", 0.0) > 0 or self.force_on or self.taste_on:
            self.events.append(("ep_start", e))
        asc = self.episode_ascension()
        if self.cfg.get("asc_curriculum"):
            self.ep[e]["asc"] = asc
        if midrun:
            snap = self.midrun_rng.choice(self.midrun_saves)
            self.ep[e].update(midrun=True, seed=snap["seed"], char=snap["char"], start_floor=snap["floor"])
            if "asc" in snap:
                self.ep[e]["asc"] = snap["asc"]
            self.midrun_loads[e] = self.midrun_loads.get(e, 0) + 1
            self.eng[e].send({"cmd": "rl_load", "data": snap["data"]})
        else:
            self.eng[e].send({"cmd": "rl_start", "character": char, "ascension": asc, "seed": seed})
        self.state[e] = self.RESET
        self.deadline[e] = time.monotonic() + self.cfg["timeout_start"]

    def episode_ascension(self):
        if not self.cfg.get("asc_curriculum"):
            return self.cfg["ascension"]
        k = self.asc_frontier
        return k if k == 0 or self.asc_rng.random() < self.cfg["asc_frontier_frac"] else self.asc_rng.randrange(k)

    def episode_shaping(self):
        """Shaping mode for a new episode. "switch" bootstraps with per-floor reward (a curriculum that gets the
        policy to occasional wins) and moves to potential-based shaping, whose objective is winning alone, once the
        global step count passes shaping_switch_steps. Decided per episode so no episode mixes the two.
        The learner publishes this shared count once per update, so it stays fixed within an update."""
        mode = self.cfg["shaping"]
        if mode == "switch":
            return "potential" if self.env_steps.value >= self.cfg["shaping_switch_steps"] else "per_floor"
        return mode

    def episode_floor_reward(self):
        """Per-floor reward of a new episode. With floor_anneal_steps it falls linearly to 0 over that many
        global env steps: floors guide an untrained policy, then bosses and wins alone count."""
        c, n = self.cfg["floor_reward"], self.cfg.get("floor_anneal_steps", 0)
        if n <= 0:
            return c
        return c * max(0.0, 1.0 - self.env_steps.value / n)

    def shaping_fade(self):
        """Factor on the floor and boss rewards of a new episode: 1 until shaping_fade_start, then linear down to
        shaping_fade_to over shaping_fade_steps global env steps. Decided per episode, so no episode mixes two scales."""
        n = self.cfg.get("shaping_fade_steps", 0)
        if n <= 0:
            return 1.0
        frac = min(max((self.env_steps.value - self.cfg.get("shaping_fade_start", 0)) / n, 0.0), 1.0)
        return 1.0 + (self.cfg.get("shaping_fade_to", 0.0) - 1.0) * frac

    def fail(self, e, why):
        """Engine error/crash/timeout: abort the episode, report, restart the process."""
        last = self.last.pop(e, None)
        if last is not None and self.state.get(e) == self.ACT:
            # Context first: engine messages carry long stack traces that would push it out of the log line.
            why = f"after {last[0]} chose {last[1]} ctx={last[2]} | {why}"
        self.events.append(("err", e, why[:4000]))
        if self.cfg.get("branch_decisions") and e in self.branches:
            b = self.branches[e]
            if not b["main"]:
                if self.state.get(e) == self.ACT:
                    self.events.append(("res", e, 0.0, False, True, False))
                self.events.append(("branch_cancel", e, True))
                b["trial"] = len(b["actions"]) - 2
                try:
                    self.next_branch(e)
                except EngineDead as ex:
                    self.fail(e, str(ex))
                return
            self.events.append(("branch_cancel", e, False))
            del self.branches[e]
        if self.cfg.get("group_k", 1) > 1 and self.groups.pop(e, None) is not None:
            self.events.append(("group_cancel", e))
        if self.state.get(e) == self.ACT:
            self.events.append(("res", e, 0.0, False, True, False))
            self.finish(e, victory=None, aborted=why[:100])
        elif self.state.get(e) in (self.RESET, self.LOAD):
            # The run never started; still report it so fixed-seed evaluation accounts for every seed.
            self.finish(e, victory=None, aborted=why[:100])
        self.state[e] = None
        self.fails[e] += 1
        if self.fails[e] > 20:
            raise RuntimeError(f"engine {e} failed 20 times in a row; last: {why}")
        if self.fails[e] > 3:
            time.sleep(min(30.0, 0.5 * 2 ** (self.fails[e] - 3)))
        self.eng[e].close()
        self.boot(e)

    def finish(self, e, victory, aborted=None):
        if self.cfg.get("branch_decisions") and self.branches.pop(e, None) is not None:
            self.events.append(("branch_cancel", e))
        if self.cfg.get("group_k", 1) > 1 and self.groups.pop(e, None) is not None:
            self.events.append(("group_cancel", e))
        ep = self.ep.pop(e, None)
        if ep is not None:
            ep.update(victory=victory, aborted=aborted)
            self.events.append(("ep", ep))

    def handle(self, e, R):
        if R.get("type") == "save":  # reply to a mid-run capture; the state machine is not waiting for it
            wait = self.midrun_wait.pop(e, None)
            if wait is not None and R.get("kind") == "room" and R.get("data"):
                self.midrun_saves.append({**wait, "data": R["data"]})
            return
        st = self.state[e]
        if st == self.BOOT:
            if R.get("type") == "ready":
                self.new_episode(e)
            return
        if R.get("type") != "rl":
            return self.fail(e, f"engine error: {R.get('message', R)}")
        if st == self.RESET:
            if R.get("done") or not R.get("cands"):
                return self.fail(e, "rl_start returned a finished run / no candidates")
            self.ep[e]["floor"] = (R.get("info") or {}).get("total_floor", 0) or 0
            self.ep[e]["act"] = (R.get("info") or {}).get("act", 0) or 0
            self.encode(e, R)
            self.events.append(("obs", e))
            self.fails[e] = 0
            self.state[e] = self.IDLE
            return
        if st == self.LOAD:
            if R.get("done") or not R.get("cands"):
                return self.fail(e, "rl_load returned a finished run / no candidates")
            self.encode(e, R)
            if self.cfg.get("branch_decisions") and e in self.branches and e not in self.groups:
                b = self.branches[e]
                self.events.append(("branch_trial", e, b["actions"][b["trial"]], b["main"]))
            self.events.append(("obs", e))
            self.state[e] = self.IDLE
            return
        # st == ACT
        ep = self.ep[e]
        done = bool(R.get("done"))
        r = progress_reward(self.cfg, ep, R.get("info") or {}, done)
        ep["steps"] += 1
        if done:
            win = bool(R.get("victory"))
            r += self.cfg["win_reward"] if win else self.cfg["loss_reward"]
            ep["ret"] += r
            self.events.append(("res", e, r, True, False, False))
            if self.cfg.get("group_k", 1) > 1 and e in self.groups:
                if self.end_trial(e, win, terminal=True):
                    return
            if self.cfg.get("branch_decisions") and e in self.branches:
                if self.end_branch(e, terminal=True):
                    return
            self.finish(e, victory=win)
            self.state[e] = None  # result already reported; a failure from here on must not report it again
            return self.next_run(e)
        ep["ret"] += r
        if not R.get("cands"):
            return self.fail(e, "not done but no candidates")
        # Safety valves only (see config): long legitimate infinites must fit far below these.
        combat = (R.get("obs") or {}).get("combat")
        if combat is None:
            ep["cstep"], ep["tstep"], ep["round"] = 0, 0, None
        else:
            ep["cstep"] = ep.get("cstep", 0) + 1
            if combat.get("round") != ep.get("round"):
                ep["round"], ep["tstep"] = combat.get("round"), 0
            ep["tstep"] = ep.get("tstep", 0) + 1
        cap = None
        if self.cfg["max_episode_steps"] and ep["steps"] >= self.cfg["max_episode_steps"]:
            cap = "episode_step_limit"
        elif ep.get("cstep", 0) >= self.cfg["max_combat_steps"]:
            cap = "combat_step_limit"
        elif ep.get("tstep", 0) >= self.cfg["max_turn_steps"]:
            cap = "turn_step_limit"
        if cap:
            if self.cfg.get("branch_decisions") and e in self.branches and not self.branches[e]["main"]:
                return self.fail(e, cap)
            self.events.append(("err", e, f"{cap}: steps={ep['steps']} combat_steps={ep.get('cstep')} "
                                          f"turn_steps={ep.get('tstep')} info={R.get('info')}"))
            self.events.append(("res", e, r, False, True, False))
            self.finish(e, victory=None, aborted=cap)
            self.state[e] = None
            return self.next_run(e)
        g = self.groups.get(e) if self.cfg.get("group_k", 1) > 1 else None
        main_path = g is None or g["last"] or g["trial"] + 1 >= self.cfg["group_k"]
        self.encode(e, R)
        if self.cfg.get("group_k", 1) > 1 and e in self.groups and combat is None:
            self.events.append(("res", e, r, False, False, False))
            repeat = self.end_trial(e, True, terminal=False)
            if not repeat and self.cfg.get("branch_decisions") and e in self.branches:
                self.groups[e]["branch_end"] = self.branch_endpoint(e, combat)
            return
        if (self.cfg.get("branch_decisions") and e in self.branches and main_path
                and self.branch_endpoint(e, combat)):
            self.events.append(("res", e, r, False, False, False))
            self.end_branch(e, terminal=False)
            return
        self.events.append(("res", e, r, False, False, True))
        self.state[e] = self.IDLE

    def end_trial(self, e, win, terminal):
        g = self.groups[e]
        g["trial"] += 1
        repeat = g["trial"] < self.cfg["group_k"] and not g["last"]
        self.events.append(("group_end", e, win, repeat, terminal))
        if terminal:
            if repeat:
                self.load_group(e)
            else:
                del self.groups[e]
        else:
            g["repeat"] = repeat
            self.state[e] = self.VALUE
        return repeat

    def load_group(self, e):
        g = self.groups[e]
        self.load_room(e, g)

    def load_room(self, e, g):
        self.ep[e] = g["ep"].copy()
        self.state[e] = self.LOAD
        self.deadline[e] = time.monotonic() + self.cfg["timeout_start"]
        self.eng[e].send({"cmd": "rl_load", "id": g["id"]})

    def value_done(self, e):
        if e in self.groups and "repeat" in self.groups[e]:
            g = self.groups[e]
            if g.pop("repeat"):
                self.load_group(e)
            else:
                del self.groups[e]
                if g.get("branch_end"):
                    self.end_branch(e, terminal=False)
                else:
                    self.state[e] = self.IDLE
                    self.events.append(("obs", e))
            return
        if self.cfg.get("branch_decisions") and e in self.branches:
            b = self.branches[e]
            if b["main"]:
                del self.branches[e]
                self.state[e] = self.IDLE
                self.events.append(("obs", e))
            else:
                self.next_branch(e)
            return

    def start_branch(self, e, action):
        b = self.branch_offers.pop(e)
        b.update(actions=[j for j in b["options"] if j != action] + [action], trial=-1)
        self.branches[e] = b
        self.next_branch(e)

    def next_branch(self, e):
        b = self.branches[e]
        b["trial"] += 1
        b["main"] = b["trial"] == len(b["actions"]) - 1
        b["boss"] = False
        self.load_room(e, b)

    def end_branch(self, e, terminal):
        b = self.branches[e]
        self.events.append(("branch_end", e, b["main"], terminal))
        if terminal:
            if not b["main"]:
                self.next_branch(e)
                return True
            del self.branches[e]
        else:
            self.state[e] = self.VALUE
        return False

    def branch_endpoint(self, e, combat):
        b, ep = self.branches[e], self.ep[e]
        return (ep["steps"] - b["ep"]["steps"] >= self.cfg["branch_max_steps"]
                or b["boss"] and combat is None or ep["act"] > b["ep"]["act"])

    def next_run(self, e):
        # A load replaces the whole run, so consecutive mid-run starts can share a process. Full games start in a fresh process
        # every restart_every_n-th game (rl_start in a used process is step-identical to a fresh one, engine/RL_PROTOCOL.md).
        reuse = self.midrun_choice(e) and self.midrun_loads.get(e, 0) < self.cfg.get("midrun_reuse", 16)
        if self.cfg["restart_every_episode"] and not reuse:
            self.runs_in_proc[e] += 1
            if self.runs_in_proc[e] >= self.cfg.get("restart_every_n", 1):
                self.eng[e].close()
                self.boot(e)
                return
        self.new_episode(e)

    def encode(self, e, R):
        slot = self.shm.slot(e)
        if self.taste_on:
            card_rows = slot.pop("cand_card_rows")
        self.enc.encode(R, slot)
        if self.force_on:
            ids = self.force_active.get(e, self.force_ids) if getattr(self, "force_subset", 0.0) > 0 else self.force_ids
            matches = force_matches(R, ids, self.force_vars)[:self.enc.c.max_cand]
            self.shm.arrays["cand_force"][e, :len(matches)] = matches
        if self.taste_on:
            card_rows.fill(-1)
            card_kind = TASTE_CARD_KINDS.get(R.get("decision"))
            if card_kind is not None:
                for j, cand in enumerate((R.get("cands") or [])[:int(slot["n_cand"])]):
                    if cand.get("k") == card_kind:
                        card_id = (cand.get("card") or {}).get("id")
                        if card_id:
                            key = k2("card.id", card_id)
                            card_rows[j] = (key >> 16) & 0xFFFF, key & 0xFFFF
        self.last[e] = (R.get("decision"), R.get("cands"), (R.get("obs") or {}).get("ctx"))
        if self.midrun_on:
            self.midrun_capture(e, R)
        branching = bool(self.cfg.get("branch_decisions"))
        if branching and e in self.branches:
            if (R.get("obs") or {}).get("combat") is not None:
                room = (R.get("obs") or {}).get("g", {}).get("room")
                save = (R.get("info") or {}).get("save") or {}
                self.branches[e]["boss"] |= room == "Boss" or save.get("room") == "Boss"
        if self.cfg.get("group_k", 1) > 1 or branching:
            save = (R.get("info") or {}).get("save")
            if save and save.get("kind") == "room" and save["id"] != self.seen_save.get(e):
                self.seen_save[e] = save["id"]
                if branching and e in self.branches and not self.branches[e]["main"]:
                    return
                if (self.cfg.get("group_k", 1) > 1 and e not in self.groups and save.get("room") in self.group_rooms
                        and (R.get("obs") or {}).get("combat") is not None):
                    self.groups[e] = dict(id=save["id"], ep=self.ep[e].copy(), trial=0, last=False)
                    self.events.append(("group_start", e))
                if (branching and e not in self.branches and e not in self.groups and save.get("room") in ("RestSite", "Rest")
                        and R.get("decision") == "rest_site" and "rest_site" in self.branch_decisions):
                    options = [i for i, c in enumerate(R["cands"][:self.enc.c.max_cand]) if c.get("k") == "rest"]
                    if len(options) >= 2 and self.branch_rng.random() < self.cfg["branch_prob"]:
                        self.branch_offers[e] = dict(id=save["id"], ep=self.ep[e].copy(), options=options)
                        self.events.append(("branch_offer", e, options, [R["cands"][i]["key"] for i in options]))

    def midrun_capture(self, e, R):
        """Ask for the data of a new room-entry save from a full game at the frontier level. The reply comes before the reply
        to the next action (the engine answers in order), and handle() takes it without touching the state machine."""
        ep = self.ep.get(e)
        info = R.get("info") or {}
        save = info.get("save")
        if (not ep or ep.get("midrun") or not save or save.get("kind") != "room" or save["id"] == self.midrun_seen.get(e)
                or save.get("room") not in self.midrun_rooms or (info.get("total_floor") or 0) < self.cfg.get("midrun_min_floor", 0)):
            return
        if self.cfg.get("asc_curriculum") and ep.get("asc") != self.asc_frontier:
            return
        self.midrun_seen[e] = save["id"]
        self.midrun_wait[e] = dict(seed=ep["seed"], char=ep["char"], floor=info.get("total_floor") or 0,
                                   **({"asc": ep["asc"]} if "asc" in ep else {}))
        self.eng[e].send({"cmd": "rl_save_get", "id": save["id"]})

    # ------------------------------------------------------------ main loop
    def run(self):
        if self.cfg.get("asc_curriculum"):
            _, self.asc_frontier = self.conn.recv()
        for e in self.ids:
            self.boot(e)
        last_trunc_report = time.monotonic()
        while True:
            busy = [e for e in self.ids if self.state[e] in (self.BOOT, self.RESET, self.ACT, self.LOAD)]
            instant = any(self.eng[e].fileno() is None or self.eng[e].pending_ok() for e in busy)
            waitables = [self.conn] + [self.eng[e] for e in busy if self.eng[e].fileno() is not None]
            ready = wait(waitables, timeout=0 if instant else (0.05 if busy else 1.0))
            if self.conn in ready or self.conn.poll():
                while self.conn.poll():
                    msg = self.conn.recv()
                    if msg[0] == "stop":
                        return self.shutdown()
                    if msg[0] == "asc_frontier":
                        self.asc_frontier = msg[1]
                        continue
                    if msg[0] == "value":
                        for e in msg[1]:
                            try:
                                self.value_done(e)
                            except EngineDead as ex:
                                self.fail(e, str(ex))
                        continue
                    if msg[0] == "branch":
                        for e, action in msg[1]:
                            try:
                                self.start_branch(e, action)
                            except EngineDead as ex:
                                self.fail(e, str(ex))
                        continue
                    for pair in msg[1]:
                        e, a = pair[:2]
                        if self.cfg.get("branch_decisions"):
                            self.branch_offers.pop(e, None)
                        if len(pair) > 2 and e in self.groups and not pair[2]:
                            self.groups[e]["last"] = True
                        if self.state[e] != self.IDLE:  # should not happen; unblock the learner
                            self.events.append(("err", e, f"action for env in state {self.state[e]}"))
                            self.events.append(("res", e, 0.0, False, True, False))
                            continue
                        if e in self.last:
                            d, cands, ctx = self.last[e]
                            self.last[e] = (d, cands[int(a)] if 0 <= int(a) < len(cands) else a, str(ctx)[:200])
                        if self.cfg.get("log_actions") and e in self.ep:
                            self.ep[e].setdefault("actions", []).append(int(a))
                        try:
                            self.eng[e].send({"cmd": "rl_act", "i": int(a)})
                            self.state[e] = self.ACT
                            self.deadline[e] = time.monotonic() + self.cfg["timeout_step"]
                        except EngineDead as ex:
                            self.state[e] = self.ACT
                            self.fail(e, str(ex))
            now = time.monotonic()
            for e in busy:
                if self.state[e] not in (self.BOOT, self.RESET, self.ACT, self.LOAD):
                    continue
                try:
                    R = self.eng[e].poll()
                    while R is not None:
                        self.handle(e, R)
                        if self.state[e] not in (self.BOOT, self.RESET, self.ACT, self.LOAD):
                            break
                        R = self.eng[e].poll()
                    if self.state[e] in (self.BOOT, self.RESET, self.ACT, self.LOAD) and now > self.deadline[e]:
                        self.fail(e, f"timeout in state {self.state[e]}")
                except EngineDead as ex:
                    self.fail(e, str(ex))
                except Exception:
                    self.fail(e, "worker exception: " + traceback.format_exc(limit=3))
            if now - last_trunc_report > 60 and self.enc.trunc:
                self.events.append(("trunc", dict(self.enc.trunc)))
                self.enc.trunc.clear()
                last_trunc_report = now
            if self.events:
                self.conn.send(self.events)
                self.events = []

    def shutdown(self):
        for e in self.ids:
            self.eng[e].close()


class ActorPool:
    """Learner-side handle: starts workers, receives events, sends actions, reads obs from shared memory."""

    def __init__(self, cfg, seed_list=None, resume_steps=0, asc_frontier=None):
        self.cfg = cfg
        self.n_envs = cfg.n_envs
        self.shm = SharedObs(cfg.caps, self.n_envs, force=cfg.force_enabled, taste=cfg.taste_enabled)
        ctx = mp.get_context("spawn")
        d = cfg.to_dict()
        d["n_envs"] = self.n_envs
        d["run_dir"] = cfg.resolved_run_dir()
        d["seed_nonce"] = f"x{random.getrandbits(32):08x}"  # fresh training seeds after every (re)start
        self.env_steps = ctx.Value("q", resume_steps)
        self.counter = counter = ctx.Value("i", 0) if seed_list is not None else None  # keep alive for children
        self.conns, self.procs, self.env_worker = [], [], {}
        for w in range(cfg.n_workers):
            ids = list(range(w * cfg.envs_per_worker, (w + 1) * cfg.envs_per_worker))
            a, b = ctx.Pipe(duplex=True)
            p = ctx.Process(target=worker_main, args=(w, ids, b, self.shm.path, d, self.env_steps, seed_list, counter), daemon=True)
            p.start()
            b.close()
            self.conns.append(a)
            self.procs.append(p)
            for e in ids:
                self.env_worker[e] = w
        if cfg.asc_curriculum:
            self.set_asc_frontier(cfg.asc_start if asc_frontier is None else asc_frontier)

    def set_asc_frontier(self, k):
        self.asc_frontier = k
        for c in self.conns:
            c.send(("asc_frontier", k))

    def poll(self, timeout):
        """Return all pending events (waits up to `timeout` seconds for the first one)."""
        out = []
        ready = wait(self.conns, timeout=timeout)
        for c in ready:
            while c.poll():
                try:
                    out.extend(c.recv())
                except EOFError:
                    raise RuntimeError("an actor worker died; see its traceback above")
        for ev in out:
            if ev[0] == "fatal":
                raise RuntimeError(f"actor worker {ev[1]} crashed:\n{ev[2]}")
        return out

    def act(self, envs, actions, allow_replay=None):
        per = {}
        for e, a in zip(envs, actions):
            pair = (int(e), int(a)) if allow_replay is None else (int(e), int(a), allow_replay)
            per.setdefault(self.env_worker[int(e)], []).append(pair)
        for w, pairs in per.items():
            self.conns[w].send(("act", pairs))

    def value_done(self, envs):
        per = {}
        for e in envs:
            per.setdefault(self.env_worker[e], []).append(e)
        for w, ids in per.items():
            self.conns[w].send(("value", ids))

    def start_branches(self, pairs):
        per = {}
        for e, action in pairs:
            per.setdefault(self.env_worker[e], []).append((e, int(action)))
        for w, actions in per.items():
            self.conns[w].send(("branch", actions))

    def close(self):
        for c in self.conns:
            try:
                c.send(("stop",))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self.shm.close()
