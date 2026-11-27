"""Where do games end? Aggregates the per-episode records of the last N evaluations at each ascension.
Run on the server from a run directory: python3 funnel.py [n_evals]"""
import json, glob, collections, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
def load(k):
    eps = []
    for f in sorted(glob.glob(f"{k}/eval_*.json"))[-N:]:
        eps += json.load(open(f))["episodes"]
    return eps
def stage(e):
    if e["victory"]:
        return "WIN"
    f = e["floor"]
    if f <= 16: return "act1 hall (1-16)"
    if f == 17: return "act1 BOSS (17)"
    if f <= 32: return "act2 hall (18-32)"
    if f == 33: return "act2 BOSS (33)"
    if f <= 47: return "act3 hall (34-47)"
    if f == 48: return "act3 BOSS1 (48)"
    return "act3 BOSS2 (49+)"
order = ["act1 hall (1-16)", "act1 BOSS (17)", "act2 hall (18-32)", "act2 BOSS (33)", "act3 hall (34-47)", "act3 BOSS1 (48)", "act3 BOSS2 (49+)", "WIN"]
for k, name in [("eval_a10", "A10"), ("eval_a8", "A8"), ("eval", "A0")]:
    eps = load(k)
    if not eps:
        continue
    c = collections.Counter(stage(e) for e in eps)
    n = len(eps)
    alive = n
    print(f"== {name}: {n} games, mean decisions {sum(e['steps'] for e in eps) / n:.0f}")
    for s in order:
        v = c[s]
        print(f"   {s:20s} {v:4d} {100 * v / n:5.1f}%   of those reaching it: {100 * v / max(alive, 1):5.1f}%")
        alive -= v
