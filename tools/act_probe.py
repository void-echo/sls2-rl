"""Does the policy use the act-1 variant? For seeds a10fork2_0..N-1 (the death_fork games), play greedily to the first
act-1 map decision, then re-encode that observation with g.act_id (and separately g.boss) swapped to the other variant's
value and compare value and map-choice policy. Also joins act_id with the death_fork outcome of the same seed.
   python tmp/fresh1111/act_probe.py <ckpt> <engine_dir> <death_fork runs.jsonl> <n_seeds> <out.json>"""
import copy, json, sys, collections
import numpy as np, torch
sys.path.insert(0, '.')
from rl.engine import EngineProcess
from rl.encode import Encoder
from rl.ppo import to_torch
from rl.train import load_policy

ckpt, edir, runs_path, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
model, vn, cfg, _ = load_policy(ckpt, 'cpu'); model.eval()
enc = Encoder(cfg.caps)
def run(Rs):
    obs = [enc.encode(R) for R in Rs]
    b = to_torch({k: np.stack([o[k] for o in obs]) for k in obs[0]}, 'cpu')
    with torch.no_grad():
        lg, v = model(b)
    return torch.log_softmax(lg.float(), -1), vn.denormalize(v.float())
eng = EngineProcess(engine_dir=edir)
rows = []
for i in range(n):
    seed = f'a10fork2_{i}'
    R = eng.reset('Ironclad', 10, seed)
    start = copy.deepcopy(R)
    for t in range(200):
        g = (R.get('obs') or {}).get('g') or {}
        if R.get('decision') == 'map_select' and g.get('act') == 1:
            break
        lp, _ = run([R]); R = eng.step(int(lp[0].argmax()))
    g = R['obs']['g']
    rows.append(dict(seed=seed, act_id=g.get('act_id'), boss=g.get('boss'), R=R, start=start))
eng.close()
acts = sorted({r['act_id'] for r in rows}); bosses = collections.defaultdict(set)
for r in rows: bosses[r['act_id']].add(r['boss'])
print('act ids', acts, {a: sorted(b) for a, b in bosses.items()}, flush=True)
other = {acts[0]: acts[1], acts[1]: acts[0]} if len(acts) == 2 else {}
res = []
for r in rows:
    if r['act_id'] not in other: continue
    o = other[r['act_id']]
    for key, R0 in (('map', r['R']), ('start', r['start'])):
        Ra = copy.deepcopy(R0); Ra['obs']['g']['act_id'] = o
        Rb = copy.deepcopy(R0); Rb['obs']['g']['boss'] = sorted(bosses[o])[0]
        Rc = copy.deepcopy(Ra); Rc['obs']['g']['boss'] = sorted(bosses[o])[0]
        lp, v = run([R0, Ra, Rb, Rc]); p = lp.exp()
        tv = lambda j: 0.5 * float((p[0] - p[j]).abs().sum())
        res.append(dict(seed=r['seed'], at=key, act_id=r['act_id'], boss=r['boss'], v=float(v[0]),
                        dv_act=float(v[1] - v[0]), dv_boss=float(v[2] - v[0]), dv_both=float(v[3] - v[0]),
                        tv_act=tv(1), tv_boss=tv(2), tv_both=tv(3), flip_both=int(p[0].argmax() != p[3].argmax()),
                        n_cand=int((lp[0] > -1e8).sum())))
won = {}
for line in open(runs_path):
    d = json.loads(line); won[d['seed']] = bool(d['victory'])
json.dump(res, open(out, 'w'))
for key in ('start', 'map'):
    q = [x for x in res if x['at'] == key]
    print(f'== at {key}: {len(q)} observations')
    for a in acts:
        qa = [x for x in q if x['act_id'] == a]
        w = [won[x['seed']] for x in qa if x['seed'] in won]
        print(f'  {a}: n={len(qa)} V={np.mean([x["v"] for x in qa]):.3f} win(death_fork games)={np.mean(w) if w else float("nan"):.3f} ({len(w)})'
              f' | swap act_id dV={np.mean([x["dv_act"] for x in qa]):+.3f} |dV|={np.mean([abs(x["dv_act"]) for x in qa]):.3f} TV={np.mean([x["tv_act"] for x in qa]):.3f}'
              f' | swap boss dV={np.mean([x["dv_boss"] for x in qa]):+.3f} TV={np.mean([x["tv_boss"] for x in qa]):.3f}'
              f' | both dV={np.mean([x["dv_both"] for x in qa]):+.3f} TV={np.mean([x["tv_both"] for x in qa]):.3f} argmax changes {np.mean([x["flip_both"] for x in qa]):.2f}')
