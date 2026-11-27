"""Act-1 variant and act-1 boss of each seed: python act_of_seed.py <engine_dir> <prefix> <n> <character> <asc> <out.json>"""
import json, sys
sys.path.insert(0, '.')
from rl.engine import EngineProcess
edir, prefix, n, ch, asc, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
eng = EngineProcess(engine_dir=edir); res = {}
for i in range(n):
    if i % 40 == 0:
        eng.close()  # fresh process now and then (memory)
    R = eng.reset(ch, asc, f'{prefix}_{i}'); g = R['obs']['g']
    res[f'{prefix}_{i}'] = [g.get('act_id'), g.get('boss')]
eng.close(); json.dump(res, open(out, 'w')); print('done', len(res))
