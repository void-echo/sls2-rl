"""Does every exported observation field reach the network input?

For recorded engine responses, change one leaf value of obs / cands at a time (number +1, bool flipped, string
rewritten), re-encode, and report the field paths whose change leaves the encoding identical: the encoder drops
them. Also reports entry-cap truncations at the given caps. Fields that are dropped on purpose are listed in
INTENDED with the reason.

  python3 tools/obs_sensitivity.py tmp/obs_audit/states_current.jsonl --max_ent 36 --max_cent 12 --map_paths --pile_counts
"""
import argparse
import collections
import copy
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.encode import Caps, Encoder, _descendants  # noqa: E402

# Leaf paths (list indices as []) that the encoder ignores on purpose.
INTENDED = {
    "cands[].src": "index of the option inside a selection; the option card itself is inline",
    "obs.ctx.unavailable[].k": "redundant when the item is a card (the card token says it)",
}


def leaves(x, path=()):
    """(path tuple, value) for every non-null leaf; path elements are dict keys (str) or list indices (int)."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    elif x is not None:
        yield path, x


def pattern(path):
    return "".join("[]" if isinstance(p, int) else (f".{p}" if i else p) for i, p in enumerate(path))


def set_path(obj, path, value):
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value


PILES = ("hand", "draw", "discard", "exhaust")


def skipped_by_design(R, p, caps):
    """Leaves the encoder leaves out on purpose, decided per state: the master deck and the map in combat
    (caps.deck_in_combat / map_in_combat), map nodes and edges no longer reachable from the current node, an
    affliction / enchantment amount on a card without one, and items beyond a group cap (counted as truncation)."""
    obs = R.get("obs") or {}
    in_combat = obs.get("combat") is not None
    if in_combat and p[:2] in (("obs", "deck"), ("obs", "map")):
        return True
    if p[-1] in ("aff_n", "ench_n"):
        parent = R
        for x in p[:-1]:
            parent = parent[x]
        if parent.get(p[-1][:-2]) is None:
            return True
    if p[:3] in (("obs", "map", "nodes"), ("obs", "map", "edges")):
        m = obs.get("map") or {}
        nodes, cur = m.get("nodes") or [], m.get("cur", -1)
        if isinstance(cur, int) and 0 <= cur < len(nodes):
            desc = _descendants(len(nodes), m.get("edges") or [])
            keep = desc[cur] | (1 << cur)
            for c in R.get("cands") or []:
                if c.get("k") == "map" and isinstance(c.get("node"), int):
                    keep |= (1 << c["node"]) | desc[c["node"]]
            if p[2] == "nodes" and not keep >> p[3] & 1:
                return True
            if p[2] == "edges" and not keep >> m["edges"][p[3]][0] & 1:  # only edges out of reachable nodes matter
                return True
    if in_combat and len(p) > 3 and p[:2] == ("obs", "combat") and p[2] in PILES and p[3] >= getattr(caps, p[2]):
        return True
    return False


def perturb(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + max(1, abs(v) // 4)  # a step the fp16 encoding can resolve
    return f"{v}_PERTURBED"


def same(a, b):
    return all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states")
    ap.add_argument("--max_ent", type=int, default=24)
    ap.add_argument("--max_cent", type=int, default=8)
    ap.add_argument("--map_paths", action="store_true")
    ap.add_argument("--pile_counts", action="store_true")
    ap.add_argument("--per_pattern", type=int, default=25, help="instances tested per field pattern")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    enc = Encoder(Caps(max_ent=a.max_ent, max_cent=a.max_cent, map_paths=a.map_paths, pile_counts=a.pile_counts))
    Rs = [json.loads(l) for l in open(a.states)]
    for R in Rs:
        R["type"] = "rl"
    rng = random.Random(a.seed)
    # group leaf occurrences by pattern, sample instances per pattern
    occ = collections.defaultdict(list)
    for si, R in enumerate(Rs):
        for root in ("obs", "cands"):
            for p, v in leaves(R[root], (root,)):
                if not skipped_by_design(R, p, enc.c):
                    occ[pattern(p)].append((si, p, v))
    tested, dropped, examples = collections.Counter(), collections.Counter(), {}
    enc.trunc.clear()
    for pat, items in sorted(occ.items()):
        for si, p, v in rng.sample(items, min(len(items), a.per_pattern)):
            R = Rs[si]
            base = enc.encode(R)
            R2 = copy.deepcopy(R)
            set_path(R2, p, perturb(v))
            tested[pat] += 1
            if same(base, enc.encode(R2)):
                dropped[pat] += 1
                examples.setdefault(pat, (pattern(p), v))
    trunc = count_truncations(enc, Rs)
    print(f"{len(Rs)} states, {len(occ)} field patterns, {sum(tested.values())} perturbations")
    print(f"truncations over all states at max_ent={a.max_ent} max_cent={a.max_cent}: {dict(trunc) or 'none'}")
    print("\nfields whose change does not change the encoding:")
    for pat in sorted(dropped):
        note = INTENDED.get(pat, "")
        tag = "intended" if note else "DROPPED"
        print(f"  {tag:8s} {pat:55s} {dropped[pat]}/{tested[pat]}  e.g. {examples[pat][0]} = {str(examples[pat][1])[:40]}"
              + (f"  ({note})" if note else ""))
    bad = [p for p in dropped if p not in INTENDED]
    print(f"\n{len(bad)} unintended dropped field patterns")
    return 1 if bad else 0


def count_truncations(enc, Rs):
    enc.trunc.clear()
    for R in Rs:
        enc.encode(R)
    return collections.Counter(enc.trunc)


if __name__ == "__main__":
    sys.exit(main())
