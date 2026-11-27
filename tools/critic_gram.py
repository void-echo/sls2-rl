"""Gram matrices of policy gradients under different advantage estimators, on a critic_lab dataset.

    PYTHONPATH=. python tools/critic_gram.py --data DIR --lams 0,0.5,0.8,0.9,0.95,0.98,0.99,1 --k 262144 --pairs 6 \
        --max_chunk 16 --out DIR/gram.json --device cuda:0

For every pair of episode-disjoint halves A, B of the held-out episodes and every lambda, the policy gradient is
computed separately on combat and on build decisions (the gradient is linear in the advantages, so any combination
of a combat lambda and a build lambda is a sum of two of these vectors). Advantages are raw (not normalised), from
critic_lab.gae_stitched with lam_build = lam. Saved per pair: the cross Gram <g_i^A, g_j^B> and the two self Grams,
over all parameters and without the hash table. `summarize` turns them into, for each (combat lambda, build lambda),
    q = 0.5 * (<g^A, r^B> / |g^A| + <g^B, r^A> / |g^B|),   r = the reference estimator (lambda 1 = Monte Carlo, unbiased)
i.e. how far one normalised step moves along the true gradient; (q / q_production)^2 is the sample-size equivalent.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.train import amp_context, load_policy
from tools.critic_lab import dataset_split, gae_stitched, load_dataset, stitch, valid_mask
from tools.grad_snr import episode_pair, policy_gradient


def gram(xs, ys, skip=()):
    out = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            out[i, j] = sum(float(torch.sum(x[n].double() * y[n].double())) for n in x.keys() & y.keys() if n not in skip)
    return out


def summarize(result, ref=None, ref_build=None):
    """Table of q and the sample-size gain for every (combat lambda, build lambda), averaged over pairs."""
    lams, names = result["lams"], result["names"]
    L = len(lams)
    ref = lams.index(1.0 if ref is None else ref)
    ref_build = ref if ref_build is None else lams.index(ref_build)
    out = {}
    for group in ("all", "no_table"):
        q = np.zeros((len(result["pairs"]), L, L))
        for p, pair in enumerate(result["pairs"]):
            X, AA, BB = (np.asarray(pair[group][k]) for k in ("cross", "self_a", "self_b"))
            ra, rb = ref, L + ref_build  # reference = combat part + build part
            for c in range(L):
                for b in range(L):
                    i, j = c, L + b
                    na = np.sqrt(AA[i, i] + AA[j, j] + 2 * AA[i, j])
                    nb = np.sqrt(BB[i, i] + BB[j, j] + 2 * BB[i, j])
                    a_ref_b = X[i, ra] + X[i, rb] + X[j, ra] + X[j, rb]
                    b_ref_a = X[ra, i] + X[rb, i] + X[ra, j] + X[rb, j]
                    q[p, c, b] = 0.5 * (a_ref_b / na + b_ref_a / nb)
        out[group] = dict(q_mean=q.mean(0).tolist(), q_se=(q.std(0, ddof=1) / np.sqrt(len(q))).tolist() if len(q) > 1 else None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--values", default="behaviour", help="behaviour, or the name of a critic_lab fit")
    ap.add_argument("--lams", default="0,0.5,0.8,0.9,0.95,0.98,0.99,1")
    ap.add_argument("--k", type=int, default=262144)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--max_chunk", type=int, default=0, help="use only transitions of the first N chunks (0 = all)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    lams = [float(x) for x in a.lams.split(",")]
    chunks, meta = load_dataset(a.data)
    info = json.loads((a.data / "meta.json").read_text())
    stitched = stitch(meta)
    split = dataset_split(a.data, meta, stitched)
    valid = valid_mask(meta, stitched) & np.isin(meta["episode"], split["measure"])
    if a.max_chunk:
        valid &= meta["chunk"] < a.max_chunk
        chunks = chunks[:a.max_chunk]
    values = meta["val"] if a.values == "behaviour" else np.load(a.data / f"fit_{a.values}" / "values.npy")
    adv = [gae_stitched(meta, stitched, values, lam, lam)[0] for lam in lams]
    combat = meta["combat"]
    print(f"valid measure transitions {int(valid.sum())}, combat share {float(combat[valid].mean()):.3f}", flush=True)
    for lam, x in zip(lams, adv):
        print(f"  lam {lam:g}: adv var combat {x[valid & combat].var():.5f} build {x[valid & ~combat].var():.5f}", flush=True)
    model, _, cfg, ck = load_policy(info["checkpoint"], "cpu")
    del ck
    model.to(a.device)
    amp_ctx = amp_context(cfg, torch.device(a.device))
    rng = np.random.default_rng(0)
    names = [f"combat:{lam:g}" for lam in lams] + [f"build:{lam:g}" for lam in lams]
    result = dict(checkpoint=info["checkpoint"], values=a.values, lams=lams, names=names, k=a.k, pairs=[])

    def gradients(indices):
        out = []
        for part in (indices[combat[indices]], indices[~combat[indices]]):
            for x in adv:
                # policy_gradient divides by len(indices): scale back so combat and build parts share one denominator
                g = policy_gradient(model, chunks, part, x, meta["act"], cfg.micro_batch, amp_ctx)
                out.append({n: v * (len(part) / len(indices)) for n, v in g.items()})
                model.zero_grad(set_to_none=True)
        return out

    for p in range(a.pairs):
        ia, ib = episode_pair(meta["episode"], valid, a.k, rng)
        ga, gb = gradients(ia), gradients(ib)
        row = dict(n_a=len(ia), n_b=len(ib))
        for group, skip in (("all", ()), ("no_table", ("table",))):
            row[group] = dict(cross=gram(ga, gb, skip).tolist(), self_a=gram(ga, ga, skip).tolist(), self_b=gram(gb, gb, skip).tolist())
        result["pairs"].append(row)
        del ga, gb
        result["summary"] = summarize(result)
        a.out.write_text(json.dumps(result) + "\n")
        print(f"pair {p + 1}/{a.pairs} n={len(ia)}/{len(ib)}", flush=True)
    L = len(lams)
    for group in ("all", "no_table"):
        q = np.asarray(result["summary"][group]["q_mean"])
        prod = q[lams.index(info["lam"]), lams.index(info["lam_build"] if info["lam_build"] is not None else info["lam"])]
        print(f"\n{group}: gain = (q / q_production)^2, rows = combat lambda, columns = build lambda; reference lambda 1")
        print("        " + " ".join(f"{lam:>7g}" for lam in lams))
        for c in range(L):
            print(f"{lams[c]:>7g} " + " ".join(f"{(q[c, b] / prod) ** 2 * np.sign(q[c, b]):7.2f}" for b in range(L)))


if __name__ == "__main__":
    main()
