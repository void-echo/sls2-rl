"""Average several checkpoints into one file that evaluates exactly like passing them all to tools/eval_variant.py
(same double-precision mean, cast back to each tensor's dtype). Used to ship one file instead of four to Colab.
    python3 tools/avg_ckpts.py OUT.pt CKPT...      (keeps model, cfg, vnorm, update, env_steps of the last one)"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.eval_variant import averaged_state  # noqa: E402

out, paths = sys.argv[1], sys.argv[2:]
last = torch.load(paths[-1], map_location="cpu", weights_only=False)
avg = averaged_state(paths)
model = {k: v.to(last["model"][k].dtype) for k, v in avg.items()}
keep = {k: last[k] for k in ("cfg", "vnorm", "update", "env_steps") if k in last}
keep["model"], keep["averaged_from"] = model, [os.path.basename(p) for p in paths]
torch.save(keep, out + ".tmp")
os.replace(out + ".tmp", out)
print(f"{out}: mean of {len(paths)} checkpoints ending u{last.get('update')}")
