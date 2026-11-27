"""Extended Data Figure 3: Parameter-path straightness, plasticity trajectories across 6.7B steps,
memorization probe curves, and post-update adaptation step-size / network-age tests.

  a, Parameter-path straightness ||Δθ||_2 / Σ||δθ||_2 across update windows k = 2, 4, 8, 16 for policy weights
     (0.69, 0.48, 0.33, 0.24) against an isotropic random walk 1/sqrt(k) (0.71, 0.50, 0.35, 0.25) and the value head
     at k = 16 (0.35, +40% straighter).
  b, Dormant-unit (τ = 0.1) and normalized trunk attention entropy trajectories across the early large-step lineage
     (0.8% -> 46.0% dormant, 0.314 -> 0.013 entropy; lineage_summary.csv) versus the distilled small-step (KL = 0.0005)
     Ironclad lineage through 6.68B steps (3.5B, 5.2B, 6.7B: 0.0% trunk dormant, 0.865 entropy).
  c, 1,000-step random-target Gaussian memorization probe MSE curves (plast1125_result.json & plast1125_26700.json)
     comparing random init, freshly distilled, 3.5B, 5.2B, 6.7B and aged large-step, alongside relative final MSE
     across all six characters (0.20-0.27x vs 14.9x).
  d, Post-update v0.113 adaptation speed & step size: 3 seeds of KL = 0.0005 (62.5 ± 0.4%) vs 3 seeds of KL = 0.002
     (55.9 ± 1.7%, t = -6.3) from paper/figdata/catchup/*.eval_a10.csv (left), and 3.5B-step old network continued
     (63.1, 64.7, 64.0, 65.2%) vs freshly distilled network (49.3, 56.4, 59.6, 60.7%) across 1-4 x 10^8 steps (right).

Writes paper/figs/ed_fig3_plasticity_steps.{png,pdf,svg} (350 DPI).
"""
import csv
import json
import math
import os
import re
import subprocess
import sys

from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_V2_DIR = os.path.join(R, "tmp", "game_assets", "fig14_v2")
ASSET_ED_DIR = os.path.join(R, "tmp", "game_assets", "ed_figs")
if not os.path.exists(os.path.join(ASSET_V2_DIR, "badge_char_ironclad.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_V2_DIR, "build_assets.py")], check=True, cwd=R)


def register_helvetica_bold():
    dst = os.path.join(ASSET_ED_DIR, "Helvetica-Bold.ttf")
    try:
        if not os.path.exists(dst):
            from fontTools.ttLib import TTCollection
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            for face in TTCollection("/System/Library/Fonts/Helvetica.ttc").fonts:
                if face["name"].getDebugName(2) == "Bold":
                    face.save(dst)
                    break
        font_manager.fontManager.addfont(dst)
    except Exception as exc:
        print("Helvetica Bold not registered:", exc)


register_helvetica_bold()

# Palette (STYLE.md)
BLUE = "#2166ac"
BLUE_PILL = "#dceaf7"
BLUE_SOFT = "#eef4fb"
TERRA = "#c85228"
TERRA_PILL = "#fbe6de"
TEAL = "#1b8a6b"
TEAL_PILL = "#d7f0e7"
AMBER = "#c98a1b"
VIOLET = "#5e4cc4"
PLUM = "#b85682"
SLATE = "#64748b"
INK = "#1f2937"
INK2 = "#475569"
ZEBRA = "#f4f6f9"
GRID = "#e8edf3"
BORDER = "#d5dde8"
OUTER_BORDER = "#e2e8f0"
MINUS = "−"
STEP = 131072

CHAR_COLORS = {
    "Ironclad": BLUE,
    "Defect": TERRA,
    "Regent": TEAL,
    "Silent": AMBER,
    "Necrobinder": PLUM,
    "Fool": VIOLET,
}
CHAR_BADGES = {
    "Ironclad": "badge_char_ironclad.png",
    "Defect": "badge_char_defect.png",
    "Regent": "badge_char_regent.png",
    "Silent": "badge_char_silent.png",
    "Necrobinder": "badge_char_necrobinder.png",
    "Fool": "badge_char_fool.png",
}

FS_TICK = 6.8
FS_AX = 7.6
FS_LAB = 7.2
FS_CELL = 6.8
FS_LETTER = 11.0

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": FS_TICK,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.labelsize": FS_AX,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica",
    "mathtext.bf": "Helvetica:bold",
})


def load_rgba_clean(fn, target_h_px):
    im = Image.open(os.path.join(ASSET_V2_DIR, fn)).convert("RGBA")
    arr = np.asarray(im, dtype=np.float32)
    a = np.where(arr[..., 3] < 8.0, 0.0, arr[..., 3]) / 255.0
    pre = np.concatenate([arr[..., :3] * a[..., None], (a * 255.0)[..., None]], axis=-1)
    im_pre = Image.fromarray(np.clip(pre + 0.5, 0, 255).astype(np.uint8), "RGBA")
    nh = max(16, int(round(target_h_px)))
    nw = max(16, int(round(im.width * (nh / im.height))))
    im_rs = im_pre.resize((nw, nh), Image.Resampling.LANCZOS)
    rs = np.asarray(im_rs, dtype=np.float32)
    ra = rs[..., 3:4] / 255.0
    rgb = np.where(ra > 1e-3, np.clip(rs[..., :3] / np.maximum(ra, 1e-3), 0, 255), 255.0)
    ra_clean = np.where(rs[..., 3:4] < 5.0, 0.0, rs[..., 3:4])
    return np.concatenate([rgb, ra_clean], axis=-1).astype(np.uint8), nh


def add_sprite(ax, fn, xy, target_h_pt, xycoords="data", align=(0.5, 0.5), zorder=6):
    arr, h_px = load_rgba_clean(fn, target_h_pt * 4.8)
    oi = OffsetImage(arr, zoom=target_h_pt / h_px, interpolation="bilinear", resample=True)
    ab = AnnotationBbox(oi, xy, xycoords=xycoords, box_alignment=align, frameon=False, pad=0,
                        zorder=zorder, annotation_clip=False)
    ab.set_clip_on(False)
    ax.add_artist(ab)


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.6, pad=2.0)


def wilson_pct(p_pct, n, z=1.96):
    p = p_pct / 100.0
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return 100.0 * max(0.0, c - h), 100.0 * min(1.0, c + h)


# =========================================================================
# 1. Load Ground-Truth Data
# =========================================================================
lin_rows = list(csv.DictReader(open(os.path.join(R, "docs", "results", "plasticity", "lineage_summary.csv"))))
p267 = json.load(open(os.path.join(R, "paper", "figdata", "plast1125_26700.json")))["checkpoints"][0]
pres = {
    c["checkpoint"]: c
    for c in json.load(open(os.path.join(R, "paper", "figdata", "plast1125_result.json")))["checkpoints"]
    if c.get("status") == "ok"
}
mc_fig2 = json.load(open(os.path.join(R, "paper", "figdata", "plast_multichar_fig2.json")))
mc_sum = {
    r["character"]: r
    for r in json.load(open(os.path.join(R, "paper", "figdata", "multichar_plasticity_1125.json")))["summary_table"]
}

# Panel a: Straightness (paper/draft.md & docs/03_强化学习策略.md line 155)
K_WIN = [2, 4, 8, 16]
POL_S = [0.69, 0.48, 0.33, 0.24]
RW_S = [1.0 / math.sqrt(k) for k in K_WIN]
VAL_S16 = 0.35

# Panel b: Lineage trajectories
scratch = sorted([r for r in lin_rows if r["ckpt"].startswith("boss_g1")], key=lambda r: float(r["env_steps_M"]))
desc = sorted([r for r in lin_rows if r["ckpt"] in ("bigb_009500", "asc_011700", "mg_011800")],
              key=lambda r: float(r["env_steps_M"]))
stu = sorted([r for r in lin_rows if r["group"] == "student"], key=lambda r: float(r["env_steps_M"]))

c40 = pres["runs/v3_iron/eval_ckpts/ckpt_040000.pt"]
c51 = pres["runs/v3_iron/eval_ckpts/ckpt_051000.pt"]
cyk = pres["runs/plast1125/young_kd.pt"]
crand = pres["random"]

x_scr_B = [float(r["env_steps_M"]) / 1000.0 for r in scratch]
d_scr = [100.0 * float(r["dormant_01"]) for r in scratch]
a_scr = [float(r["attn_entropy"]) for r in scratch]

x_desc_B = [float(r["env_steps_M"]) / 1000.0 for r in desc]
d_desc = [100.0 * float(r["dormant_01"]) for r in desc]
a_desc = [float(r["attn_entropy"]) for r in desc]

x_stu_B = [float(r["env_steps_M"]) / 1000.0 for r in stu]
d_stu_all = [100.0 * float(r["dormant_01"]) for r in stu]
a_stu = [float(r["attn_entropy"]) for r in stu]

x_long_B = [p267["update"] * STEP / 1e9, c40["update"] * STEP / 1e9, c51["update"] * STEP / 1e9]
d_long_trunk = [
    100.0 * p267["dormant"]["mean_by_kind"]["blocks"]["0.1"],
    100.0 * c40["dormant"]["mean_by_kind"]["blocks"]["0.1"],
    100.0 * c51["dormant"]["mean_by_kind"]["blocks"]["0.1"],
]
d_long_all = [
    100.0 * p267["dormant"]["mean"]["0.1"],
    100.0 * c40["dormant"]["mean"]["0.1"],
    100.0 * c51["dormant"]["mean"]["0.1"],
]
a_long = [p267["attention_entropy_mean"], c40["attention_entropy_mean"], c51["attention_entropy_mean"]]

# Panel c: Memorization probe curves (steps = 0, 100, 300, 1000) & 6-char relative final MSE
PROBE_STEPS = [0, 100, 300, 1000]


def get_mem_curve(ck_obj):
    m_dict = ck_obj["probe"]["memory"]["mse"]
    return [float(m_dict[str(s)]) for s in PROBE_STEPS]


aged_obj = mc_fig2["baselines"]["aged_large_step"]
aged_curve = list(aged_obj["probe"]["memory"]["mse"])
rand_curve = get_mem_curve(crand)
ykd_curve = get_mem_curve(cyk)
u267_curve = get_mem_curve(p267)
u400_curve = get_mem_curve(c40)
u510_curve = get_mem_curve(c51)

RAND_AGED_BANK = float(next(r for r in lin_rows if r["ckpt"] == "random")["probe_memory_1000"])
rel_aged = aged_obj["probe"]["memory"]["final_mse"] / RAND_AGED_BANK  # 14.88x
rel_6chars = []
for ch in ("Ironclad", "Silent", "Necrobinder", "Regent", "Defect", "Fool"):
    ch_mse = mc_fig2["characters_latest"][ch]["probe"]["memory"]["final_mse"]
    ch_rand = mc_sum[ch]["random_memory_probe_mse_1000"]
    ch_B = mc_fig2["characters_latest"][ch]["env_steps_B"]
    rel_6chars.append((ch, ch_B, ch_mse / ch_rand))

# Panel d left: Catch-up KL = 0.0005 vs KL = 0.002 (3 seeds each)
FORK_PATCH_U = 26736
catch_dir = os.path.join(R, "paper", "figdata", "catchup")


def load_catchup_csv(fn):
    xs, ys = [], []
    for r in csv.DictReader(open(os.path.join(catch_dir, fn))):
        if r["mode"] != "greedy":
            continue
        u = int(r["update"])
        if u <= 27200:
            xs.append((u - FORK_PATCH_U) * STEP / 1e6)
            ys.append(100.0 * float(r["win_rate"]))
    return np.array(xs), np.array(ys)


k05_trajs = [load_catchup_csv(f) for f in ("v3k_iron_k05.eval_a10.csv", "v3k_iron_k05_s12.eval_a10.csv", "v3k_iron_k05_s13.eval_a10.csv")]
k20_trajs = [load_catchup_csv(f) for f in ("v3k_iron_k20.eval_a10.csv", "v3k_iron_k20_s12.eval_a10.csv", "v3k_iron_k20_s13.eval_a10.csv")]

# 2,000-seed final evaluations at 61M steps (docs/03 line 223 + tmp/fresh1117/{k05,k20}s{12,13}_u27200.json)
docs03 = open(os.path.join(R, "docs", "03_强化学习策略.md"), encoding="utf-8").read()
m_k05_s1 = re.search(r"0\.0005 三颗种子是 ([0-9.]+)、([0-9.]+)、([0-9.]+)", docs03)
m_k20_s1 = re.search(r"0\.002 三颗是 ([0-9.]+)、([0-9.]+)、([0-9.]+)", docs03)
k05_2k = [float(x) for x in m_k05_s1.groups()]  # [62.20, 63.00, 62.25]
k20_2k = [float(x) for x in m_k20_s1.groups()]  # [57.93, 55.20, 54.70]
# Cross-check s12/s13 directly against JSON files if present
for idx_s, stem in [(1, "k05s12_u27200.json"), (2, "k05s13_u27200.json")]:
    p_j = os.path.join(R, "tmp", "fresh1117", stem)
    if os.path.exists(p_j):
        eps = json.load(open(p_j))["episodes"]
        k05_2k[idx_s] = 100.0 * sum(bool(e["victory"]) for e in eps) / len(eps)
for idx_s, stem in [(1, "k20s12_u27200.json"), (2, "k20s13_u27200.json")]:
    p_j = os.path.join(R, "tmp", "fresh1117", stem)
    if os.path.exists(p_j):
        eps = json.load(open(p_j))["episodes"]
        k20_2k[idx_s] = 100.0 * sum(bool(e["victory"]) for e in eps) / len(eps)

k05_mean, k05_sd = float(np.mean(k05_2k)), float(np.std(k05_2k, ddof=1))
k20_mean, k20_sd = float(np.mean(k20_2k)), float(np.std(k20_2k, ddof=1))
t_stat = (k20_mean - k05_mean) / math.sqrt(k05_sd ** 2 / 3.0 + k20_sd ** 2 / 3.0)

# Panel d right: 3.5B old network continued vs freshly distilled network (1-4 x 10^8 steps, 2,000 seeds/pt)
old_2k, young_2k = [], []
for s_M, folder in [(100, "fresh1118"), (200, "fresh1118"), (300, "fresh1119"), (400, "fresh1119")]:
    po = os.path.join(R, "tmp", folder, f"old_p{s_M}M.json")
    py = os.path.join(R, "tmp", folder, f"young_p{s_M}M.json")
    eo = json.load(open(po))["episodes"]
    ey = json.load(open(py))["episodes"]
    old_2k.append(100.0 * sum(bool(e["victory"]) for e in eo) / len(eo))
    young_2k.append(100.0 * sum(bool(e["victory"]) for e in ey) / len(ey))

# =========================================================================
# 2. Figure Layout (8.5 x 5.65 in, 350 DPI)
# =========================================================================
FIG_W, FIG_H = 8.5, 5.65
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

fig.add_artist(FancyBboxPatch(
    (0.006, 0.008), 0.988, 0.984, transform=fig.transFigure,
    boxstyle="round,pad=0.0,rounding_size=0.014",
    facecolor="none", edgecolor=OUTER_BORDER, linewidth=0.8, zorder=0
))

# Top row: Panel a (left 38%), Panel b (right 55%, stacked dormant + attention entropy)
ax_a = fig.add_axes([0.072, 0.562, 0.335, 0.372])
ax_b1 = fig.add_axes([0.488, 0.760, 0.482, 0.174])
ax_b2 = fig.add_axes([0.488, 0.562, 0.482, 0.174])

# Bottom row: Panel c (left: probe curves + 6-char relative column), Panel d (right: KL seeds + old vs fresh)
ax_c1 = fig.add_axes([0.072, 0.102, 0.258, 0.365])
ax_c2 = fig.add_axes([0.368, 0.102, 0.118, 0.365])

ax_d1 = fig.add_axes([0.556, 0.102, 0.205, 0.365])
ax_d2 = fig.add_axes([0.812, 0.102, 0.162, 0.365])

# =========================================================================
# Panel a: Parameter-path straightness across update windows k = 2, 4, 8, 16
# =========================================================================
clean_ax(ax_a)
for gy in (0.2, 0.4, 0.6, 0.8):
    ax_a.axhline(gy, color=GRID, lw=0.55, zorder=0)

ax_a.plot(K_WIN, RW_S, "s--", color=SLATE, ms=4.8, mec="white", mew=0.5, lw=1.35,
          label="Random walk", zorder=3)
ax_a.plot(K_WIN, POL_S, "o-", color=BLUE, ms=5.4, mec="white", mew=0.55, lw=1.75,
          label="Policy network", zorder=4)
ax_a.scatter([16], [VAL_S16], s=52, marker="D", color=TEAL, edgecolor="white", linewidth=0.65,
             label="Value head", zorder=5)

for kx, py in zip(K_WIN[:-1], POL_S[:-1]):
    ax_a.text(kx, py - 0.038, f"{py:.2f}", fontsize=FS_LAB, fontweight="bold", color=BLUE,
              ha="center", va="top", zorder=5)

# Vertical gap callout at k = 16 (placed cleanly in open space above the diamond)
ax_a.annotate("", xy=(16, VAL_S16 - 0.008), xytext=(16, POL_S[-1] + 0.010),
              arrowprops=dict(arrowstyle="<->,head_width=0.18,head_length=0.22", lw=1.0, color=TEAL), zorder=4)
ax_a.text(16.4, 0.405, "Value: 0.35\n(+40% straighter)", fontsize=FS_LAB, fontweight="bold",
          color=TEAL, ha="right", va="bottom", linespacing=1.06, zorder=5)
ax_a.plot([15.2, 15.85], [0.400, 0.362], color=TEAL, lw=0.75, zorder=4)
ax_a.text(16.0, 0.205, "0.24 vs 0.25", fontsize=FS_LAB, fontweight="bold",
          color=BLUE, ha="right", va="top", zorder=5)
add_sprite(ax_a, CHAR_BADGES["Ironclad"], (2.35, 0.20), target_h_pt=19.5, zorder=6)

ax_a.set_xscale("log", base=2)
ax_a.set_xlim(1.68, 20.5)
ax_a.set_ylim(0.10, 0.84)
ax_a.set_xticks(K_WIN)
ax_a.set_xticklabels([str(k) for k in K_WIN])
ax_a.minorticks_off()
ax_a.set_yticks([0.2, 0.4, 0.6, 0.8])
ax_a.set_xlabel("Window length (updates)", labelpad=2.5)
ax_a.set_ylabel("Straightness of the weight path", labelpad=2.0)
ax_a.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=True, facecolor="white",
            edgecolor=BORDER, framealpha=0.95, fontsize=FS_TICK, handlelength=1.4, borderpad=0.35)

# =========================================================================
# Panel b: Dormant units & attention entropy trajectories (log10 env steps)
# =========================================================================
clean_ax(ax_b1)
clean_ax(ax_b2)
for gy in (20, 40):
    ax_b1.axhline(gy, color=GRID, lw=0.55, zorder=0)
for gy in (0.4, 0.8):
    ax_b2.axhline(gy, color=GRID, lw=0.55, zorder=0)

for ax_sub in (ax_b1, ax_b2):
    ax_sub.axvline(3.50, color=BORDER, lw=0.85, ls=":", zorder=1)

# Top: Dormant units (τ = 0.1, %)
ax_b1.plot(x_scr_B, d_scr, "o-", color=TERRA, ms=4.2, mec="white", mew=0.45, lw=1.55,
           label="Large steps", zorder=4)
ax_b1.scatter(x_desc_B, d_desc, s=22, marker="s", facecolor="white", edgecolor=TERRA, linewidth=1.0, zorder=4)

# Distilled small-step: student A0 -> 3.5B, 5.2B, 6.7B
ax_b1.plot(x_stu_B + x_long_B, d_stu_all + d_long_all, "s--", color=TEAL, ms=3.8, mec="white",
           mew=0.4, lw=1.1, alpha=0.85, label="Our step size, all layers", zorder=3)
ax_b1.plot(x_long_B, d_long_trunk, "o-", color=BLUE, ms=5.0, mec="white", mew=0.5, lw=1.65,
           label="Our step size, trunk", zorder=5)

ax_b1.text(0.433, 47.8, "46.0%", fontsize=FS_LAB, fontweight="bold", color=TERRA, ha="center", va="bottom")
ax_b1.text(6.68, 5.2, "0.0% trunk", fontsize=FS_LAB, fontweight="bold", color=BLUE, ha="right", va="bottom")
ax_b1.text(3.40, 46.5, "v0.113", fontsize=FS_TICK, color=INK2, ha="right", va="center")

ax_b1.set_xscale("log")
ax_b1.set_xlim(0.0025, 9.5)
ax_b1.set_ylim(-3.5, 56)
ax_b1.set_xticks([0.01, 0.1, 1.0, 6.7])
ax_b1.set_xticklabels([])
ax_b1.minorticks_off()
ax_b1.set_yticks([0, 20, 40])
ax_b1.set_ylabel("Dormant units (%)", labelpad=2.5)
ax_b1.legend(loc="upper left", bbox_to_anchor=(0.01, 1.02), frameon=False, fontsize=FS_TICK - 0.1,
             handlelength=1.3, labelspacing=0.18)

# Bottom: Normalized trunk attention entropy
ax_b2.plot(x_scr_B, a_scr, "o-", color=TERRA, ms=4.2, mec="white", mew=0.45, lw=1.55, zorder=4)
ax_b2.scatter(x_desc_B, a_desc, s=22, marker="s", facecolor="white", edgecolor=TERRA, linewidth=1.0, zorder=4)
ax_b2.plot(x_stu_B, a_stu, "o-", color=TEAL, ms=4.0, mec="white", mew=0.45, lw=1.35, zorder=4)
ax_b2.plot([x_stu_B[-1], x_long_B[0]], [a_stu[-1], a_long[0]], "--", color=BLUE, lw=1.0, alpha=0.75, zorder=3)
ax_b2.plot(x_long_B, a_long, "o-", color=BLUE, ms=5.0, mec="white", mew=0.5, lw=1.65, zorder=5)

ax_b2.text(0.433, 0.08, "0.013", fontsize=FS_LAB, fontweight="bold", color=TERRA, ha="center", va="bottom")
ax_b2.text(6.68, 0.90, "0.865", fontsize=FS_LAB, fontweight="bold", color=BLUE, ha="right", va="bottom")
add_sprite(ax_b2, CHAR_BADGES["Ironclad"], (0.0052, 0.20), target_h_pt=17.5, zorder=6)

ax_b2.set_xscale("log")
ax_b2.set_xlim(0.0025, 9.5)
ax_b2.set_ylim(-0.04, 1.06)
ax_b2.set_xticks([0.01, 0.1, 1.0, 3.5, 6.7])
ax_b2.set_xticklabels(["0.01", "0.1", "1", "3.5", "6.7"])
ax_b2.minorticks_off()
ax_b2.set_yticks([0.0, 0.4, 0.8])
ax_b2.set_xlabel("Cumulative environment steps (billions, log scale)", labelpad=2.5)
ax_b2.set_ylabel("Attention entropy", labelpad=2.5)

# =========================================================================
# Panel c1: 1,000-step Gaussian memorization probe MSE curves
# =========================================================================
clean_ax(ax_c1)
for gy in (0.05, 0.2, 1.0):
    ax_c1.axhline(gy, color=GRID, lw=0.55, zorder=0)

xp_idx = np.arange(len(PROBE_STEPS))
ax_c1.plot(xp_idx, aged_curve, "^:", color=TERRA, ms=4.5, mec="white", mew=0.45, lw=1.35,
           label="Large steps", zorder=3)
ax_c1.plot(xp_idx, rand_curve, "s--", color=SLATE, ms=4.5, mec="white", mew=0.45, lw=1.35,
           label="Untrained", zorder=3)
ax_c1.plot(xp_idx, ykd_curve, "o-", color=TEAL, ms=4.6, mec="white", mew=0.45, lw=1.4,
           label="Freshly distilled", zorder=4)
ax_c1.plot(xp_idx, u267_curve, "D-", color=AMBER, ms=4.4, mec="white", mew=0.45, lw=1.35,
           label="Ours, 3.5B steps", zorder=4)
ax_c1.plot(xp_idx, u400_curve, "v-", color=VIOLET, ms=4.4, mec="white", mew=0.45, lw=1.35,
           label="Ours, 5.2B steps", zorder=5)
ax_c1.plot(xp_idx, u510_curve, "o-", color=BLUE, ms=5.0, mec="white", mew=0.5, lw=1.75,
           label="Ours, 6.7B steps", zorder=6)

ax_c1.text(3.06, aged_curve[-1], f"{aged_curve[-1]:.2f}", fontsize=FS_CELL, fontweight="bold",
           color=TERRA, va="center")
ax_c1.text(3.06, rand_curve[-1], f"{rand_curve[-1]:.2f}", fontsize=FS_CELL, fontweight="bold",
           color=SLATE, va="center")
ax_c1.text(3.06, u510_curve[-1], f"{u510_curve[-1]:.3f}", fontsize=FS_CELL, fontweight="bold",
           color=BLUE, va="center")

ax_c1.set_yscale("log")
ax_c1.set_xlim(-0.15, 3.68)
ax_c1.set_ylim(0.032, 2.75)
ax_c1.set_xticks(xp_idx)
ax_c1.set_xticklabels(["0", "100", "300", "1,000"])
ax_c1.minorticks_off()
ax_c1.set_yticks([0.05, 0.2, 1.0])
ax_c1.set_yticklabels(["0.05", "0.2", "1.0"])
ax_c1.set_xlabel("Fitting steps", labelpad=2.5)
ax_c1.set_ylabel("Fitting error (log)", labelpad=1.5)
ax_c1.legend(loc="lower left", bbox_to_anchor=(0.01, 0.01), frameon=True, facecolor="white",
             edgecolor=BORDER, framealpha=0.95, fontsize=FS_TICK - 0.2, handlelength=1.3,
             borderpad=0.28, labelspacing=0.18)

# =========================================================================
# Panel c2: Relative final MSE vs random init across Aged + all 6 characters
# =========================================================================
clean_ax(ax_c2)
ax_c2.axvline(1.0, color=SLATE, lw=0.85, ls="--", zorder=1)

rel_rows = [("Aged", TERRA, rel_aged, None)] + [
    (ch, CHAR_COLORS[ch], r_val, CHAR_BADGES[ch]) for ch, _, r_val in rel_6chars
]
for i, (nm, col, r_val, b_fn) in enumerate(rel_rows):
    ry = i + (0.28 if i >= 1 else 0.0)
    # Bar from 1.0x (random init) to r_val on log scale
    left_v = min(1.0, r_val)
    width_v = abs(r_val - 1.0)
    ax_c2.barh(ry, width_v, left=left_v, height=0.58, color=col, alpha=0.86, edgecolor="white",
               linewidth=0.5, zorder=2)
    if b_fn is not None:
        add_sprite(ax_c2, b_fn, (0.092, ry), target_h_pt=15.0, zorder=6)
        ax_c2.text(1.28, ry, f"{r_val:.2f}×", fontsize=FS_CELL, fontweight="bold",
                   color=col, va="center", ha="left", zorder=4)
    else:
        ax_c2.text(0.092, ry, "Large steps", fontsize=FS_CELL, fontweight="bold",
                   color=TERRA, va="center", ha="center", zorder=4)
        ax_c2.text(r_val * 0.84, ry, f"{r_val:.1f}×", fontsize=FS_CELL, fontweight="bold",
                   color="white", va="center", ha="right", zorder=4)

ax_c2.axhline(0.64, color=BORDER, lw=0.75, zorder=2)
ax_c2.set_xscale("log")
ax_c2.set_xlim(0.06, 23.0)
ax_c2.set_ylim(6.85, -0.65)
ax_c2.set_xticks([0.2, 1.0, 10.0])
ax_c2.set_xticklabels(["0.2×", "1×", "10×"])
ax_c2.minorticks_off()
ax_c2.set_yticks([])
ax_c2.spines["left"].set_visible(False)
ax_c2.set_xlabel("Error relative to untrained", labelpad=2.5)
ax_c2.text(1.0, -0.68, "untrained", fontsize=FS_TICK, color=SLATE, fontweight="bold", ha="center", va="bottom")

# =========================================================================
# Panel d1: Post-update catch-up (3 seeds KL = 0.0005 vs 3 seeds KL = 0.002)
# =========================================================================
clean_ax(ax_d1)
for gy in (50, 55, 60, 65):
    ax_d1.axhline(gy, color=GRID, lw=0.55, zorder=0)

for xs, ys in k05_trajs:
    ax_d1.plot(xs, ys, "-", color=BLUE, lw=0.85, alpha=0.35, zorder=2)
for xs, ys in k20_trajs:
    ax_d1.plot(xs, ys, "-", color=TERRA, lw=0.85, alpha=0.35, zorder=2)

# Mean fixed-200-seed curves across the common update grid
common_x = k05_trajs[1][0]  # up to ~55M
m_k05 = np.mean([ys[:len(common_x)] for _, ys in k05_trajs], axis=0)
m_k20 = np.mean([ys[:len(common_x)] for _, ys in k20_trajs], axis=0)
ax_d1.plot(common_x, m_k05, "-", color=BLUE, lw=1.7, label="Our step size", zorder=4)
ax_d1.plot(common_x, m_k20, "-", color=TERRA, lw=1.7, label="4× larger steps", zorder=4)

# 2,000-seed final evaluations at 61M steps
ax_d1.axvline(61.0, color=BORDER, lw=0.8, ls=":", zorder=1)
for j_s, v in enumerate(k05_2k):
    ax_d1.scatter([61.0 + (j_s - 1) * 1.4], [v], s=25, color=BLUE, edgecolor="white", linewidth=0.55, zorder=6)
for j_s, v in enumerate(k20_2k):
    ax_d1.scatter([61.0 + (j_s - 1) * 1.4], [v], s=25, color=TERRA, edgecolor="white", linewidth=0.55, zorder=6)

ax_d1.text(64.2, k05_mean, f"{k05_mean:.1f}\n±{k05_sd:.1f}%", fontsize=FS_CELL, fontweight="bold",
           color=BLUE, ha="left", va="center", linespacing=1.02, zorder=6)
ax_d1.text(64.2, k20_mean - 0.4, f"{k20_mean:.1f}\n±{k20_sd:.1f}%",
           fontsize=FS_CELL, fontweight="bold", color=TERRA, ha="left", va="center",
           linespacing=1.02, zorder=6)

ax_d1.set_xlim(0, 81.5)
ax_d1.set_ylim(45.5, 69.2)
ax_d1.set_xticks([0, 20, 40, 61])
ax_d1.set_yticks([50, 55, 60, 65])
ax_d1.set_xlabel("Steps on v0.113 (millions)", labelpad=2.5)
ax_d1.set_ylabel("Ascension 10 win rate (%)", labelpad=2.0)
ax_d1.legend(loc="lower left", bbox_to_anchor=(0.01, 0.01), frameon=True, facecolor="white",
             edgecolor=BORDER, framealpha=0.95, fontsize=FS_TICK - 0.1, handlelength=1.15, borderpad=0.28)

# =========================================================================
# Panel d2: 3.5B old network continued vs freshly distilled (1-4 x 10^8 steps)
# =========================================================================
clean_ax(ax_d2)
for gy in (50, 55, 60, 65):
    ax_d2.axhline(gy, color=GRID, lw=0.55, zorder=0)

xs_d2 = np.array([1.0, 2.0, 3.0, 4.0])
lo_o = [wilson_pct(v, 2000)[0] for v in old_2k]
hi_o = [wilson_pct(v, 2000)[1] for v in old_2k]
lo_y = [wilson_pct(v, 2000)[0] for v in young_2k]
hi_y = [wilson_pct(v, 2000)[1] for v in young_2k]

ax_d2.fill_between(xs_d2, lo_o, hi_o, color=BLUE, alpha=0.16, lw=0, zorder=2)
ax_d2.fill_between(xs_d2, lo_y, hi_y, color=SLATE, alpha=0.16, lw=0, zorder=2)
ax_d2.plot(xs_d2, old_2k, "o-", color=BLUE, ms=4.8, mec="white", mew=0.5, lw=1.7, label="Trained 3.5B", zorder=5)
ax_d2.plot(xs_d2, young_2k, "s-", color=SLATE, ms=4.6, mec="white", mew=0.5, lw=1.55, label="Freshly distilled", zorder=4)

for x_i, vo, vy in zip(xs_d2, old_2k, young_2k):
    vo_s = str(Decimal(str(vo)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    vy_s = str(Decimal(str(vy)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    ax_d2.text(x_i, vo + 1.55, vo_s, fontsize=FS_CELL, fontweight="bold", color=BLUE, ha="center", va="bottom")
    ax_d2.text(x_i, vy - 1.70, vy_s, fontsize=FS_CELL, fontweight="bold", color=SLATE, ha="center", va="top")

add_sprite(ax_d2, CHAR_BADGES["Ironclad"], (1.18, 57.5), target_h_pt=17.5, zorder=6)

ax_d2.set_xlim(0.60, 4.40)
ax_d2.set_ylim(45.5, 69.2)
ax_d2.set_xticks([1, 2, 3, 4])
ax_d2.set_yticks([50, 55, 60, 65])
ax_d2.set_yticklabels([])
ax_d2.set_xlabel("Steps on v0.113 (×10$^{8}$)", labelpad=2.5)
ax_d2.legend(loc="lower right", bbox_to_anchor=(0.99, 0.03), frameon=False, fontsize=FS_TICK,
             handlelength=1.1, handletextpad=0.32)

# =========================================================================
# Panel letters
# =========================================================================
for lab, (gx, gy) in zip("abcd", [
    (0.014, 0.980), (0.435, 0.980),
    (0.014, 0.495), (0.506, 0.495),
]):
    fig.text(gx, gy, lab, fontsize=FS_LETTER, fontweight="bold", va="top", color=INK)

out_base = os.path.join(R, "paper", "figs", "ed_fig3_plasticity_steps")
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out_base}.{ext}", dpi=350)
print(f"written {out_base}.{{png,pdf,svg}}")
print(f"d1: KL=5e-4 {k05_mean:.2f}±{k05_sd:.2f}% vs KL=2e-3 {k20_mean:.2f}±{k20_sd:.2f}% (t={t_stat:.2f})")
print("d2 old vs fresh:", list(zip(old_2k, young_2k)))
