"""Figure 2: a small network learns the game from scratch to above 70% and keeps learning for billions of steps,
because it moves in small steps.

Left to right, top row = how good it is, bottom row = why it can keep going.
  a, Greedy Ascension 10 win rate of the six characters across the v0.111 -> v0.113 update (1,000 paired seeds before
     and right after the update from tmp/cold1115/res, 2,000 fresh seeds per later point including the 6th post-update
     reading from paper/figdata/fresh1127_5char_rv1.json), with the five-character mean before / right after / adapted
     in the strip on top (52.5% -> 48.3% -> 73.0%) and the six-character key of the whole figure.
  b, Frozen v0.107.1: our five networks zero-shot against three published agents (Beta posteriors, value columns).
  c, Same-seed RV1 comparison with AlphaDefect across all four released characters (paper/figdata/fresh1127_5char_rv1.json).
  d, Inter-batch policy-gradient cosine (0.51 -> 0.06) and implied batch size for SNR = 1 (0.13M -> 2.05M decisions)
     across four Ironclad training stages (docs/03, tools/grad_snr.py).
  e, Step-size catastrophe at ~73% A10: five-arm causal fork from Ironclad u49300 over 300 updates = 39.3M steps evaluated
     on 2,000 paired fresh1105 seeds (paper/figdata/kl_catastrophe_1127.json), with aligned weight-displacement strip.
  f, Six-character plasticity heat-map matrix on 24,576 states: aged large-step network, random initialisation, freshly
     distilled network and all six characters (paper/figdata/plast_multichar_fig2.json, multichar_plasticity_1125.json,
     docs/results/plasticity/lineage_summary.csv).
Writes paper/figs/fig2_learning.{png,pdf,svg} (350 dpi).
"""
import csv
import json
import math
import os
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from PIL import Image
from scipy.stats import beta as beta_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(R, "tmp", "game_assets", "fig14_v2")
if not os.path.exists(os.path.join(ASSET_DIR, "badge_char_ironclad.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_DIR, "build_assets.py")], check=True, cwd=R)


def register_helvetica_bold():
    """matplotlib reads only the first face of Helvetica.ttc, so bold lettering silently falls back to regular.
    The bold face is extracted once into tmp/ and registered (as in fig4_learned.py); without the system font the
    figure stays regular."""
    dst = os.path.join(R, "tmp", "game_assets", "fig2_learning", "Helvetica-Bold.ttf")
    try:
        if not os.path.exists(dst):
            from fontTools.ttLib import TTCollection
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            for face in TTCollection("/System/Library/Fonts/Helvetica.ttc").fonts:
                if face["name"].getDebugName(2) == "Bold":
                    face.save(dst)
                    break
        font_manager.fontManager.addfont(dst)
    except Exception as exc:  # other platforms: keep the regular face
        print("Helvetica Bold not registered:", exc)


register_helvetica_bold()

# Palette (paper/figs/STYLE.md)
BLUE = "#2166ac"
BLUE_PILL = "#dceaf7"
BLUE_SOFT = "#eef4fb"
TERRA = "#c85228"
TERRA_PILL = "#fbe6de"
TEAL = "#1b8a6b"
AMBER = "#c98a1b"
PLUM = "#b85682"
VIOLET = "#5e4cc4"
VIOLET_PILL = "#e4e0f7"
SLATE = "#64748b"
INK = "#1f2937"
INK2 = "#475569"
ZEBRA = "#f4f6f9"
GRID = "#e8edf3"
BORDER = "#d5dde8"
OUTER_BORDER = "#e2e8f0"

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
FS_LAB = 7.4
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
STEP = 131072
MINUS = "−"


def load_rgba_clean(fn, target_h_px):
    im = Image.open(os.path.join(ASSET_DIR, fn)).convert("RGBA")
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


def add_sprite(ax, fn, xy, target_h_pt, zorder=6):
    arr, h_px = load_rgba_clean(fn, target_h_pt * 4.8)
    oi = OffsetImage(arr, zoom=target_h_pt / h_px, interpolation="bilinear", resample=True)
    ab = AnnotationBbox(oi, xy, xycoords="data", box_alignment=(0.5, 0.5), frameon=False, pad=0, zorder=zorder,
                        annotation_clip=False)
    ab.set_clip_on(False)
    ax.add_artist(ab)


def wilson(p_pct, n, z=1.96):
    p = p_pct / 100.0
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return 100.0 * max(0.0, c - h), 100.0 * min(1.0, c + h)


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.6, pad=2.0)


def draw_beta_ridge(ax, y_base, k, n, col_line, col_fill, max_h=0.52, alpha_fill=0.42, lw=0.9, zorder=3):
    """Beta(k+1, n-k+1) posterior of a win rate, drawn upwards from y_base on an inverted y axis."""
    a_p, b_p = k + 1.0, (n - k) + 1.0
    mu = a_p / (a_p + b_p)
    sd = math.sqrt(a_p * b_p / (((a_p + b_p) ** 2) * (a_p + b_p + 1.0)))
    xg = np.linspace(max(0.001, mu - 4.4 * sd), min(0.999, mu + 4.4 * sd), 180)
    dens = beta_dist.pdf(xg, a_p, b_p)
    dens = dens / dens.max() * max_h
    xp = xg * 100.0
    ax.fill_between(xp, y_base, y_base - dens, color=col_fill, alpha=alpha_fill, lw=0, zorder=zorder)
    ax.plot(xp, y_base - dens, color=col_line, lw=lw, zorder=zorder + 0.2)


def mix(c0, c1, t):
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(a + (b - a) * t)


def pct1(k, n):
    """k / n as a percentage with one decimal, rounded half up (66.35 -> 66.4, as in the manuscript)."""
    return str((Decimal(100 * k) / Decimal(n)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def round1_str(val):
    return str(Decimal(str(val)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


# =========================================================================
# Data
# =========================================================================
def load_paired_cold1115(short_key, fallback_v2, fallback_v3):
    p2 = os.path.join(R, "tmp", "cold1115", "res", f"{short_key}_v2_a10.json")
    p3 = os.path.join(R, "tmp", "cold1115", "res", f"{short_key}_v3_a10.json")
    if os.path.exists(p2) and os.path.exists(p3):
        w2 = round(100.0 * json.load(open(p2))["summary"]["all"]["win_rate"], 1)
        w3 = round(100.0 * json.load(open(p3))["summary"]["all"]["win_rate"], 1)
        return w2, w3
    return fallback_v2, fallback_v3


fresh1127 = json.load(open(os.path.join(R, "paper", "figdata", "fresh1127_5char_rv1.json")))
fresh_latest = fresh1127["fresh1105_2000"]

# Panel a: 1,000 paired seeds before / right after the update, then 2,000 fresh seeds per checkpoint (6 post-update pts).
PAIRED_1K = {
    "Ironclad": load_paired_cold1115("iron", 62.1, 52.4),
    "Defect": load_paired_cold1115("defect", 57.6, 55.1),
    "Regent": load_paired_cold1115("regent", 54.1, 52.6),
    "Silent": load_paired_cold1115("silent", 48.3, 43.2),
    "Necrobinder": load_paired_cold1115("necro", 40.6, 38.2),
}
BASE_PTS = {
    "Ironclad": (26736, [(27650, 63.05), (31750, 65.75), (39800, 69.30), (50100, 71.55), (50950, 72.90)]),
    "Defect": (16724, [(17250, 60.75), (19450, 64.95), (24450, 70.04), (29650, 72.30), (30250, 73.00)]),
    "Regent": (16778, [(17350, 58.33), (19550, 61.13), (24650, 67.58), (29900, 71.12), (30500, 71.05)]),
    "Silent": (15652, [(16400, 49.70), (19200, 57.00), (26950, 60.40), (36050, 67.35), (39000, 66.85)]),
    "Necrobinder": (13898, [(14600, 44.12), (16850, 49.10), (23850, 58.40), (32350, 64.50), (35200, 65.30)]),
}
T = {}
for ch, (fork_u, pts) in BASE_PTS.items():
    lp = fresh_latest[ch]
    T[ch] = (fork_u, PAIRED_1K[ch][0], PAIRED_1K[ch][1], pts + [(int(lp["update"]), float(lp["win_pct"]))])

FOOL = [(2750, 35.75), (7650, 47.75), (12650, 54.30)]

# Panel b: frozen v0.107.1 (name, wins, games).
V107_EXT = [("GPT-6 Astra", 26, 75), ("Spire Mentor", round(0.140 * 2500), 2500),
            ("Chen & Xie", round(0.081 * 12800), 12800)]
V107_OURS = [("Regent", 1383, 2000), ("Defect", 1327, 2000), ("Ironclad", 1181, 2000),
             ("Silent", 1083, 1999), ("Necrobinder", 1066, 2000)]

# Panel c: same seeds as AlphaDefect across all 4 released characters, ordered by delta_pct descending.
RV1 = []
for ch, r in sorted(fresh1127["rv1_paired"].items(), key=lambda kv: kv[1]["delta_pct"], reverse=True):
    RV1.append((ch, int(r["ours_wins"]), int(r["theirs_wins"]), int(r["n"]),
                float(r["delta_pct"]), float(r["z"]), abs(float(r["z"])) >= 1.96))

# Panel d: cosine between policy gradients of two disjoint batches of 131,072 decisions, and implied SNR=1 batch size.
COS_STAGES = [(0.51, "A0\n4%"), (0.22, "A0\n80%"), (0.17, "A0\n88%"), (0.06, "A10\n61%")]
COS_VALS = [c for c, _ in COS_STAGES]
SNR1_BATCH_M = [STEP * (1.0 - c) / c / 1e6 for c in COS_VALS]  # [0.126, 0.465, 0.640, 2.053]

# Panel e: 5-arm causal step-size fork from Ironclad u49300 (300 updates = 39.3M steps, 2,000 fresh1105 seeds).
kl_cat = json.load(open(os.path.join(R, "paper", "figdata", "kl_catastrophe_1127.json")))
kl_rows = sorted(kl_cat["summary_table"], key=lambda r: r["lr_kl_target"])
KL_MILLI = [r["lr_kl_target"] * 1e3 for r in kl_rows]         # [0.5, 2.0, 5.0, 10.0, 20.0]
KL_WIN = [100.0 * r["win_rate"] for r in kl_rows]             # [72.9, 67.7, 53.6, 11.25, 5.1]
KL_WIN_STR = [pct1(int(r["wins"]), int(r["n"])) for r in kl_rows]
KL_CI_LO = [r["ci95_pct"][0] for r in kl_rows]
KL_CI_HI = [r["ci95_pct"][1] for r in kl_rows]
KL_DW = [r["end_u49600_delta_w_l2"] for r in kl_rows]         # [0.97, 2.74, 7.06, 18.91, 21.95]
PPO_BAND_MILLI = (10.0, 20.0)

# Panel f: plasticity measurements across 9 networks (24,576 states across the 6 characters).
lin_rows = list(csv.DictReader(open(os.path.join(R, "docs", "results", "plasticity", "lineage_summary.csv"))))
mc_data = json.load(open(os.path.join(R, "paper", "figdata", "plast_multichar_fig2.json")))
mc_chars = mc_data["characters_latest"]
mc_base = mc_data["baselines"]
_mc_sum = {r["character"]: r for r in json.load(open(os.path.join(R, "paper", "figdata", "multichar_plasticity_1125.json")))["summary_table"]}
RAND_MSE_IRON = mc_base["random_init"]["probe"]["memory"]["final_mse"]
RAND_MSE_AGED_BANK = float(next(r for r in lin_rows if r["ckpt"] == "random")["probe_memory_1000"])


def rand_mse_for(char, rkind):
    if rkind == "aged":
        return RAND_MSE_AGED_BANK
    if rkind == "char":
        return _mc_sum[char]["random_memory_probe_mse_1000"]
    return RAND_MSE_IRON


# =========================================================================
# Canvas (8.6 x 6.0 in, 6 panels a-f)
# =========================================================================
FIG_W, FIG_H = 8.6, 6.0
PT_W, PT_H = FIG_W * 72.0, FIG_H * 72.0
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
fig.add_artist(FancyBboxPatch(
    (0.004, 0.006), 0.992, 0.988, transform=fig.transFigure,
    boxstyle="round,pad=0.0,rounding_size=0.012",
    facecolor="none", edgecolor=OUTER_BORDER, linewidth=0.8, zorder=0
))

TOP_Y0, TOP_H = 0.578, 0.384
BOT_Y0, BOT_H = 0.088, 0.394

A_BOX = [0.056, TOP_Y0, 0.338, TOP_H - 0.052]
A_TOP = [0.056, TOP_Y0 + TOP_H - 0.044, 0.338, 0.044]
B_BOX = [0.508, TOP_Y0, 0.220, TOP_H]
C_BOX = [0.784, TOP_Y0, 0.204, TOP_H]

D1_BOX = [0.064, BOT_Y0 + 0.210, 0.174, 0.184]
D2_BOX = [0.064, BOT_Y0, 0.174, 0.184]
E_BOX = [0.302, BOT_Y0, 0.232, BOT_H - 0.052]
E_TOP = [0.302, BOT_Y0 + BOT_H - 0.044, 0.232, 0.044]
F_BOX = [0.556, 0.030, 0.432, 0.464]

ax_a = fig.add_axes(A_BOX)
ax_a_top = fig.add_axes(A_TOP)
ax_b = fig.add_axes(B_BOX)
ax_c = fig.add_axes(C_BOX)
ax_d1 = fig.add_axes(D1_BOX)
ax_d2 = fig.add_axes(D2_BOX)
ax_e = fig.add_axes(E_BOX)
ax_e_top = fig.add_axes(E_TOP)
ax_f = fig.add_axes(F_BOX)

# =========================================================================
# Panel a: six characters across v0.111 -> v0.113 and beyond
# =========================================================================
XA0, XA1 = -0.78, 4.46
YA0, YA1 = 32.0, 81.5
X_BEFORE = -0.40

mean_before = float(np.mean([T[c][1] for c in T]))
mean_cold = float(np.mean([T[c][2] for c in T]))
mean_adapted = float(np.mean([T[c][3][-1][1] for c in T]))

ax_a_top.set_xlim(XA0, XA1)
ax_a_top.set_ylim(0, 1)
ax_a_top.axis("off")
ax_a_top.add_patch(Rectangle((XA0, 0.06), -0.04 - XA0, 0.88, facecolor="#e9edf2", edgecolor="none"))
grad = np.linspace(0, 1, 256)[None, :]
ax_a_top.imshow(grad, extent=[0.04, XA1, 0.06, 0.94], aspect="auto", zorder=1,
                cmap=LinearSegmentedColormap.from_list("strip", [TERRA_PILL, "#f3ecee", BLUE_PILL, BLUE_PILL]))
ax_a_top.text((XA0 - 0.04) / 2.0, 0.48, f"{mean_before:.1f}%", fontsize=7.8, fontweight="bold", color=INK,
              ha="center", va="center", zorder=3)
ax_a_top.text(0.12, 0.48, f"{mean_cold:.1f}%", fontsize=8.0, fontweight="bold", color=TERRA,
              ha="left", va="center", zorder=3)
ax_a_top.text(XA1 - 0.12, 0.48, f"{mean_adapted:.1f}%", fontsize=8.0, fontweight="bold", color=BLUE,
              ha="right", va="center", zorder=3)
ax_a_top.annotate("", xy=(XA1 - 0.92, 0.5), xytext=(0.94, 0.5),
                  arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45", lw=0.9, color=INK2), zorder=3)
ax_a_top.text(2.25, 0.5, "mean of five", fontsize=FS_LAB, fontweight="bold", color=INK, ha="center", va="center",
              zorder=4, bbox=dict(boxstyle="round,pad=0.25", facecolor="#f1eef2", edgecolor="none"))

clean_ax(ax_a)
ax_a.axvspan(XA0, 0.0, color="#eef1f5", zorder=0)
ax_a.axvline(0.0, color=INK2, lw=0.8, zorder=2)
for gy in (40, 50, 60, 70, 80):
    ax_a.axhline(gy, color=GRID, lw=0.55, zorder=0.5)
ax_a.text(XA0 / 2.0, 80.6, "v0.111", fontsize=FS_LAB, fontweight="bold", color=INK, ha="center", va="top")
ax_a.text(0.10, 80.6, "v0.113", fontsize=FS_LAB, fontweight="bold", color=INK, ha="left", va="top")

for ch, (fork, before, cold, pts) in T.items():
    col = CHAR_COLORS[ch]
    xs = [0.0] + [(u - fork) * STEP / 1e9 for u, _ in pts]
    ys = [cold] + [w for _, w in pts]
    ns = [1000] + [2000] * len(pts)
    los = [wilson(w, n)[0] for w, n in zip(ys, ns)]
    his = [wilson(w, n)[1] for w, n in zip(ys, ns)]
    ax_a.fill_between(xs, los, his, color=col, alpha=0.10, lw=0, zorder=1)
    ax_a.plot([X_BEFORE, 0.0], [before, cold], color=col, lw=1.1, ls=(0, (2.2, 1.5)), zorder=2)
    ax_a.plot(xs, ys, color=col, lw=1.65, zorder=3)
    ax_a.scatter([X_BEFORE], [before], s=17, color=col, marker="s", zorder=4, edgecolor="white", linewidth=0.5)
    ax_a.scatter(xs[:-1], ys[:-1], s=16, color=col, zorder=4, edgecolor="white", linewidth=0.5)
    ax_a.scatter([xs[-1]], [ys[-1]], s=24, color=col, zorder=5, edgecolor="white", linewidth=0.6)

fx = [u * STEP / 1e9 for u, _ in FOOL]
fy = [w for _, w in FOOL]
ax_a.fill_between(fx, [wilson(w, 2000)[0] for w in fy], [wilson(w, 2000)[1] for w in fy],
                  color=VIOLET, alpha=0.11, lw=0, zorder=1)
ax_a.plot(fx, fy, color=VIOLET, lw=1.65, ls=(0, (4, 1.5)), zorder=3)
ax_a.scatter(fx[:-1], fy[:-1], s=16, color=VIOLET, zorder=4, edgecolor="white", linewidth=0.5)
ax_a.scatter([fx[-1]], [fy[-1]], s=24, color=VIOLET, zorder=5, edgecolor="white", linewidth=0.6)

# Single crisp 2-column 6-character badge+name key in the open lower-right area (no duplicate line-end badges)
KEY = [["Ironclad", "Silent", "Regent"], ["Defect", "Necrobinder", "Fool"]]
KEY_X = [1.72, 3.02]
KEY_Y = [46.6, 40.9, 35.2]
for ci, col_names in enumerate(KEY):
    for ri, ch in enumerate(col_names):
        add_sprite(ax_a, CHAR_BADGES[ch], (KEY_X[ci], KEY_Y[ri]), target_h_pt=16.5, zorder=6)
        ax_a.text(KEY_X[ci] + 0.25, KEY_Y[ri], ch + (", new" if ch == "Fool" else ""), fontsize=FS_LAB,
                  fontweight="bold", color=CHAR_COLORS[ch], ha="left", va="center", zorder=6)

ax_a.set_xlim(XA0, XA1)
ax_a.set_ylim(YA0, YA1)
ax_a.set_xticks([0, 1, 2, 3, 4])
ax_a.set_yticks([40, 50, 60, 70, 80])
ax_a.set_xlabel("Environment steps after the update (billions)")
ax_a.set_ylabel("Win rate at Ascension 10 (%)")

# =========================================================================
# Panel b: frozen v0.107.1, posterior ridges with value columns
# =========================================================================
B_W_PT = B_BOX[2] * PT_W           # 1 x unit = 1 pt
XB_PLOT = 85.0
XB_C1 = XB_PLOT + 14.5
XB_C2 = B_W_PT - 12.5
XB_LEFT = -(B_BOX[0] - 0.408) * PT_W
YB_HEAD = -0.28
clean_ax(ax_b)
ax_b.spines["left"].set_visible(False)

ROWS_B = [(0.5 + i, "ext", r) for i, r in enumerate(V107_EXT)] + [(3.72 + i, "ours", r) for i, r in enumerate(V107_OURS)]
YB_BOTTOM = 8.26
mean_107 = float(np.mean([100.0 * k / n for _, k, n in V107_OURS]))
rate_cmap = LinearSegmentedColormap.from_list("rate", ["#f3f6fa", "#cfe1f3", "#8fb9e3"])

ax_b.text(XB_C1 - 1.0, YB_HEAD, "Win %", fontsize=FS_LAB, fontweight="bold", color=INK, ha="center", va="center")
ax_b.text(XB_C2 + 1.5, YB_HEAD, "Games", fontsize=FS_LAB, fontweight="bold", color=INK, ha="center", va="center")
ax_b.plot([XB_LEFT, B_W_PT], [0.0, 0.0], color=BORDER, lw=0.8, clip_on=False, zorder=1)
ax_b.plot([XB_LEFT, B_W_PT], [3.11, 3.11], color="#94a3b8", lw=0.8, clip_on=False, zorder=2)

for ry, kind, (name, k, n) in ROWS_B:
    idx = int(ry)
    if idx % 2 == 0:
        ax_b.add_patch(Rectangle((XB_LEFT, ry - 0.5), XB_PLOT - XB_LEFT + 0.8, 1.0, facecolor=ZEBRA,
                                 edgecolor="none", zorder=0, clip_on=False))
    pct = 100.0 * k / n
    lo, hi = wilson(pct, n)
    base = ry + 0.34
    if kind == "ext":
        col, fill = INK2, "#b8c2cf"
        ax_b.text(-4.0, ry, name, fontsize=FS_LAB, color=INK, va="center", ha="right", clip_on=False)
        draw_beta_ridge(ax_b, base, k, n, col, fill, max_h=0.70, alpha_fill=0.75)
        ax_b.plot([lo, hi], [base, base], color=col, lw=1.1, zorder=4)
        ax_b.scatter([pct], [base], s=20, marker="D", color=col, edgecolor="white", linewidth=0.5, zorder=5)
    else:
        col = CHAR_COLORS[name]
        add_sprite(ax_b, CHAR_BADGES[name], (-11.0, ry), target_h_pt=16.0)
        ax_b.text(-21.5, ry, name, fontsize=FS_LAB, color=INK, va="center", ha="right", clip_on=False)
        draw_beta_ridge(ax_b, base, k, n, col, col, max_h=0.70, alpha_fill=0.40)
        ax_b.plot([lo, hi], [base, base], color=col, lw=1.1, zorder=4)
        ax_b.scatter([pct], [base], s=22, color=col, edgecolor="white", linewidth=0.5, zorder=5)
    ax_b.add_patch(Rectangle((XB_PLOT + 1.6, ry - 0.46), 2 * (XB_C1 - XB_PLOT - 1.6), 0.92,
                             facecolor=rate_cmap(min(1.0, pct / 70.0)), edgecolor="white", linewidth=0.8, zorder=2))
    ax_b.text(XB_C1, ry, pct1(k, n), fontsize=FS_CELL + 0.4, fontweight="bold", color=INK, ha="center", va="center",
              zorder=3)
    ax_b.text(XB_C2, ry, f"{n:,}", fontsize=FS_CELL, color=INK, ha="center", va="center", zorder=3)

# Clipped to span ONLY our 5 rows (y = 3.11 .. YB_BOTTOM)
ax_b.plot([mean_107, mean_107], [3.11, YB_BOTTOM], color=BLUE, lw=1.0, ls=(0, (2.6, 1.8)), zorder=2.5)
ax_b.text(mean_107 - 3.5, 4.18, f"our mean\n{mean_107:.1f}%", fontsize=FS_LAB + 0.3, fontweight="bold",
          color=BLUE, ha="right", va="center", linespacing=1.08, zorder=5)
ax_b.plot([0, 0], [0.0, YB_BOTTOM], color=INK2, lw=0.6, zorder=3)

ax_b.set_xlim(0, B_W_PT)
ax_b.set_ylim(YB_BOTTOM, -0.62)
ax_b.spines["bottom"].set_bounds(0, 80)
ax_b.set_xticks([0, 20, 40, 60, 80])
ax_b.set_yticks([])
ax_b.set_xlabel("Win rate on frozen v0.107.1 (%)", x=0.5 * XB_PLOT / B_W_PT - 0.06)

# =========================================================================
# Panel c: same-seed RV1 comparison with AlphaDefect (all 4 released chars)
# =========================================================================
C_W_PT = C_BOX[2] * PT_W
XC0, XC_PLOT = 55.5, 83.2
C_PT_PER = 3.45
XC1 = XC0 + C_W_PT / C_PT_PER
XC_COL = (XC_PLOT + XC1) / 2.0 + 0.15
XC_LEFT = XC0 - (C_BOX[0] - 0.742) * PT_W / C_PT_PER
YC_BOTTOM = float(len(RV1))  # 4.0 rows
clean_ax(ax_c)
ax_c.spines["left"].set_visible(False)

YC_HEAD = YB_HEAD * (YC_BOTTOM / YB_BOTTOM)
ax_c.scatter([XC0 + 1.5], [YC_HEAD], s=22, color=INK, edgecolor="white", linewidth=0.5, zorder=5, clip_on=False)
ax_c.text(XC0 + 3.0, YC_HEAD, "ours", fontsize=FS_LAB, fontweight="bold", color=INK, ha="left", va="center")
ax_c.scatter([XC0 + 10.8], [YC_HEAD], s=22, facecolor="white", edgecolor=VIOLET, linewidth=1.2, zorder=5,
             clip_on=False)
ax_c.text(XC0 + 12.3, YC_HEAD, "AlphaDefect", fontsize=FS_LAB, fontweight="bold", color=VIOLET, ha="left",
          va="center")
ax_c.text(XC_COL, YC_HEAD, "Δ", fontsize=FS_LAB + 0.6, fontweight="bold", color=INK, ha="center", va="center")
ax_c.plot([XC_LEFT, XC1], [0.0, 0.0], color=BORDER, lw=0.8, clip_on=False, zorder=1)

for i, (ch, ok, tk, n, d_pct, z_val, sig) in enumerate(RV1):
    ry = i + 0.5
    if i % 2 == 0:
        ax_c.add_patch(Rectangle((XC_LEFT, ry - 0.5), XC_PLOT - XC_LEFT + 0.3, 1.0, facecolor=ZEBRA,
                                 edgecolor="none", zorder=0, clip_on=False))
    col = CHAR_COLORS[ch]
    opct, tpct = 100.0 * ok / n, 100.0 * tk / n
    ostr, tstr = pct1(ok, n), pct1(tk, n)
    base = i + 0.56
    add_sprite(ax_c, CHAR_BADGES[ch], (XC0 - 13.0 / C_PT_PER, ry), target_h_pt=19.0)
    draw_beta_ridge(ax_c, base, tk, n, VIOLET, VIOLET, max_h=0.45, alpha_fill=0.20, lw=0.9, zorder=2)
    draw_beta_ridge(ax_c, base, ok, n, col, col, max_h=0.50, alpha_fill=0.42, lw=1.0, zorder=3)
    ax_c.plot([tpct, opct], [base, base], color=INK, lw=1.25, zorder=4)
    ax_c.scatter([tpct], [base], s=25, facecolor="white", edgecolor=VIOLET, linewidth=1.25, zorder=5)
    ax_c.scatter([opct], [base], s=25, color=col, edgecolor="white", linewidth=0.55, zorder=6)
    # Non-colliding left/right labels cleanly offset below the baseline
    lo_is_ours = opct < tpct
    pad_x = 0.70 if abs(opct - tpct) < 2.5 else 0.50
    for val, s_val, is_ours in ((opct, ostr, True), (tpct, tstr, False)):
        left_side = (is_ours == lo_is_ours)
        dx = -pad_x if left_side else pad_x
        ax_c.text(val + dx, i + 0.80, s_val, fontsize=FS_LAB - 0.1, fontweight="bold",
                  color=(col if is_ours else VIOLET), ha="right" if left_side else "left", va="center", zorder=6)
    d_txt = f"+{d_pct:.1f}" if d_pct >= 0 else f"{MINUS}{abs(d_pct):.1f}"
    pw = XC1 - XC_PLOT - 0.9
    ax_c.add_patch(Rectangle((XC_COL - pw / 2.0, ry - 0.45), pw, 0.90,
                             facecolor=BLUE_PILL if sig else "#eef1f5", edgecolor="white", linewidth=0.8, zorder=2))
    ax_c.text(XC_COL, ry, d_txt, fontsize=FS_CELL + 0.5, fontweight="bold" if sig else "normal",
              color=BLUE if sig else INK, ha="center", va="center", zorder=3)

for gx in (60, 70, 80):
    ax_c.plot([gx, gx], [0.0, YC_BOTTOM], color=GRID, lw=0.55, zorder=0.5)
ax_c.plot([XC0, XC0], [0.0, YC_BOTTOM], color=INK2, lw=0.6, zorder=3)
ax_c.set_xlim(XC0, XC1)
ax_c.set_ylim(YC_BOTTOM, -0.62 * (YC_BOTTOM / YB_BOTTOM))
ax_c.spines["bottom"].set_bounds(XC0, XC_PLOT)
ax_c.set_xticks([60, 70, 80])
ax_c.set_yticks([])
ax_c.set_xlabel("Win rate on the same seeds (%)", x=0.40)

# =========================================================================
# Panel d: inter-batch policy-gradient cosine & implied SNR=1 batch size
# =========================================================================
clean_ax(ax_d1)
clean_ax(ax_d2)
xd = np.arange(len(COS_STAGES))

# Top sub-panel: Gradient cosine (0.51 -> 0.06)
for gy in (0.2, 0.4):
    ax_d1.axhline(gy, color=GRID, lw=0.55, zorder=0)
bars_d1 = ax_d1.bar(xd, COS_VALS, width=0.52, color=BLUE, alpha=0.86, edgecolor="white", linewidth=0.6, zorder=3)
for xi, c_val in zip(xd, COS_VALS):
    ax_d1.text(xi, c_val + 0.024, f"{c_val:.2f}", fontsize=FS_LAB, fontweight="bold", color=BLUE,
               ha="center", va="bottom", zorder=5)
add_sprite(ax_d1, CHAR_BADGES["Ironclad"], (2.82, 0.47), target_h_pt=18.0, zorder=6)
ax_d1.set_xlim(-0.55, 3.55)
ax_d1.set_ylim(0.0, 0.64)
ax_d1.set_xticks(xd)
ax_d1.set_xticklabels([])
ax_d1.set_yticks([0.0, 0.2, 0.4, 0.6])
ax_d1.set_ylabel("Agreement of\ntwo batches", labelpad=2.5, linespacing=1.0)

# Bottom sub-panel: Implied batch size for SNR = 1 (0.13M -> 2.05M decisions)
for gy in (1.0, 2.0):
    ax_d2.axhline(gy, color=GRID, lw=0.55, zorder=0)
ax_d2.axhline(STEP / 1e6, color=SLATE, lw=0.75, ls=(0, (2.2, 1.6)), zorder=1.5)
ax_d2.fill_between(xd, 0, SNR1_BATCH_M, color=TERRA, alpha=0.14, lw=0, zorder=2)
ax_d2.plot(xd, SNR1_BATCH_M, "o-", color=TERRA, lw=1.75, ms=5.2, mec="white", mew=0.6, zorder=4)
snr_offsets = [(0.08, 0.12), (-0.08, 0.13), (-0.20, 0.14), (-0.06, 0.12)]
for (xi, b_val), (dx, dy) in zip(enumerate(SNR1_BATCH_M), snr_offsets):
    ax_d2.text(xi + dx, b_val + dy, f"{b_val:.2f}M", fontsize=FS_CELL + 0.2, fontweight="bold", color=TERRA,
               ha="center", va="bottom", zorder=5)
ax_d2.text(3.45, STEP / 1e6 + 0.06, "1 batch", fontsize=FS_TICK, fontweight="bold", color=SLATE,
           ha="right", va="bottom", zorder=4)
ax_d2.set_xlim(-0.55, 3.55)
ax_d2.set_ylim(0.0, 2.58)
ax_d2.set_xticks(xd)
ax_d2.set_xticklabels([lbl for _, lbl in COS_STAGES], linespacing=0.95)
ax_d2.set_yticks([0, 1, 2])
ax_d2.set_ylabel("Decisions needed\n(millions)", labelpad=2.5, linespacing=1.0)
ax_d2.set_xlabel("Ironclad training stage", labelpad=2.5)
ax_d1.yaxis.set_label_coords(-0.165, 0.5)
ax_d2.yaxis.set_label_coords(-0.165, 0.5)

# =========================================================================
# Panel e: step-size catastrophe at ~73% A10 (kl_catastrophe_1127.json)
# =========================================================================
XE0, XE1 = 0.32, 32.0

# Top aligned strip: parameter displacement ||Δw||_2 over 39.3M steps (1.0 -> 22.0)
ax_e_top.set_xscale("log")
ax_e_top.set_xlim(XE0, XE1)
ax_e_top.set_ylim(0, 1)
ax_e_top.axis("off")
dw_cmap = LinearSegmentedColormap.from_list("dw", [BLUE_PILL, "#f4ede6", "#f3bfa8", "#d96b43"])
ax_e_top.text(XE0 * 0.94, 0.50, "Weight\nchange", fontsize=FS_CELL + 0.3, fontweight="bold", color=INK,
              ha="right", va="center", linespacing=1.0, clip_on=False)
for xm, dw in zip(KL_MILLI, KL_DW):
    t_dw = min(1.0, max(0.0, (dw - 1.0) / (22.0 - 1.0)))
    x_lo, x_hi = xm / 1.35, xm * 1.35
    ax_e_top.add_patch(Rectangle((x_lo, 0.08), x_hi - x_lo, 0.84, facecolor=dw_cmap(t_dw),
                                 edgecolor="white", linewidth=0.7, zorder=2))
    ax_e_top.text(xm, 0.50, f"{dw:.1f}", fontsize=FS_CELL, fontweight="bold" if t_dw > 0.45 else "normal",
                  color="white" if t_dw > 0.75 else INK, ha="center", va="center", zorder=3)

clean_ax(ax_e)
for gy in (20, 40, 60, 80):
    ax_e.axhline(gy, color=GRID, lw=0.55, zorder=0)

# Usual PPO band [10, 20] x 10^-3
ax_e.axvspan(PPO_BAND_MILLI[0], PPO_BAND_MILLI[1], color=TERRA, alpha=0.15, lw=0, zorder=0.6)
ax_e.text(math.sqrt(PPO_BAND_MILLI[0] * PPO_BAND_MILLI[1]), 48.0, "usual PPO", fontsize=FS_LAB,
          fontweight="bold", color=TERRA, rotation=90, ha="center", va="center", zorder=2)

# Control reference line at 72.9%
ax_e.axhline(KL_WIN[0], color=BLUE, lw=0.75, ls=(0, (2.4, 1.8)), alpha=0.55, zorder=1)

# Trajectory & 95% Wilson CIs
ax_e.fill_between(KL_MILLI, KL_CI_LO, KL_CI_HI, color=TERRA, alpha=0.10, lw=0, zorder=1.2)
ax_e.plot(KL_MILLI, KL_WIN, "-", color=INK2, lw=1.55, zorder=2.5)

pt_cols = [BLUE, mix(BLUE, TERRA, 0.42), TERRA, TERRA, TERRA]
for idx_k, (xm, wy, w_s, clo, chi, pcol) in enumerate(zip(KL_MILLI, KL_WIN, KL_WIN_STR, KL_CI_LO, KL_CI_HI, pt_cols)):
    ax_e.errorbar([xm], [wy], yerr=[[wy - clo], [chi - wy]], fmt="none", ecolor=pcol, elinewidth=1.15,
                  capsize=2.6, capthick=1.1, zorder=3.5)
    ax_e.scatter([xm], [wy], s=32, color=pcol, edgecolor="white", linewidth=0.65, zorder=4.5)

# Non-overlapping value annotations (none crossed by trajectory lines)
ax_e.text(KL_MILLI[0] * 1.04, KL_CI_HI[0] + 1.8, f"{KL_WIN_STR[0]}%", fontsize=FS_LAB + 0.3, fontweight="bold",
          color=BLUE, ha="center", va="bottom", zorder=5)
ax_e.text(KL_MILLI[0] * 1.02, KL_CI_LO[0] - 2.6, "ours (1×)", fontsize=FS_TICK, fontweight="bold",
          color=BLUE, ha="left", va="top", zorder=5)
ax_e.text(KL_MILLI[1], KL_CI_LO[1] - 2.2, f"{KL_WIN_STR[1]}%", fontsize=FS_CELL + 0.2, fontweight="bold",
          color=pt_cols[1], ha="center", va="top", zorder=5)
ax_e.text(KL_MILLI[2] * 0.84, KL_WIN[2], f"{KL_WIN_STR[2]}%", fontsize=FS_CELL + 0.2, fontweight="bold",
          color=TERRA, ha="right", va="center", zorder=5)
ax_e.text(KL_MILLI[3] * 0.84, KL_WIN[3] - 0.3, f"{KL_WIN_STR[3]}%", fontsize=FS_LAB + 0.2, fontweight="bold",
          color=TERRA, ha="right", va="center", zorder=5)
ax_e.text(KL_MILLI[4], KL_CI_HI[4] + 1.8, f"{KL_WIN_STR[4]}%", fontsize=FS_LAB + 0.2, fontweight="bold",
          color=TERRA, ha="center", va="bottom", zorder=5)

add_sprite(ax_e, CHAR_BADGES["Ironclad"], (0.62, 16.0), target_h_pt=18.5, zorder=6)
ax_e.text(1.05, 16.0, "after 39M\nsteps", fontsize=FS_LAB, fontweight="bold", color=INK,
          ha="left", va="center", linespacing=1.05, zorder=5)

ax_e.set_xscale("log")
ax_e.set_xlim(XE0, XE1)
ax_e.set_ylim(0.0, 84.5)
ax_e.set_xticks([0.5, 2, 5, 10, 20])
ax_e.set_xticklabels(["1", "4", "10", "20", "40"])  # step size relative to ours (0.5e-3)
ax_e.minorticks_off()
ax_e.set_yticks([0, 20, 40, 60, 80])
ax_e.set_xlabel("Step size (× ours)", labelpad=2.5)
ax_e.set_ylabel("Win rate at Ascension 10 (%)", labelpad=2.0)

# =========================================================================
# Panel f: plasticity matrix as a 9-row heat-map table
# =========================================================================
F_W_PT = F_BOX[2] * PT_W
ax_f.axis("off")
ax_f.set_xlim(0, F_W_PT)
YF_TOP, YF_BOTTOM = -1.42, 9.20
ax_f.set_ylim(YF_BOTTOM, YF_TOP)

tbl_rows = [
    ("Large steps", None, mc_base["aged_large_step"], "aged", f"{mc_base['aged_large_step']['env_steps_B']:.2f}B"),
    ("Untrained", None, mc_base["random_init"], "rand", "0"),
    ("Freshly distilled", None, mc_base["young_kd"], "kd", "0.2B"),
    (None, "Ironclad", mc_chars["Ironclad"], "char", f"{mc_chars['Ironclad']['env_steps_B']:.1f}B"),
    (None, "Silent", mc_chars["Silent"], "char", f"{mc_chars['Silent']['env_steps_B']:.1f}B"),
    (None, "Necrobinder", mc_chars["Necrobinder"], "char", f"{mc_chars['Necrobinder']['env_steps_B']:.1f}B"),
    (None, "Regent", mc_chars["Regent"], "char", f"{mc_chars['Regent']['env_steps_B']:.1f}B"),
    (None, "Defect", mc_chars["Defect"], "char", f"{mc_chars['Defect']['env_steps_B']:.1f}B"),
    (None, "Fool", mc_chars["Fool"], "char", f"{mc_chars['Fool']['env_steps_B']:.1f}B"),
]
mod_keys = ["blocks", "cand_blocks", "logit", "value"]
col_w = [21.5, 21.5, 21.5, 21.5, 28.0, 26.0, 28.5]
X_CELLS = F_W_PT - sum(col_w)
X_STEPS = X_CELLS - 3.5            # right edge of the step-count column
col_x = [X_CELLS + sum(col_w[:j]) for j in range(len(col_w))]
health = LinearSegmentedColormap.from_list("health", [(0.0, "#cfeadf"), (0.30, "#f6efdc"), (0.62, "#f0a988"),
                                                      (1.0, "#bf4a22")])
AGED_NORM = mc_base["aged_large_step"]["weight_norms"]["non_table"]
RAND_NORM = mc_base["random_init"]["weight_norms"]["non_table"]
RAND_H = mc_base["random_init"]["attention_entropy_mean"]
rel_all = []

# header
x_d0, x_d1 = col_x[0], col_x[3] + col_w[3]
ax_f.text((x_d0 + x_d1) / 2.0, -1.06, "Dormant units (%)", fontsize=FS_LAB, fontweight="bold", color=INK,
          ha="center", va="center")
ax_f.plot([x_d0 + 2.0, x_d1 - 2.0], [-0.72, -0.72], color=INK2, lw=0.7)
for j, t in enumerate(["Trunk", "Pointer", "Policy", "Value"]):
    ax_f.text(col_x[j] + col_w[j] / 2.0, -0.34, t, fontsize=FS_CELL, color=INK, ha="center", va="center")
for j, t in zip((4, 5, 6), ("Attn.\nentropy", "Weight\nnorm", "Fitting\nerror")):
    ax_f.text(col_x[j] + col_w[j] / 2.0, -0.62, t, fontsize=FS_LAB - 0.3, fontweight="bold", color=INK, ha="center",
              va="center", linespacing=1.05)

# colour key in the header corner, step-count column header next to it
key_x0, key_w = 2.0, 66.0
ax_f.imshow(np.linspace(0, 1, 200)[None, :], extent=[key_x0, key_x0 + key_w, -0.66, -0.90], aspect="auto",
            cmap=health, zorder=2)
ax_f.text(key_x0, -1.14, "healthy", fontsize=FS_TICK, color=TEAL, fontweight="bold", ha="left", va="center")
ax_f.text(key_x0 + key_w, -1.14, "aged", fontsize=FS_TICK, color=TERRA, fontweight="bold", ha="right",
          va="center")
ax_f.text(X_STEPS, -0.34, "Steps", fontsize=FS_CELL, fontweight="bold", color=INK, ha="right", va="center")

for idx_r, (rlab, ch_k, cobj, rkind, step_str) in enumerate(tbl_rows):
    ry = idx_r + 0.5 + (0.16 if idx_r >= 3 else 0.0)
    if ch_k is not None:
        add_sprite(ax_f, CHAR_BADGES[ch_k], (10.5, ry), target_h_pt=16.5, zorder=5)
        ax_f.text(21.5, ry, ch_k, fontsize=FS_CELL + 0.2, color=INK, va="center", ha="left")
        ax_f.text(X_STEPS, ry, step_str, fontsize=FS_LAB, fontweight="bold",
                  color=CHAR_COLORS[ch_k], va="center", ha="right")
    else:
        ax_f.text(2.0, ry, rlab, fontsize=FS_LAB, fontweight="bold" if rkind == "aged" else "normal",
                  color=TERRA if rkind == "aged" else INK, va="center", ha="left")
        ax_f.text(X_STEPS, ry, step_str, fontsize=FS_LAB if rkind == "aged" else FS_CELL + 0.2,
                  fontweight="bold" if rkind == "aged" else "normal",
                  color=TERRA if rkind == "aged" else INK2, va="center", ha="right")
    mbk = cobj["dormant"]["mean_by_kind_01"]
    ah = cobj["attention_entropy_mean"]
    wn = cobj["weight_norms"]["non_table"]
    rel = cobj["probe"]["memory"]["final_mse"] / rand_mse_for(ch_k, rkind)
    rel_all.append(rel)
    cells = [(f"{100.0 * mbk[mk]:.1f}", min(1.0, 100.0 * mbk[mk] / 60.0)) for mk in mod_keys]
    cells.append((f"{ah:.2f}", min(1.0, max(0.0, 1.0 - ah / RAND_H))))
    cells.append((f"{wn:.0f}", min(1.0, max(0.0, (wn - RAND_NORM) / (AGED_NORM - RAND_NORM)))))
    rel_t = min(1.0, max(0.0, math.log(rel / 0.2) / math.log(15.0 / 0.2)))
    cells.append(((f"{rel:.1f}" if rel >= 10 else f"{rel:.2f}") + "×", rel_t))
    for j, (txt, t) in enumerate(cells):
        ax_f.add_patch(Rectangle((col_x[j], ry - 0.5), col_w[j], 1.0, facecolor=health(t), edgecolor="white",
                                 linewidth=1.0, zorder=2))
        ax_f.text(col_x[j] + col_w[j] / 2.0, ry, txt, fontsize=FS_CELL, fontweight="bold" if t > 0.5 else "normal",
                  color="white" if t > 0.72 else INK, ha="center", va="center", zorder=3)
ax_f.plot([0.0, F_W_PT], [3.08, 3.08], color="#94a3b8", lw=0.8, zorder=4)

# =========================================================================
# Panel letters (a-f)
# =========================================================================
for lab, (gx, gy) in zip("abcdef", [
    (0.012, 0.984), (0.408, 0.984), (0.742, 0.984),
    (0.012, 0.512), (0.252, 0.512), (0.548, 0.512),
]):
    fig.text(gx, gy, lab, fontsize=FS_LETTER, fontweight="bold", va="top", color=INK)

for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(R, "paper", "figs", f"fig2_learning.{ext}"), dpi=350)
print("written paper/figs/fig2_learning.{png,pdf,svg}")
print(f"a: mean before {mean_before:.1f}%, right after {mean_cold:.1f}%, adapted {mean_adapted:.1f}%")
print("a latest 5-char:", {ch: T[ch][3][-1] for ch in T})
print(f"b: mean {mean_107:.1f}%")
print("c RV1:", [(ch, pct1(ok, n), pct1(tk, n), f"{d:+.1f}", f"z={z:+.2f}", sig) for ch, ok, tk, n, d, z, sig in RV1])
print("d SNR=1 batch (M):", [round(x, 2) for x in SNR1_BATCH_M])
print("e KL catastrophe:", list(zip(KL_MILLI, KL_WIN_STR, [round(x, 1) for x in KL_DW])))
print("f probe error relative to random init:", ", ".join(f"{r:.2f}" for r in rel_all))
