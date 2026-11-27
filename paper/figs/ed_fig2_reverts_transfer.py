"""Extended Data Figure 2: Silent v0.113 78-rule knockout profile, zero-shot backward evaluation on v0.107.1,
and new-character curriculum warm-start transfer.

  a, Silent v0.113 78-rule counterfactual knockout on n = 2,000 paired seeds (baseline 42.95%: Phantom Blades +1.45,
     Speedster +0.15, Memento Mori -0.05, Expertise -0.70, all 4 Silent cards +0.00, all 78 changes +1.20 against the
     -5.1-pt cold-start drop), aligned with pre-patch pick and deck shares (Phantom Blades 83.0% vs Memento Mori 2.4%).
  b, Zero-shot backward evaluation on frozen v0.107.1 (2,000 seeds per character): adapted v0.113 networks (60.4% mean)
     versus earlier v0.111 final networks (47.0% mean).
  c, Early Ascension 0 curriculum transfer onto the newly added Fool: 5-character joint init (25.4% zero-shot) versus
     single-character Ironclad init (12.4% zero-shot) versus random init (0%), converging by 39M steps.

Writes paper/figs/ed_fig2_reverts_transfer.{png,pdf,svg} (350 DPI).
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
from scipy.stats import beta as beta_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_V2_DIR = os.path.join(R, "tmp", "game_assets", "fig14_v2")
ASSET_ED_DIR = os.path.join(R, "tmp", "game_assets", "ed_figs")
ASSET_W_DIR = os.path.join(R, "tmp", "game_assets", "fig1_world")

if not os.path.exists(os.path.join(ASSET_V2_DIR, "badge_char_ironclad.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_V2_DIR, "build_assets.py")], check=True, cwd=R)
if not os.path.exists(os.path.join(ASSET_ED_DIR, "card_memento_mori.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_ED_DIR, "build_assets.py")], check=True, cwd=R)


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
AMBER = "#c98a1b"
AMBER_PILL = "#f9edd2"
VIOLET = "#5e4cc4"
VIOLET_PILL = "#e4e0f7"
PLUM = "#b85682"
SLATE = "#64748b"
SLATE_PILL = "#e9eef4"
INK = "#1f2937"
INK2 = "#475569"
ZEBRA = "#f4f6f9"
GRID = "#e8edf3"
BORDER = "#d5dde8"
OUTER_BORDER = "#e2e8f0"
MINUS = "−"

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


def load_rgba_clean(path, target_h_px):
    im = Image.open(path).convert("RGBA")
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


def add_sprite_path(ax, path, xy, target_h_pt, xycoords="data", align=(0.5, 0.5), zorder=6):
    arr, h_px = load_rgba_clean(path, target_h_pt * 4.8)
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


def draw_beta_ridge(ax, y_base, p_pct, n, col, max_h=0.50, alpha_fill=0.34, lw=0.9, zorder=2):
    k = p_pct / 100.0 * n
    a_p, b_p = k + 1.0, (n - k) + 1.0
    lo, hi = beta_dist.ppf([0.002, 0.998], a_p, b_p)
    xs = np.linspace(lo * 100.0, hi * 100.0, 140)
    pdf = beta_dist.pdf(xs / 100.0, a_p, b_p)
    ys = y_base - (pdf / pdf.max()) * max_h
    ax.fill_between(xs, y_base, ys, color=col, alpha=alpha_fill, lw=0, zorder=zorder)
    ax.plot(xs, ys, color=col, lw=lw, zorder=zorder + 0.1)


def fmt1_half_up(val):
    return str(Decimal(str(val)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def fmt_pts(val, decimals=2):
    if abs(val) < 10 ** (-decimals) / 2.0:
        return f"+{0.0:.{decimals}f}"
    return f"+{val:.{decimals}f}" if val > 0 else f"{MINUS}{abs(val):.{decimals}f}"


# =========================================================================
# 1. Load Ground-Truth Data
# =========================================================================
# Panel a: Silent 78-rule knockout (tmp/silent1120/*.json) + pre-patch pick/deck rates
pf_data = json.load(open(os.path.join(R, "paper", "figdata", "prepatch_forecast_5char.json")))
sil_sum = next(x for x in pf_data["summary_table"] if x["character"] == "Silent")
sil_cards_pf = pf_data["characters"]["Silent"]["per_changed_card"]
SILENT_SHOCK = abs(sil_sum["cold_start_delta_pts"])  # 5.1 pts

sil_dir = os.path.join(R, "tmp", "silent1120")
base_eps = {e["seed"]: bool(e["victory"]) for e in json.load(open(os.path.join(sil_dir, "silent_none.json")))["episodes"]}
BASE_WIN_PCT = 100.0 * sum(base_eps.values()) / len(base_eps)  # 42.95%


def paired_silent(fn_stem):
    ep = {e["seed"]: bool(e["victory"]) for e in json.load(open(os.path.join(sil_dir, f"{fn_stem}.json")))["episodes"]}
    ks = [k for k in base_eps if k in ep]
    b = sum(1 for k in ks if ep[k] and not base_eps[k])
    c = sum(1 for k in ks if base_eps[k] and not ep[k])
    delta = 100.0 * (b - c) / len(ks)
    se = 100.0 * math.sqrt(b + c) / len(ks) if (b + c) > 0 else 0.0
    return delta, se


knockout_rows = [
    ("All 78 changes", "silent_all", None),
    ("All 4 Silent cards", "silent_cards", None),
    ("Phantom Blades", "silent_blades", "CARD.PHANTOM_BLADES"),
    ("Speedster", "silent_speedster", "CARD.SPEEDSTER"),
    ("Memento Mori", "silent_memento", "CARD.MEMENTO_MORI"),
    ("Expertise", "silent_expertise", "CARD.EXPERTISE"),
]
ko_parsed = []
for label, stem, card_key in knockout_rows:
    d_pts, se_pts = paired_silent(stem)
    pick_pct = 100.0 * sil_cards_pf[card_key]["pick_rate"] if card_key else None
    deck_pct = 100.0 * sil_cards_pf[card_key]["deck_game_frac"] if card_key else None
    ko_parsed.append((label, d_pts, se_pts, pick_pct, deck_pct, card_key is not None))

# Panel b: Frozen v0.107.1 backward evaluation (tmp/fresh1124/v107_readings.txt)
v107_raw = {}
for line in open(os.path.join(R, "tmp", "fresh1124", "v107_readings.txt")):
    m = re.match(r"(v\d+_[a-z]+_v\d+)\s+read\s+\S+\s+n=(\d+)\s+win=([0-9.]+)", line.strip())
    if m:
        v107_raw[m.group(1)] = (int(m.group(2)), 100.0 * float(m.group(3)))

CHAR_KEY_MAP = [
    ("Regent", "regent"),
    ("Defect", "defect"),
    ("Ironclad", "iron"),
    ("Silent", "silent"),
    ("Necrobinder", "necro"),
]
v107_rows = []
for ch_name, short_k in CHAR_KEY_MAP:
    n111, w111 = v107_raw[f"v107_{short_k}_v111"]
    n113, w113 = v107_raw[f"v107_{short_k}_v113"]
    v107_rows.append((ch_name, w111, n111, w113, n113))
mean_v111 = sum(r[1] for r in v107_rows) / len(v107_rows)  # 47.0%
mean_v113 = sum(r[3] for r in v107_rows) / len(v107_rows)  # 60.4%

# Panel c: Fool Ascension 0 curriculum warm-start transfer (paper/figdata/fool_evals_raw.json + docs/03 zero-shot)
docs03 = open(os.path.join(R, "docs", "03_强化学习策略.md"), encoding="utf-8").read()
m_zs = re.search(r"五角色联合模型\s*([0-9.]+)%.*战士主线\s*([0-9.]+)%", docs03)
ZS_JOINT = float(m_zs.group(1)) if m_zs else 25.4
ZS_IRON = float(m_zs.group(2)) if m_zs else 12.4
ZS_SCRATCH = 0.0

fool_raw = json.load(open(os.path.join(R, "paper", "figdata", "fool_evals_raw.json")))


def load_fool_curve(key, zs_val, max_M=82.0):
    xs, ys, ns = [0.0], [zs_val], [200]
    for row in csv.DictReader(fool_raw[key].splitlines()):
        if row["mode"] != "greedy":
            continue
        sm = int(row["env_steps"]) / 1e6
        if sm <= max_M:
            xs.append(sm)
            ys.append(100.0 * float(row["win_rate"]))
            ns.append(int(row["n"]))
    return np.array(xs), np.array(ys), ns


fx_joint, fy_joint, fn_joint = load_fool_curve("epsilon:runs/v3_fool_joint/eval.csv", ZS_JOINT, max_M=82.0)
fx_iron, fy_iron, fn_iron = load_fool_curve("a100:runs/v3_fool_iron/eval.csv", ZS_IRON, max_M=82.0)
fx_scr, fy_scr, fn_scr = load_fool_curve("epsilon:runs/v3_fool_scratch/eval.csv", ZS_SCRATCH, max_M=82.0)

# =========================================================================
# 2. Figure Layout (8.5 x 4.85 in, 350 DPI)
# =========================================================================
FIG_W, FIG_H = 8.5, 4.85
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

fig.add_artist(FancyBboxPatch(
    (0.006, 0.008), 0.988, 0.984, transform=fig.transFigure,
    boxstyle="round,pad=0.0,rounding_size=0.014",
    facecolor="none", edgecolor=OUTER_BORDER, linewidth=0.8, zorder=0
))

# Layout: top row, panel a (Silent rule knockout with pre-patch pick/deck rates and card art) and panel b
# (v0.107.1 backward evaluation); bottom row, panel c (Fool A0 warm-start curves, then A0/A8/A10 at matched steps).

ax_a = fig.add_axes([0.145, 0.548, 0.220, 0.365])
ax_a_hm = fig.add_axes([0.372, 0.548, 0.098, 0.365])

ax_b = fig.add_axes([0.568, 0.548, 0.398, 0.365])

ax_c1 = fig.add_axes([0.072, 0.105, 0.515, 0.342])
ax_c2 = fig.add_axes([0.658, 0.105, 0.308, 0.342])

# =========================================================================
# Panel a: Silent v0.113 78-rule knockout + pre-patch pick/deck table + card art
# =========================================================================
clean_ax(ax_a)
ax_a.axvline(0.0, color=INK2, lw=0.75, zorder=2)
ax_a.axvline(SILENT_SHOCK, color=TERRA, lw=1.0, ls=(0, (3.2, 1.8)), zorder=2)
ax_a.text(SILENT_SHOCK, -0.70, f"loss ({SILENT_SHOCK:.1f})", fontsize=FS_TICK, fontweight="bold",
          color=TERRA, ha="center", va="bottom")

for i, (label, d_pts, se_pts, pick_pct, deck_pct, is_card) in enumerate(ko_parsed):
    ry = i + (0.22 if i >= 2 else 0.0)
    col = SLATE if not is_card else (AMBER if d_pts > 0.5 else (TERRA if d_pts < -0.3 else BLUE))
    if i == 0:
        col = "#334155"
    ax_a.barh(ry, d_pts, height=0.58, color=col, alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)
    if se_pts > 0:
        ax_a.plot([d_pts - se_pts, d_pts + se_pts], [ry, ry], color=INK, lw=0.85, zorder=4)
    tx = max(d_pts + se_pts, 0.0) + 0.22 if d_pts >= 0 else min(d_pts - se_pts, 0.0) - 0.22
    ha_t = "left" if d_pts >= 0 else "right"
    ax_a.text(tx, ry, fmt_pts(d_pts, 2), fontsize=FS_CELL, fontweight="bold",
              color=col if abs(d_pts) >= 0.5 else INK2, ha=ha_t, va="center", zorder=5)

ax_a.axhline(1.61, color=BORDER, lw=0.8, zorder=2)
yt_a = [i + (0.22 if i >= 2 else 0.0) for i in range(len(ko_parsed))]
ax_a.set_xlim(-3.05, 6.2)
ax_a.set_ylim(yt_a[-1] + 0.58, -0.62)
ax_a.set_xticks([-2, 0, 2, 4, 6])
ax_a.set_xticklabels([f"{MINUS}2", "0", "+2", "+4", "+6"])
ax_a.set_yticks(yt_a)
ax_a.set_yticklabels([
    r[0] for r in ko_parsed
], fontsize=FS_TICK)
for idx_lbl, tick_lbl in enumerate(ax_a.get_yticklabels()):
    if ko_parsed[idx_lbl][5]:
        tick_lbl.set_fontstyle("italic")
ax_a.tick_params(axis="y", length=0, pad=3.0)
ax_a.set_xlabel("Points regained on 2,000 seeds", labelpad=2.5)
add_sprite_path(ax_a, os.path.join(ASSET_V2_DIR, CHAR_BADGES["Silent"]), (-1.95, 0.46), target_h_pt=19.0, zorder=6)
ax_a.text(-2.90, -0.70, f"baseline {BASE_WIN_PCT:.2f}%", fontsize=FS_TICK, fontweight="bold",
          color=INK2, ha="left", va="bottom")

# Companion aligned pre-patch exposure columns (Picked %, In deck %) + real card renders
ax_a_hm.set_xlim(0, 2.0)
ax_a_hm.set_ylim(yt_a[-1] + 0.58, -0.62)
ax_a_hm.axis("off")

amber_cmap = LinearSegmentedColormap.from_list("amb", [(0.0, "#f8fafc"), (0.35, "#fef3c7"), (1.0, "#d97706")])
ax_a_hm.text(0.48, -0.70, "Taken", fontsize=FS_TICK, fontweight="bold", color=INK, ha="center", va="bottom")
ax_a_hm.text(1.48, -0.70, "Deck", fontsize=FS_TICK, fontweight="bold", color=INK, ha="center", va="bottom")
ax_a_hm.text(0.98, -1.16, "Before the update (%)", fontsize=FS_TICK, fontweight="bold", color=AMBER, ha="center", va="bottom")

for i, (label, d_pts, se_pts, pick_pct, deck_pct, is_card) in enumerate(ko_parsed):
    ry = yt_a[i]
    if not is_card:
        continue
    for j, val in enumerate((pick_pct, deck_pct)):
        t_v = min(1.0, val / 85.0)
        ax_a_hm.add_patch(Rectangle((j + 0.04, ry - 0.36), 0.90, 0.72,
                                    facecolor=amber_cmap(t_v), edgecolor="white", linewidth=0.8, zorder=2))
        ax_a_hm.text(j + 0.49, ry, f"{val:.1f}", fontsize=FS_CELL,
                     fontweight="bold" if val >= 15 or "Memento" in label else "normal",
                     color="white" if t_v > 0.72 else INK, ha="center", va="center", zorder=3)

# Place real Silent card thumbnails in the top two aggregate rows of ax_a_hm (zero overlap with card rows!)
add_sprite_path(ax_a_hm, os.path.join(ASSET_ED_DIR, "card_phantom_blades.png"),
                (0.46, 0.56), target_h_pt=38.0, zorder=7)
add_sprite_path(ax_a_hm, os.path.join(ASSET_ED_DIR, "card_memento_mori.png"),
                (1.50, 0.56), target_h_pt=38.0, zorder=7)

# =========================================================================
# Panel b: Zero-shot backward evaluation on frozen v0.107.1 (v0.111 vs v0.113 nets)
# =========================================================================
clean_ax(ax_b)
for gx in (30, 40, 50, 60, 70):
    ax_b.axvline(gx, color=GRID, lw=0.55, zorder=0)

ax_b.axvline(mean_v111, color=SLATE, lw=0.9, ls=":", zorder=1.5)
ax_b.axvline(mean_v113, color=BLUE, lw=0.95, ls="--", zorder=1.5)

X_COL_111 = 76.8
X_COL_113 = 84.8

for i, (ch_name, w111, n111, w113, n113) in enumerate(v107_rows):
    ry = i
    if i % 2 == 0:
        ax_b.axhspan(ry - 0.46, ry + 0.46, color=ZEBRA, zorder=0.1)
    base = ry + 0.25
    ch_col = CHAR_COLORS[ch_name]
    draw_beta_ridge(ax_b, base, w111, n111, SLATE, max_h=0.50, alpha_fill=0.22, lw=0.8, zorder=2)
    draw_beta_ridge(ax_b, base, w113, n113, ch_col, max_h=0.50, alpha_fill=0.38, lw=0.95, zorder=3)
    ax_b.plot([w111, w113], [base, base], color=INK, lw=1.2, zorder=4)
    ax_b.scatter([w111], [base], s=23, facecolor="white", edgecolor=SLATE, linewidth=1.25, zorder=5)
    ax_b.scatter([w113], [base], s=25, color=ch_col, edgecolor="white", linewidth=0.6, zorder=6)

    # Character badge at left
    add_sprite_path(ax_b, os.path.join(ASSET_V2_DIR, CHAR_BADGES[ch_name]), (27.2, ry), target_h_pt=17.0, zorder=6)

    # Aligned value pills on the right (ROUND_HALF_UP matching manuscript Table 1 / Fig. 2b)
    ax_b.add_patch(Rectangle((X_COL_111 - 3.6, ry - 0.36), 7.2, 0.72, facecolor=SLATE_PILL, edgecolor="none", zorder=2))
    ax_b.text(X_COL_111, ry, fmt1_half_up(w111), fontsize=FS_CELL, color=INK2, ha="center", va="center", zorder=3)
    ax_b.add_patch(Rectangle((X_COL_113 - 3.6, ry - 0.36), 7.2, 0.72, facecolor=BLUE_PILL, edgecolor="none", zorder=2))
    ax_b.text(X_COL_113, ry, fmt1_half_up(w113), fontsize=FS_CELL, fontweight="bold", color=BLUE,
              ha="center", va="center", zorder=3)

ax_b.set_xlim(23.5, 89.2)
ax_b.set_ylim(4.58, -0.78)
ax_b.set_xticks([30, 40, 50, 60, 70])
ax_b.set_yticks(range(len(v107_rows)))
ax_b.set_yticklabels([r[0] for r in v107_rows], fontsize=FS_TICK)
ax_b.tick_params(axis="y", length=0, pad=2.0)
ax_b.set_xlabel("Zero-shot Ascension 10 win rate on frozen v0.107.1 (%)", labelpad=2.5)

# Top mean headers
ax_b.text(mean_v111 - 0.8, -0.52, f"mean {fmt1_half_up(mean_v111)}%", fontsize=FS_TICK, fontweight="bold",
          color=SLATE, ha="right", va="center")
ax_b.text(mean_v113 + 0.8, -0.52, f"mean {fmt1_half_up(mean_v113)}%", fontsize=FS_TICK, fontweight="bold",
          color=BLUE, ha="left", va="center")
ax_b.text(X_COL_111, -0.52, "v0.111", fontsize=FS_TICK, fontweight="bold", color=SLATE, ha="center", va="center")
ax_b.text(X_COL_113, -0.52, "v0.113", fontsize=FS_TICK, fontweight="bold", color=BLUE, ha="center", va="center")

# =========================================================================
# Panel c1: Fool Ascension 0 curriculum warm-start transfer curves
# =========================================================================
clean_ax(ax_c1)
for gy in (25, 50, 75):
    ax_c1.axhline(gy, color=GRID, lw=0.55, zorder=0)
ax_c1.axvline(39.32, color=BORDER, lw=0.95, ls=(0, (3.0, 2.0)), zorder=1)


def plot_fool_band(ax, xs, ys, ns, col, label, marker="o", lw=1.6, zorder=4):
    lo = [wilson_pct(y, n)[0] if x > 0 else max(0.0, y - 2.5) for x, y, n in zip(xs, ys, ns)]
    hi = [wilson_pct(y, n)[1] if x > 0 else min(100.0, y + 2.5) for x, y, n in zip(xs, ys, ns)]
    ax.fill_between(xs, lo, hi, color=col, alpha=0.14, lw=0, zorder=zorder - 1)
    ax.plot(xs, ys, marker=marker, color=col, ms=4.2, mec="white", mew=0.45, lw=lw, label=label, zorder=zorder)


# Subsample dense scratch curve to match 6.55M cadence cleanly while keeping zero-shot
scr_mask = [i for i, x in enumerate(fx_scr) if i == 0 or abs((x / 6.5536) - round(x / 6.5536)) < 0.08]
plot_fool_band(ax_c1, fx_scr[scr_mask], fy_scr[scr_mask], [fn_scr[i] for i in scr_mask],
               SLATE, "Untrained", marker="^", lw=1.35, zorder=3)
plot_fool_band(ax_c1, fx_iron, fy_iron, fn_iron,
               BLUE, "From the Ironclad agent", marker="s", lw=1.55, zorder=4)
plot_fool_band(ax_c1, fx_joint, fy_joint, fn_joint,
               VIOLET, "From the five-character agent", marker="o", lw=1.75, zorder=5)

# Zero-shot & convergence callouts (to the left of x = 0, completely clear of rising curves)
ax_c1.text(-1.1, ZS_JOINT + 1.0, f"{ZS_JOINT:.1f}%", fontsize=FS_LAB, fontweight="bold",
           color=VIOLET, ha="right", va="center")
ax_c1.text(-1.1, ZS_IRON, f"{ZS_IRON:.1f}%", fontsize=FS_LAB, fontweight="bold",
           color=BLUE, ha="right", va="center")
ax_c1.text(-1.1, 0.5, "0%", fontsize=FS_LAB - 0.2, fontweight="bold",
           color=SLATE, ha="right", va="center")
ax_c1.text(40.8, 73.5, "converged\nby 39M steps", fontsize=FS_LAB - 0.2, fontweight="bold",
           color=INK, ha="left", va="top", linespacing=1.08)

add_sprite_path(ax_c1, os.path.join(ASSET_V2_DIR, CHAR_BADGES["Fool"]), (74.5, 23.0), target_h_pt=21.0, zorder=6)

ax_c1.set_xlim(-9.8, 81.5)
ax_c1.set_ylim(-3, 98)
ax_c1.set_xticks([0, 20, 39, 60, 80])
ax_c1.set_yticks([0, 25, 50, 75, 100])
ax_c1.set_xlabel("Environment steps on the Fool (millions)", labelpad=2.5)
ax_c1.set_ylabel("Ascension 0 win rate (%)", labelpad=2.0)
ax_c1.legend(loc="center right", bbox_to_anchor=(0.98, 0.45), frameon=True, facecolor="white",
             edgecolor=BORDER, framealpha=0.95, fontsize=FS_TICK, handlelength=1.4, borderpad=0.35)

# =========================================================================
# Panel c2: Multi-Ascension (A0 / A8 / A10) parity at 39.3M steps
# =========================================================================
clean_ax(ax_c2)
for gy in (25, 50, 75):
    ax_c2.axhline(gy, color=GRID, lw=0.55, zorder=0)


def eval_at_step(key, target_M=39.32):
    rows = [r for r in csv.DictReader(fool_raw[key].splitlines()) if r["mode"] == "greedy"]
    best = min(rows, key=lambda r: abs(int(r["env_steps"]) / 1e6 - target_M))
    return 100.0 * float(best["win_rate"]), int(best["n"])


asc_keys = [
    ("Ascension 0", "eval.csv"),
    ("Ascension 8", "eval_a8.csv"),
    ("Ascension 10", "eval_a10.csv"),
]
xc2 = np.arange(3)
bw2 = 0.25

for j_arm, (prefix, col, lbl) in enumerate([
    ("epsilon:runs/v3_fool_joint", VIOLET, "Five characters"),
    ("a100:runs/v3_fool_iron", BLUE, "Ironclad"),
    ("epsilon:runs/v3_fool_scratch", SLATE, "Untrained"),
]):
    vals, errs_lo, errs_hi = [], [], []
    for _, suf in asc_keys:
        v, n_v = eval_at_step(f"{prefix}/{suf}", 39.32)
        lo_v, hi_v = wilson_pct(v, n_v)
        vals.append(v)
        errs_lo.append(v - lo_v)
        errs_hi.append(hi_v - v)
    xpos = xc2 + (j_arm - 1) * (bw2 + 0.02)
    ax_c2.bar(xpos, vals, width=bw2, color=col, alpha=0.90, edgecolor="white", linewidth=0.5, label=lbl, zorder=3)
    ax_c2.errorbar(xpos, vals, yerr=[errs_lo, errs_hi], fmt="none", ecolor=INK, elinewidth=0.75, capsize=1.8, zorder=4)
    for idx_v, (xp_i, v_i) in enumerate(zip(xpos, vals)):
        if v_i >= 5.0 or j_arm < 2:
            v_lbl = str(Decimal(str(v_i)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            ax_c2.text(xp_i, v_i + errs_hi[idx_v] + 1.8, v_lbl,
                       fontsize=FS_TICK - 0.1, fontweight="bold", color=col, ha="center", va="bottom", zorder=5)

# Add Fool full-body game sprite in open upper-right space above Ascension 10
if os.path.exists(os.path.join(ASSET_W_DIR, "char_fool.png")):
    add_sprite_path(ax_c2, os.path.join(ASSET_W_DIR, "char_fool.png"), (2.05, 64.0), target_h_pt=50.0, zorder=6)

ax_c2.set_xlim(-0.52, 2.55)
ax_c2.set_ylim(0, 104)
ax_c2.set_xticks(xc2)
ax_c2.set_xticklabels(["A0", "A8", "A10"], fontsize=FS_TICK)
ax_c2.set_yticks([0, 25, 50, 75, 100])
ax_c2.set_xlabel("Ascension level at 39M steps", labelpad=2.5)
ax_c2.set_ylabel("Win rate (%)", labelpad=2.0)
ax_c2.legend(loc="upper right", bbox_to_anchor=(0.76, 1.01), frameon=False, fontsize=FS_TICK - 0.1,
             handlelength=0.95, handletextpad=0.32, labelspacing=0.22)

# =========================================================================
# Panel letters
# =========================================================================
for lab, (gx, gy) in zip("abc", [(0.014, 0.980), (0.495, 0.980), (0.014, 0.485)]):
    fig.text(gx, gy, lab, fontsize=FS_LETTER, fontweight="bold", va="top", color=INK)

out_base = os.path.join(R, "paper", "figs", "ed_fig2_reverts_transfer")
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out_base}.{ext}", dpi=350)
print(f"written {out_base}.{{png,pdf,svg}}")
print("a Silent knockouts:", [(r[0], round(r[1], 2), r[3], r[4]) for r in ko_parsed])
print(f"b v0.107.1 means: v0.111={mean_v111:.1f}%, v0.113={mean_v113:.1f}%")
