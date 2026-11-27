"""Extended Data Figure 4: Diagnostic probes of exploration and one-step lookahead at the combination boundary.

  a, Upstream reward-screen evasion when forced card-taking is non-mandatory (19.9 and 19.4 card-reward openings
     per run in control and mandatory v3 versus 15.3 in v1 neglected-card forcing and 11.8 in v1 exhaust forcing).
  b, Five-character uniform zero-gradient neglected-card exposure (v3n2_* across Ironclad, Defect, Regent, Silent,
     Necrobinder): voluntary pick rates in unforced runs dropping to 1/2.2-1/6.0 (1/3-1/7 rounded) of control,
     paired with 2,000-seed win-rate differences (-0.60, -0.25, -1.85, +0.55, -2.60 pts).
  c, Episode-coherent random embedding-direction ("taste", β = 120) exploration probe versus control on 2,000
     paired seeds (70.85% vs 69.75%, +1.10 pts; neglected-card pick share 2.2% vs 3.6%).
  d, One-step public-information combat value lookahead on 163 reproducible Act-3 boss fights (safe: 8 rescued vs
     4 lost, exit HP 53 -> 49; oracle: 9 rescued vs 10 lost) and 100-seed full-run paired test (68% -> 53%,
     11 rescued vs 26 lost).

Writes paper/figs/ed_fig4_probes.{png,pdf,svg} (350 DPI).
"""
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
AMBER_PILL = "#f9edd2"
VIOLET = "#5e4cc4"
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


def draw_beta_ridge(ax, y_base, k, n, col_line, col_fill, max_h=0.48, alpha_fill=0.36, lw=0.95, zorder=3):
    a_p, b_p = k + 1.0, (n - k) + 1.0
    mu = a_p / (a_p + b_p)
    sd = math.sqrt(a_p * b_p / (((a_p + b_p) ** 2) * (a_p + b_p + 1.0)))
    xg = np.linspace(max(0.001, mu - 4.3 * sd), min(0.999, mu + 4.3 * sd), 150)
    dens = beta_dist.pdf(xg, a_p, b_p)
    dens = dens / dens.max() * max_h
    xp = xg * 100.0
    ax.fill_between(xp, y_base, y_base - dens, color=col_fill, alpha=alpha_fill, lw=0, zorder=zorder)
    ax.plot(xp, y_base - dens, color=col_line, lw=lw, zorder=zorder + 0.2)


def fmt_pts(val, decimals=2):
    if abs(val) < 10 ** (-decimals) / 2.0:
        return f"+{0.0:.{decimals}f}"
    return f"+{val:.{decimals}f}" if val > 0 else f"{MINUS}{abs(val):.{decimals}f}"


# =========================================================================
# 1. Parse Ground-Truth Data From probe_readout_1123.txt, winvec_*.txt,
#    combat_lookahead_1120.md, and docs/03_强化学习策略.md
# =========================================================================
probe_txt = open(os.path.join(R, "docs", "results", "v3_iron", "probe_readout_1123.txt"), encoding="utf-8").read()
docs03 = open(os.path.join(R, "docs", "03_强化学习策略.md"), encoding="utf-8").read()
la_txt = open(os.path.join(R, "docs", "results", "v3_iron", "combat_lookahead_1120.md"), encoding="utf-8").read()


def parse_probe_picks(trace_name):
    pat = rf"== {re.escape(trace_name)}: games (\d+), win ([0-9.]+)%; rewards offering a listed card (\d+), took one (\d+)"
    m = re.search(pat, probe_txt)
    assert m is not None, trace_name
    pat_cr = rf"== {re.escape(trace_name)}: games \d+; card_reward decisions per game ([0-9.]+)"
    m_cr = re.search(pat_cr, probe_txt)
    cr_pg = float(m_cr.group(1)) if m_cr else None
    return int(m.group(3)), int(m.group(4)), cr_pg


# Load 2,000-seed paired binary vectors from tmp/fresh1123/winvec_*.txt
vecs = {}
for fn in ("winvec_a100.txt", "winvec_b100.txt", "winvec_epsilon.txt", "winvec_n10.txt", "winvec_z.txt"):
    p = os.path.join(R, "tmp", "fresh1123", fn)
    for line in open(p):
        parts = line.strip().split()
        if len(parts) == 2:
            vecs[parts[0]] = parts[1]


def paired_winvec(k_probe, k_ctl):
    vp, vc = vecs[k_probe], vecs[k_ctl]
    valid = [(a, b) for a, b in zip(vp, vc) if a in "01" and b in "01"]
    n = len(valid)
    wp = sum(a == "1" for a, _ in valid)
    wc = sum(b == "1" for _, b in valid)
    only_p = sum(a == "1" and b == "0" for a, b in valid)
    only_c = sum(a == "0" and b == "1" for a, b in valid)
    diff = 100.0 * (only_p - only_c) / n
    se = 100.0 * math.sqrt(only_p + only_c) / n
    return wp, wc, n, diff, se


# Panel a: Upstream reward-screen evasion
_, _, cr_iron_ctl = parse_probe_picks("iron_ctl_free.jsonl.gz")      # 19.92
_, _, cr_iron_neg2 = parse_probe_picks("iron_neg_free.jsonl.gz")     # 18.45
m_v1_exh = re.search(r"从 ([0-9.]+) 次掉到 ([0-9.]+) 次", docs03)
cr_ctl_early = float(m_v1_exh.group(1))  # 19.5
cr_v1_exh = float(m_v1_exh.group(2))     # 11.8
m_v3_exh = re.search(r"每局打开 ([0-9.]+) 次卡牌奖励", docs03)
cr_v3_exh = float(m_v3_exh.group(1))     # 19.4
m_v1_neg = re.search(r"第一版是 ([0-9.]+) 对 ([0-9.]+)", docs03)
cr_v1_neg = float(m_v1_neg.group(1))     # 15.3

evasion_rows = [
    ("Control\n(no forcing)", cr_iron_ctl, SLATE),
    ("Exhaust cards\nsupplied", cr_v3_exh, TEAL),
    ("Neglected cards\nsupplied", cr_iron_neg2, BLUE),
    ("Neglected cards,\nforcing avoidable", cr_v1_neg, AMBER),
    ("Exhaust cards,\nforcing avoidable", cr_v1_exh, TERRA),
]

# Panel b: Five-character v3n2 neglected-card exposure
CHAR_SPECS_B = [
    ("Ironclad", "iron_ctl_free.jsonl.gz", "iron_neg_free.jsonl.gz", "iron_neg", "iron_ctl"),
    ("Defect", "defect_main_free.jsonl.gz", "defect_neg_free.jsonl.gz", "defect_neg", "defect_main"),
    ("Regent", "regent_main_free.jsonl.gz", "regent_neg_free.jsonl.gz", "regent_neg", "regent_main"),
    ("Silent", "silent_main_free.jsonl.gz", "silent_neg_free.jsonl.gz", "silent_neg", "silent_main"),
    ("Necrobinder", "necro_main_free.jsonl.gz", "necro_neg_free.jsonl.gz", "necro_neg", "necro_main"),
]
rows_b = []
for ch_nm, f_ctl, f_neg, wv_neg, wv_ctl in CHAR_SPECS_B:
    off_c, tk_c, _ = parse_probe_picks(f_ctl)
    off_n, tk_n, _ = parse_probe_picks(f_neg)
    wp, wc, n_wv, d_wv, se_wv = paired_winvec(wv_neg, wv_ctl)
    rows_b.append({
        "char": ch_nm,
        "off_c": off_c, "tk_c": tk_c, "pct_c": 100.0 * tk_c / off_c,
        "off_n": off_n, "tk_n": tk_n, "pct_n": 100.0 * tk_n / off_n,
        "ratio": (tk_c / off_c) / max(1e-9, tk_n / off_n),
        "diff": d_wv, "se": se_wv,
    })

# Panel c: Taste β = 120 exploration probe vs Ironclad control
off_taste, tk_taste, _ = parse_probe_picks("iron_taste_free.jsonl.gz")
off_ictl, tk_ictl, _ = parse_probe_picks("iron_ctl_free.jsonl.gz")
wp_taste, wc_taste, n_taste, diff_taste, se_taste = paired_winvec("iron_taste", "iron_ctl")

# Panel d: One-step combat value lookahead (combat_lookahead_1120.md)
def parse_la_mode(prefix):
    for line in la_txt.splitlines():
        if line.startswith(f"| {prefix} |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            resc = int(cells[1].split("/")[0])
            resc_tot = int(cells[1].split("/")[1].split("（")[0])
            lost = int(cells[2].split("/")[0])
            lost_tot = int(cells[2].split("/")[1].split("（")[0])
            win_hp = float(cells[5])
            return resc, resc_tot, lost, lost_tot, win_hp
    raise KeyError(prefix)


_, _, _, _, hp_pol = parse_la_mode("policy")
resc_safe, tot_loss_b, lost_safe, tot_win_b, hp_safe = parse_la_mode("safe")
resc_ora, _, lost_ora, _, hp_ora = parse_la_mode("oracle（信息泄漏上界）")

m_full = re.search(
    r"\|\s*100\s*\|\s*100\s*\|\s*(\d+)/100[^|]*\|\s*(\d+)/100[^|]*\|\s*(\d+)\s*\|\s*(\d+)\s*\|",
    la_txt,
)
assert m_full is not None
full_greedy_w, full_safe_w, full_resc, full_lost = [int(x) for x in m_full.groups()]

# =========================================================================
# 2. Figure Layout (8.5 x 5.05 in, 350 DPI)
# =========================================================================
FIG_W, FIG_H = 8.5, 5.05
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

fig.add_artist(FancyBboxPatch(
    (0.006, 0.008), 0.988, 0.984, transform=fig.transFigure,
    boxstyle="round,pad=0.0,rounding_size=0.014",
    facecolor="none", edgecolor=OUTER_BORDER, linewidth=0.8, zorder=0
))

# Top row: Panel a (left 34%), Panel b (right 59%: voluntary pick rates + paired 2k-seed Δ win rate)
ax_a = fig.add_axes([0.130, 0.560, 0.245, 0.365])
ax_b1 = fig.add_axes([0.468, 0.560, 0.305, 0.365])
ax_b2 = fig.add_axes([0.805, 0.560, 0.168, 0.365])

# Bottom row: Panel c (left 41%: Taste β=120 win posterior + neglected-card picks),
#             Panel d (right 50%: 1-step lookahead outcome flips + full-run win drop)
ax_c1 = fig.add_axes([0.088, 0.290, 0.335, 0.168])
ax_c2 = fig.add_axes([0.088, 0.100, 0.335, 0.128])

ax_d1 = fig.add_axes([0.530, 0.100, 0.250, 0.362])
ax_d2 = fig.add_axes([0.848, 0.100, 0.128, 0.362])

# =========================================================================
# Panel a: Upstream reward-screen evasion
# =========================================================================
clean_ax(ax_a)
for gx in (5, 10, 15, 20):
    ax_a.axvline(gx, color=GRID, lw=0.55, zorder=0)
ax_a.axvline(cr_iron_ctl, color=SLATE, lw=0.85, ls="--", zorder=1)

for i, (lbl, val, col) in enumerate(evasion_rows):
    ax_a.barh(i, val, height=0.58, color=col, alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)
    v_str = str(Decimal(str(val)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    ax_a.text(val + 0.45, i, v_str, fontsize=FS_LAB, fontweight="bold", color=col, va="center", ha="left")

ax_a.axhline(2.50, color=BORDER, lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
ax_a.text(17.4, 4.02, "opens\nfewer", fontsize=FS_TICK, fontweight="bold",
          color=TERRA, ha="center", va="center", linespacing=1.02)
add_sprite(ax_a, CHAR_BADGES["Ironclad"], (23.4, 3.72), target_h_pt=17.5, zorder=6)

ax_a.set_xlim(0, 26.2)
ax_a.set_ylim(4.65, -0.58)
ax_a.set_xticks([0, 5, 10, 15, 20])
ax_a.set_yticks(range(len(evasion_rows)))
ax_a.set_yticklabels([r[0] for r in evasion_rows], fontsize=FS_TICK)
ax_a.tick_params(axis="y", length=0, pad=2.5)
ax_a.set_xlabel("Card rewards opened per game", labelpad=2.5)

# =========================================================================
# Panel b1: Five-character uniform zero-gradient neglected-card exposure
# =========================================================================
clean_ax(ax_b1)
for gx in (1, 2, 3, 4):
    ax_b1.axvline(gx, color=GRID, lw=0.55, zorder=0)

for i, r in enumerate(rows_b):
    ch = r["char"]
    col = CHAR_COLORS[ch]
    if i % 2 == 0:
        ax_b1.axhspan(i - 0.46, i + 0.46, color=ZEBRA, zorder=0.1)

    # Dumbbell from Control -> Exposed (voluntary picks in 300 unforced runs)
    ax_b1.plot([r["pct_n"], r["pct_c"]], [i, i], color=INK, lw=1.25, zorder=3)
    ax_b1.scatter([r["pct_c"]], [i], s=28, facecolor="white", edgecolor=SLATE, linewidth=1.3, zorder=4)
    ax_b1.scatter([r["pct_n"]], [i], s=30, color=col, edgecolor="white", linewidth=0.6, zorder=5)

    # Character badge on left
    add_sprite(ax_b1, CHAR_BADGES[ch], (-0.42, i), target_h_pt=17.0, zorder=6)

    # Exact fraction readout on right side of ax_b1
    frac_txt = f"{r['tk_n']}/{r['off_n']} vs {r['tk_c']}/{r['off_c']} (÷{r['ratio']:.1f})"
    ax_b1.text(3.92, i, frac_txt, fontsize=FS_TICK - 0.15, color=INK, va="center", ha="left", zorder=5)

ax_b1.set_xlim(-0.10, 7.65)
ax_b1.set_ylim(4.58, -0.72)
ax_b1.set_xticks([0, 1, 2, 3, 4])
ax_b1.set_yticks([])
ax_b1.spines["left"].set_color(INK2)
ax_b1.set_xlabel("Voluntary neglected-card picks (%)", labelpad=2.5)

ax_b1.scatter([0.20], [-0.46], s=22, color=BLUE, edgecolor="white", linewidth=0.6, clip_on=False, zorder=5)
ax_b1.text(0.45, -0.46, "Cards supplied", fontsize=FS_TICK, fontweight="bold", color=BLUE, va="center")
ax_b1.scatter([2.95], [-0.46], s=22, facecolor="white", edgecolor=SLATE, linewidth=1.25, clip_on=False, zorder=5)
ax_b1.text(3.20, -0.46, "Control", fontsize=FS_TICK, fontweight="bold", color=SLATE, va="center")
ax_b1.text(5.80, -0.46, "Picks / offers", fontsize=FS_TICK, fontweight="bold", color=INK, ha="center", va="center")

# =========================================================================
# Panel b2: Paired 2,000-seed win-rate difference (v3n2 - control)
# =========================================================================
clean_ax(ax_b2)
ax_b2.axvline(0.0, color=INK2, lw=0.8, zorder=2)
for gx in (-4, -2, 2):
    ax_b2.axvline(gx, color=GRID, lw=0.55, zorder=0)

for i, r in enumerate(rows_b):
    if i % 2 == 0:
        ax_b2.axhspan(i - 0.46, i + 0.46, color=ZEBRA, zorder=0.1)
    d_v, se_v = r["diff"], r["se"]
    col = CHAR_COLORS[r["char"]]
    ax_b2.barh(i, d_v, height=0.52, color=col, alpha=0.86, edgecolor="white", linewidth=0.5, zorder=3)
    ax_b2.plot([d_v - se_v, d_v + se_v], [i, i], color=INK, lw=0.85, zorder=4)
    tx = d_v + se_v + 0.25 if d_v >= 0 else d_v - se_v - 0.25
    ha_t = "left" if d_v >= 0 else "right"
    ax_b2.text(tx, i, fmt_pts(d_v, 2), fontsize=FS_CELL, fontweight="bold",
               color=col, ha=ha_t, va="center", zorder=5)

ax_b2.set_xlim(-6.5, 4.2)
ax_b2.set_ylim(4.58, -0.72)
ax_b2.set_xticks([-4, -2, 0, 2])
ax_b2.set_xticklabels([f"{MINUS}4", f"{MINUS}2", "0", "+2"])
ax_b2.set_yticks([])
ax_b2.spines["left"].set_visible(False)
ax_b2.set_xlabel("Δ win rate (points)", labelpad=2.5)
ax_b2.text(0.0, -0.46, "2,000 paired seeds", fontsize=FS_TICK, fontweight="bold",
           color=INK, ha="center", va="center")

# =========================================================================
# Panel c: Taste (β = 120) coherent embedding-direction exploration probe
# =========================================================================
clean_ax(ax_c1)
for gx in (66, 68, 70, 72, 74):
    ax_c1.axvline(gx, color=GRID, lw=0.55, zorder=0)

p_ctl_c = 100.0 * wc_taste / n_taste      # 69.75%
p_tst_c = 100.0 * wp_taste / n_taste      # 70.85%
base_c = 0.62
draw_beta_ridge(ax_c1, base_c, wc_taste, n_taste, SLATE, SLATE, max_h=0.46, alpha_fill=0.25, lw=0.85, zorder=2)
draw_beta_ridge(ax_c1, base_c, wp_taste, n_taste, AMBER, AMBER, max_h=0.46, alpha_fill=0.38, lw=1.0, zorder=3)
ax_c1.plot([p_ctl_c, p_tst_c], [base_c, base_c], color=INK, lw=1.3, zorder=4)
ax_c1.scatter([p_ctl_c], [base_c], s=28, facecolor="white", edgecolor=SLATE, linewidth=1.3, zorder=5)
ax_c1.scatter([p_tst_c], [base_c], s=28, color=AMBER, edgecolor="white", linewidth=0.6, zorder=6)

ax_c1.text(p_ctl_c - 0.35, 0.77, f"Control {p_ctl_c:.2f}%", fontsize=FS_LAB - 0.1, fontweight="bold",
           color=SLATE, ha="right", va="top")
ax_c1.text(p_tst_c + 0.35, 0.77, f"Random taste  {p_tst_c:.2f}%", fontsize=FS_LAB - 0.1, fontweight="bold",
           color=AMBER, ha="left", va="top")
ax_c1.add_patch(FancyBboxPatch(
    (73.1, 0.16), 2.55, 0.34,
    boxstyle="round,pad=0.0,rounding_size=0.08",
    facecolor=AMBER_PILL, edgecolor="none", zorder=4
))
ax_c1.text(74.38, 0.33, f"+{diff_taste:.2f} points", fontsize=FS_CELL, fontweight="bold",
           color=AMBER, ha="center", va="center", zorder=5)
add_sprite(ax_c1, CHAR_BADGES["Ironclad"], (65.8, 0.34), target_h_pt=18.5, zorder=6)

ax_c1.set_xlim(64.8, 76.0)
ax_c1.set_ylim(0.95, 0.04)
ax_c1.set_xticks([66, 68, 70, 72, 74])
ax_c1.set_yticks([])
ax_c1.spines["left"].set_visible(False)
ax_c1.set_xlabel("Win rate on 2,000 paired seeds (%)", labelpad=2.0)

# Bottom part of Panel c: Neglected-card pick share (2.2% vs 3.6%)
clean_ax(ax_c2)
for gx in (1, 2, 3, 4):
    ax_c2.axvline(gx, color=GRID, lw=0.55, zorder=0)

pct_ictl = 100.0 * tk_ictl / off_ictl      # 3.58%
pct_itst = 100.0 * tk_taste / off_taste    # 2.23%
ax_c2.barh(0, pct_ictl, height=0.55, color=SLATE, alpha=0.82, edgecolor="white", linewidth=0.5, zorder=3)
ax_c2.barh(1, pct_itst, height=0.55, color=AMBER, alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)
ax_c2.text(pct_ictl + 0.12, 0, f"{pct_ictl:.1f}% ({tk_ictl}/{off_ictl})", fontsize=FS_CELL,
           fontweight="bold", color=SLATE, va="center")
ax_c2.text(pct_itst + 0.12, 1, f"{pct_itst:.1f}% ({tk_taste}/{off_taste})", fontsize=FS_CELL,
           fontweight="bold", color=AMBER, va="center")

ax_c2.set_xlim(0, 5.8)
ax_c2.set_ylim(1.62, -0.58)
ax_c2.set_xticks([0, 1, 2, 3, 4, 5])
ax_c2.set_yticks([0, 1])
ax_c2.set_yticklabels(["Control", "Random taste"], fontsize=FS_TICK)
ax_c2.tick_params(axis="y", length=0, pad=3.0)
ax_c2.set_xlabel("Neglected-card pick share (%)", labelpad=2.0)

# =========================================================================
# Panel d1: One-step combat value lookahead outcome flips
# =========================================================================
clean_ax(ax_d1)
ax_d1.axvline(0, color=INK2, lw=0.85, zorder=3)
for gx in (-20, -10, 10, 20):
    ax_d1.axvline(gx, color=GRID, lw=0.55, zorder=0)

la_rows = [
    ("Boss fights\n(n = 163)", resc_safe, lost_safe, f"HP {hp_pol:.0f} to {hp_safe:.0f}"),
    ("Boss fights,\nsees random\noutcomes", resc_ora, lost_ora, ""),
    ("Full games\n(n = 100)", full_resc, full_lost, ""),
]

for i, (lbl, r_cnt, l_cnt, note_s) in enumerate(la_rows):
    ax_d1.barh(i, r_cnt, height=0.54, color=TEAL, alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)
    ax_d1.barh(i, -l_cnt, height=0.54, color=TERRA, alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)
    ax_d1.text(r_cnt + 0.9, i, f"+{r_cnt}", fontsize=FS_LAB, fontweight="bold", color=TEAL, va="center", ha="left")
    ax_d1.text(-l_cnt - 0.9, i, f"{MINUS}{l_cnt}", fontsize=FS_LAB, fontweight="bold",
               color=TERRA, va="center", ha="right")
    if note_s:
        ax_d1.text(-13.5, i, note_s, fontsize=FS_TICK, fontweight="bold", color=INK2, va="center", ha="right")

ax_d1.axhline(1.50, color=BORDER, lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)

# Top directional labels
ax_d1.text(-16, -0.58, "Turned win to loss", fontsize=FS_TICK, fontweight="bold", color=TERRA, ha="center", va="bottom")
ax_d1.text(11, -0.58, "Rescued loss", fontsize=FS_TICK, fontweight="bold", color=TEAL, ha="center", va="bottom")

ax_d1.set_xlim(-33.5, 19.5)
ax_d1.set_ylim(2.58, -0.58)
ax_d1.set_xticks([-20, -10, 0, 10])
ax_d1.set_xticklabels(["20", "10", "0", "10"])
ax_d1.set_yticks(range(len(la_rows)))
ax_d1.set_yticklabels([r[0] for r in la_rows], fontsize=FS_TICK)
ax_d1.tick_params(axis="y", length=0, pad=2.5)
ax_d1.set_xlabel("Fights or games flipped", labelpad=2.5)

# =========================================================================
# Panel d2: Full-run 100-seed paired win rate (68% -> 53%) + boss art
# =========================================================================
clean_ax(ax_d2)
for gy in (25, 50, 75):
    ax_d2.axhline(gy, color=GRID, lw=0.55, zorder=0)

xd2 = np.array([0.0, 1.0])
vals_d2 = [full_greedy_w, full_safe_w]
cols_d2 = [BLUE, TERRA]
for xi, vi, ci in zip(xd2, vals_d2, cols_d2):
    ax_d2.bar(xi, vi, width=0.54, color=ci, alpha=0.88, edgecolor="white", linewidth=0.6, zorder=3)
    ax_d2.text(xi, vi + 2.2, f"{vi}%", fontsize=FS_LAB, fontweight="bold", color=ci, ha="center", va="bottom")

ax_d2.annotate("", xy=(0.88, full_safe_w + 11.5), xytext=(0.12, full_greedy_w + 11.5),
               arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.22", lw=1.0, color=TERRA), zorder=4)
ax_d2.text(0.50, 82.5, f"{MINUS}{full_greedy_w - full_safe_w} points", fontsize=FS_LAB,
           fontweight="bold", color=TERRA, ha="center", va="bottom")

# Boss icons row at top
for bx, bfn in [(-0.02, "boss_queen.png"), (0.50, "boss_test_subject.png"), (1.02, "boss_aeonglass.png")]:
    add_sprite(ax_d2, bfn, (bx, 96.5), target_h_pt=14.5, zorder=6)

ax_d2.set_xlim(-0.52, 1.52)
ax_d2.set_ylim(0, 105)
ax_d2.set_xticks(xd2)
ax_d2.set_xticklabels(["Policy's\nchoice", "One-step\nlookahead"], fontsize=FS_TICK)
ax_d2.set_yticks([0, 25, 50, 75, 100])
ax_d2.set_ylabel("Full-game win rate (%)", labelpad=1.5)

# =========================================================================
# Panel letters
# =========================================================================
for lab, (gx, gy) in zip("abcd", [
    (0.014, 0.980), (0.415, 0.980),
    (0.014, 0.495), (0.452, 0.495),
]):
    fig.text(gx, gy, lab, fontsize=FS_LETTER, fontweight="bold", va="top", color=INK)

out_base = os.path.join(R, "paper", "figs", "ed_fig4_probes")
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out_base}.{ext}", dpi=350)
print(f"written {out_base}.{{png,pdf,svg}}")
print("b 5-char exposure:", [(r["char"], f"{r['tk_n']}/{r['off_n']}", f"{r['tk_c']}/{r['off_c']}", round(r["diff"], 2)) for r in rows_b])
print(f"c taste: {p_tst_c:.2f}% vs {p_ctl_c:.2f}% ({diff_taste:+.2f} pts), picks {pct_itst:.2f}% vs {pct_ictl:.2f}%")
print(f"d lookahead: safe +{resc_safe}/-{lost_safe} (HP {hp_pol:.1f}->{hp_safe:.1f}), oracle +{resc_ora}/-{lost_ora}, full {full_greedy_w}%->{full_safe_w}%")
