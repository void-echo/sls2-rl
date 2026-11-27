"""Figure 3: two real updates, taken apart.

The first update (v0.113) hurt through one rule and was absorbed within hours; the second (v0.114) rebalanced forty
cards of one character, its damage is spread over many cards, and it was not absorbed. Both could be reverted rule by
rule, both add up, and for both it was possible to say in advance who would be hurt. The two rows mirror each other,
column by column: who was hurt, which rules carry the damage, do the parts add up, how it was absorbed.

Row 1, v0.113 (57 cards, 69 relics, 35 monsters changed):
  a, untrained networks moved from v0.111 to v0.113 (1,000 paired A10 seeds per character): posterior ridges and
     dumbbells, with the aligned columns "share of pre-update runs whose deck holds a weakened card" and the loss.
  b, Ironclad, restoring the old rules one at a time (Forgotten Ritual alone +10.2 of the 9.7 lost points; all 78 regain
     more than was lost because the update's additions, which stay, are worth +2.8 to the old policy).
  c, twelve random half-updates: additive prediction from the single restorations against the measured recovery.
  d, absorbed locally and without forgetting: identity-embedding drift of 124 cards after adapting (the two changed
     cards rank first and second), and the pre-update vs the adapted network on v0.113 and on the old version.
Row 2, v0.114 (40 Fool cards changed):
  e, same form as a, six untrained networks moved from v0.113 to v0.114 on 2,000 same seeds each; only the Fool falls,
     as recorded before any network played v0.114.
  f, the Fool, restoring the old rules (same scale as b): all rules +34.2, the 40 cards +33.0, no single card above 4.5.
  g, six random complementary splits of the 62 revert units: the two halves sum to about 90% of the whole.
  h, top: sampled A10 win rate after each update across 3.2e8 steps (Ironclad after v0.113 rebounds above its 55.6%
     pre-update baseline within hours; the Fool after v0.114 stalls near 25% sampled / 31% greedy vs 46.1% / 55.3%
     before). Bottom: pick rates of the 12 buffed Fool pool cards before v0.114 (0.3% overall, 9 at 0%) vs after 3.2e8
     steps (standalone Serrated Arrows reaches 45.5%, while 8 of 12 buffed cards remain below 1%).

Sources: tmp/cold1115/res, tmp/ablate1115, tmp/semipatch1117/analyze.py, tmp/reverse1117,
paper/figdata/{prepatch_forecast_5char,drift_u30000,v0114_coldstart,forecast_v0114,v0114_reverts,
fool_v0114_mechanism}.json, paper/figdata/{iron_v2_main_before_v0113,iron_v3_main_after_v0113,
fool_v3_before_v0114,v4_fool_adapt.progress}.csv.
Art: tmp/game_assets/{fig14_v2,fig34,fig3_restyle}.

Run:
    python3 paper/figs/fig3_patch.py -> paper/figs/fig3_patch.{png,pdf,svg}
"""
import csv
import json
import math
import os
import re
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from PIL import Image
from scipy.stats import beta as beta_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GA = os.path.join(R, "tmp", "game_assets")
ASSET_V2_DIR = os.path.join(GA, "fig14_v2")
ASSET_34_DIR = os.path.join(GA, "fig34")
ASSET_3_DIR = os.path.join(GA, "fig3_restyle")
if not os.path.exists(os.path.join(ASSET_V2_DIR, "badge_char_ironclad.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_V2_DIR, "build_assets.py")], check=True, cwd=R)
if not os.path.exists(os.path.join(ASSET_34_DIR, "fullcard_forgotten_ritual.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_34_DIR, "make_assets.py")], check=True, cwd=R)
if not os.path.exists(os.path.join(ASSET_3_DIR, "card_wah_ki.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_3_DIR, "build_assets.py")], check=True, cwd=R)


def register_helvetica_bold():
    """matplotlib reads only the first face of Helvetica.ttc, so bold lettering silently falls back to regular.
    The bold face is extracted once into tmp/ and registered; without the system font the figure stays regular."""
    dst = os.path.join(ASSET_3_DIR, "Helvetica-Bold.ttf")
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

# Palette (STYLE.md)
BLUE = "#2166ac"
TERRA = "#c85228"
TEAL = "#1b8a6b"
AMBER = "#c98a1b"
AMBER_DEEP = "#8a5a10"
VIOLET = "#5e4cc4"
VIOLET_LIGHT = "#a99ee6"
PLUM = "#b85682"
SLATE = "#64748b"
DARK = "#334155"
INK = "#14181f"
INK2 = "#4a5568"
ZEBRA = "#f4f6f9"
BORDER = "#dce3ec"
OUTER_BORDER = "#e2e8f0"
GRID = "#edf2f7"

CHAR_COLORS = {"Ironclad": BLUE, "Defect": TERRA, "Regent": TEAL, "Silent": AMBER, "Necrobinder": PLUM,
               "Fool": VIOLET}
CHAR_BADGES = {
    "Ironclad": "badge_char_ironclad.png",
    "Silent": "badge_char_silent.png",
    "Defect": "badge_char_defect.png",
    "Necrobinder": "badge_char_necrobinder.png",
    "Regent": "badge_char_regent.png",
    "Fool": "badge_char_fool.png",
}

FS_TICK = 6.8
FS_AX = 7.6
FS_LAB = 7.4
FS_CELL = 6.9
FS_LETTER = 11.0
MINUS = "−"

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
    "axes.unicode_minus": True,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

FIG_W, FIG_H = 8.7, 6.1
PT_W, PT_H = FIG_W * 72.0, FIG_H * 72.0


def box(x0_pt, y0_pt, w_pt, h_pt):
    return [x0_pt / PT_W, y0_pt / PT_H, w_pt / PT_W, h_pt / PT_H]


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.6, pad=2.0)


def load_rgba_clean(path, target_h_px, rotate=0.0):
    im = Image.open(path).convert("RGBA")
    if rotate:
        im = im.rotate(rotate, resample=Image.Resampling.BICUBIC, expand=True)
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


def add_sprite(ax, folder, fn, xy, target_h_pt, xycoords="data", offset=None, align=(0.5, 0.5), zorder=6,
               rotate=0.0):
    """Place an image whose (unrotated) height is target_h_pt points."""
    path = os.path.join(folder, fn)
    if rotate:
        h0 = Image.open(path).height
        h1 = Image.open(path).convert("RGBA").rotate(rotate, expand=True).height
        target_h_pt = target_h_pt * h1 / h0
    arr, h_px = load_rgba_clean(path, target_h_pt * 5.0, rotate=rotate)
    oi = OffsetImage(arr, zoom=target_h_pt / h_px, interpolation="bilinear", resample=True)
    kw = dict(xycoords=xycoords, box_alignment=align, frameon=False, pad=0, zorder=zorder, annotation_clip=False)
    if offset is not None:
        kw.update(xybox=offset, boxcoords="offset points")
    ab = AnnotationBbox(oi, xy, **kw)
    ab.set_clip_on(False)
    ax.add_artist(ab)


def badge(ax, name, xy, h=18.0, xycoords="data", offset=None, zorder=8):
    add_sprite(ax, ASSET_V2_DIR, CHAR_BADGES[name], xy, h, xycoords=xycoords, offset=offset, zorder=zorder)


def rate(path):
    s = json.load(open(os.path.join(R, path)))["summary"]["all"]
    return 100.0 * s["win_rate"], s["n"]


def draw_beta_ridge(ax, y_base, p_pct, n, col, max_h=0.42, alpha_fill=0.28, lw=0.75, zorder=2):
    k = p_pct / 100.0 * n
    a_p, b_p = k + 1.0, (n - k) + 1.0
    lo, hi = beta_dist.ppf([0.002, 0.998], a_p, b_p)
    xs = np.linspace(lo * 100.0, hi * 100.0, 120)
    pdf = beta_dist.pdf(xs / 100.0, a_p, b_p)
    ys = y_base - (pdf / pdf.max()) * max_h
    ax.fill_between(xs, y_base, ys, color=col, alpha=alpha_fill, lw=0, zorder=zorder)
    ax.plot(xs, ys, color=col, lw=lw, zorder=zorder + 0.1)


def fmt1(x):
    """One decimal, half away from zero, with a real minus sign."""
    s = str(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    if not s.startswith("-") and float(s) != 0.0:
        s = "+" + s
    return s.replace("-", MINUS)


def mix(c0, c1, t):
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(a + (b - a) * float(np.clip(t, 0.0, 1.0)))


def read_csv(path):
    return list(csv.DictReader(open(os.path.join(R, path))))


def trailing_mean(y, k):
    c = np.cumsum(np.insert(np.asarray(y, dtype=float), 0, 0.0))
    idx = np.arange(1, len(y) + 1)
    lo = np.maximum(0, idx - k)
    return (c[idx] - c[lo]) / (idx - lo)


def beeswarm_2d(xs, y_center, bin_w=1.0, dx=0.42, dy=0.14):
    """Deterministic compact 2D pebble offsets so exact-zero ties form a neat cluster without vertical stacking."""
    order = sorted(range(len(xs)), key=lambda idx: xs[idx])
    out_x = [float(x) for x in xs]
    out_y = [float(y_center)] * len(xs)
    # precomputed symmetric (col, row) slots around (0, 0) for up to 12 tied points
    slots = [(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (1.0, -0.5), (1.0, 0.5),
             (-0.85, -0.5), (-0.85, 0.5), (1.0, -1.4), (1.0, 1.4),
             (0.0, -1.8), (0.0, 1.8), (1.8, 0.0)]
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] - xs[order[i]] <= bin_w:
            j += 1
        m = j - i
        if m > 1:
            xm = float(np.mean([xs[idx] for idx in order[i:j]]))
            for k_pos, idx in enumerate(order[i:j]):
                sx, sy = slots[k_pos % len(slots)]
                out_x[idx] = xm + sx * dx
                out_y[idx] = y_center + sy * dy
        i = j
    return out_x, out_y


# =========================================================================
# Data
# =========================================================================
pre_fc = json.load(open(os.path.join(R, "paper", "figdata", "prepatch_forecast_5char.json")))
fc_by_char = {row["character"]: row for row in pre_fc["summary_table"]}
rho_deck = pre_fc["correlations"]["all_nerfed_cards_deck_frac"]["vs_delta_paper"]["spearman_rho"]

v114 = json.load(open(os.path.join(R, "paper", "figdata", "v0114_coldstart.json")))
fc114 = json.load(open(os.path.join(R, "paper", "figdata", "forecast_v0114.json")))["characters"]
rev114 = json.load(open(os.path.join(R, "paper", "figdata", "v0114_reverts.json")))
REV = rev114["reverts"]
mech114 = json.load(open(os.path.join(R, "paper", "figdata", "fool_v0114_mechanism.json")))

CHARS_A = [("iron", "Ironclad"), ("silent", "Silent"), ("defect", "Defect"), ("necro", "Necrobinder"),
           ("regent", "Regent")]
CHARS_E = CHARS_A + [("fool", "Fool")]

ROWS_A = []
for k, name in CHARS_A:
    v_old, n_old = rate(f"tmp/cold1115/res/{k}_v2_a10.json")
    v_new, n_new = rate(f"tmp/cold1115/res/{k}_v3_a10.json")
    ROWS_A.append(dict(name=name, old=(v_old, n_old), new=(v_new, n_new),
                       expo=100.0 * fc_by_char[name]["all_nerfed_cards_deck_frac"],
                       delta=fc_by_char[name]["cold_start_delta_pts"]))
ROWS_E = []
for k, name in CHARS_E:
    r = v114[k]
    ROWS_E.append(dict(name=name, old=(r["v113"], r["n"]), new=(r["v114"], r["n"]),
                       expo=100.0 * fc114[name]["deck_frac_weakened"], delta=r["delta"]))

# =========================================================================
# Canvas and layout (points; 626.4 x 439.2). Two mirrored rows, four aligned columns.
# =========================================================================
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
fig.add_artist(FancyBboxPatch(
    (0.004, 0.006), 0.992, 0.988, transform=fig.transFigure,
    boxstyle="round,pad=0.0,rounding_size=0.012",
    facecolor="none", edgecolor=OUTER_BORDER, linewidth=0.8, zorder=0,
))

HEAD = 17.0                     # header band above a and e (pt)
R1_Y0, R1_Y1 = 253.0, 421.0     # plot band of row 1 (x axes at R1_Y0)
R2_Y0, R2_Y1 = 29.5, 203.5      # plot band of row 2
RH_A = (R1_Y1 - R1_Y0 - HEAD) / 5.0
RH_E = (R2_Y1 - R2_Y0 - HEAD) / 6.0

# columns (x in pt)
X_A0, W_A = 86.0, 86.0          # ridge plots a, e
X_T0, W_T = 175.0, 58.0         # value columns a, e
X_B0, W_B = 300.0, 92.0         # bars b, f (same scale)
X_C0, W_C = 421.0, 70.0         # c, g
X_D0, W_D = 528.0, 90.0         # d/d2, h/h2

ax_a = fig.add_axes(box(X_A0, R1_Y0, W_A, R1_Y1 - R1_Y0))
ax_at = fig.add_axes(box(X_T0, R1_Y0, W_T, R1_Y1 - R1_Y0))
ax_e = fig.add_axes(box(X_A0, R2_Y0, W_A, R2_Y1 - R2_Y0))
ax_et = fig.add_axes(box(X_T0, R2_Y0, W_T, R2_Y1 - R2_Y0))
ax_b = fig.add_axes(box(X_B0, R1_Y0, W_B, R1_Y1 - R1_Y0 - 6.0))
ax_f = fig.add_axes(box(X_B0, R2_Y0, W_B, R2_Y1 - R2_Y0 - 6.0))
ax_c = fig.add_axes(box(X_C0, R1_Y0, W_C, R1_Y1 - R1_Y0))
ax_g = fig.add_axes(box(X_C0, R2_Y0, W_C, R2_Y1 - R2_Y0))

D_SPLIT = 60.0                  # no-forgetting strip at bottom of d, with clear gap below "124 cards, ranked"
ax_d = fig.add_axes(box(X_D0, R1_Y0 + D_SPLIT, W_D, R1_Y1 - R1_Y0 - D_SPLIT))
ax_d2 = fig.add_axes(box(X_D0, R1_Y0, W_D, 31.0))

H_SPLIT = 63.0                  # buff-lockout mechanism strip at bottom of h, mirroring d/d2
ax_h = fig.add_axes(box(X_D0, R2_Y0 + H_SPLIT, W_D, R2_Y1 - R2_Y0 - H_SPLIT))
ax_h2 = fig.add_axes(box(X_D0, R2_Y0, W_D, 34.0))

XLIM_AE = (16.0, 77.0)
XLIM_BF = (-2.5, 37.5)


# =========================================================================
# Row labels at the left edge
# =========================================================================
def row_label(y0, y1, version, what):
    ym = 0.5 * (y0 + y1)
    fig.text(8.0 / PT_W, ym / PT_H, version, rotation=90, fontsize=8.6, fontweight="bold", color=INK,
             ha="center", va="center")
    fig.text(16.2 / PT_W, ym / PT_H, what, rotation=90, fontsize=6.7, color=INK2, ha="center", va="center")


row_label(R1_Y0 - 20.0, R1_Y1 + 8.0, "v0.113", "57 cards, 69 relics, 35 monsters changed")
row_label(R2_Y0 - 20.0, R2_Y1 + 8.0, "v0.114", "40 Fool cards changed")


# =========================================================================
# Panels a and e: same form, same scale
# =========================================================================
def ridge_table(ax, axt, rows, rh, v_old_lab, v_new_lab):
    n = len(rows)
    y_top = -0.5 - HEAD / rh
    for a in (ax, axt):
        a.set_ylim(n - 0.5, y_top)
    clean_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(*XLIM_AE)
    ax.spines["bottom"].set_bounds(*XLIM_AE)
    axt.set_xlim(0.0, 2.0)
    axt.axis("off")

    for gx in (20, 30, 40, 50, 60, 70):
        ax.plot([gx, gx], [n - 0.5, -0.5], color=GRID, lw=0.55, zorder=0)
    for i in range(n):
        if i % 2 == 0:
            ax.add_patch(Rectangle((XLIM_AE[0], i - 0.5), XLIM_AE[1] - XLIM_AE[0], 1.0, facecolor=ZEBRA,
                                   edgecolor="none", zorder=0))
    yh = y_top + 0.5 * HEAD / rh
    ax.scatter([XLIM_AE[0] + 2.5], [yh], s=22, color=BLUE, edgecolor="white", linewidth=0.5, clip_on=False, zorder=5)
    ax.text(XLIM_AE[0] + 4.8, yh, v_old_lab, fontsize=FS_LAB, fontweight="bold", color=BLUE, va="center", ha="left")
    ax.scatter([XLIM_AE[0] + 30.5], [yh], s=22, color=TERRA, edgecolor="white", linewidth=0.5, clip_on=False,
               zorder=5)
    ax.text(XLIM_AE[0] + 32.8, yh, v_new_lab, fontsize=FS_LAB, fontweight="bold", color=TERRA, va="center",
            ha="left")
    axt.text(0.5, yh, "Weakened\ncard (%)", fontsize=FS_TICK, fontweight="bold", color=AMBER_DEEP,
             ha="center", va="center", linespacing=0.95)
    axt.text(1.5, yh, "Δ\n(points)", fontsize=FS_TICK, fontweight="bold", color=INK,
             ha="center", va="center", linespacing=0.95)

    for i, r in enumerate(rows):
        (vo, no), (vn, nn) = r["old"], r["new"]
        yb = i + 0.25
        draw_beta_ridge(ax, yb, vo, no, BLUE, max_h=0.46)
        draw_beta_ridge(ax, yb, vn, nn, TERRA, max_h=0.46)
        ax.plot([vo, vn], [yb, yb], color=INK2, lw=1.1, zorder=4)
        ax.scatter([vo], [yb], s=22, color=BLUE, edgecolor="white", linewidth=0.55, zorder=5)
        ax.scatter([vn], [yb], s=22, color=TERRA, edgecolor="white", linewidth=0.55, zorder=6)
        badge(ax, r["name"], (0.0, i), 18.0, xycoords=("axes fraction", "data"), offset=(-11.0, 0.0))
        ax.annotate(r["name"], (0.0, i), xycoords=("axes fraction", "data"), xytext=(-22.5, 0.0),
                    textcoords="offset points", fontsize=FS_LAB, color=INK, ha="right", va="center")
        t_exp = (r["expo"] / 100.0) ** 0.6
        axt.add_patch(Rectangle((0.03, i - 0.47), 0.94, 0.94, facecolor=mix("#fbf4e6", AMBER, t_exp),
                                edgecolor="none", zorder=1))
        axt.text(0.5, i, f"{r['expo']:.1f}", fontsize=FS_CELL, fontweight="bold" if r["expo"] > 30 else "normal",
                 color="white" if t_exp > 0.55 else INK, ha="center", va="center", zorder=2)
        d = r["delta"]
        t_d = min(1.0, max(0.0, -d) / 12.0) ** 0.75
        axt.add_patch(Rectangle((1.03, i - 0.47), 0.94, 0.94, facecolor=mix("#f1f4f8", TERRA, t_d),
                                edgecolor="none", zorder=1))
        axt.text(1.5, i, fmt1(d), fontsize=FS_CELL, fontweight="bold" if t_d > 0.4 else "normal",
                 color="white" if t_d > 0.55 else INK, ha="center", va="center", zorder=2)

    ax.set_yticks([])
    ax.set_xticks([20, 30, 40, 50, 60, 70])
    ax.set_xlabel("Win rate at Ascension 10 (%)", labelpad=2.0)


ridge_table(ax_a, ax_at, ROWS_A, RH_A, "v0.111", "v0.113")
ax_at.text(1.0, 4.60, "rank correlation", fontsize=FS_LAB, fontweight="bold", color=AMBER_DEEP,
           ha="center", va="top")
ax_at.text(1.0, 4.60 + 9.5 / RH_A, f"{MINUS}{abs(rho_deck):.2f}", fontsize=FS_LAB + 0.5, fontweight="bold",
           color=AMBER_DEEP, ha="center", va="top")

ridge_table(ax_e, ax_et, ROWS_E, RH_E, "v0.113", "v0.114")
ax_e.text(18.5, 2.5, "forecast: only\nthe Fool falls", fontsize=FS_LAB, fontweight="bold",
          color=VIOLET, ha="left", va="center", linespacing=1.02, zorder=9)


# =========================================================================
# Panels b and f: restoring old rules, points regained on one scale
# =========================================================================
def bar_panel(ax, rows, loss, rh_pad=0.5):
    """rows: (label, gain, colour, bold) with gain either float (bar) or list of floats (dots in one row)."""
    clean_ax(ax)
    ax.spines["left"].set_visible(False)   # the zero line is the baseline
    n = len(rows)
    ax.set_xlim(*XLIM_BF)
    ax.set_ylim(n - 1 + rh_pad, -rh_pad)
    for gx in (10, 20, 30):
        ax.axvline(gx, color=GRID, lw=0.55, zorder=0)
    for i in range(n):
        if i % 2 == 0:
            ax.add_patch(Rectangle((XLIM_BF[0], i - 0.5), XLIM_BF[1] - XLIM_BF[0], 1.0, facecolor=ZEBRA,
                                   edgecolor="none", zorder=0))
    ax.axvline(0.0, color=INK2, lw=0.6, zorder=2)
    for i, (lab, val, col, bold) in enumerate(rows):
        if isinstance(val, list):
            ax.scatter(val, [i] * len(val), s=18, color=col, edgecolor=VIOLET, linewidth=0.5, alpha=0.9, zorder=4)
            continue
        ax.barh(i, val, height=0.64, color=col, edgecolor="none", zorder=3)
        if val > 12.0:
            ax.text(val - 0.6, i, fmt1(val), va="center", ha="right", fontsize=FS_CELL, fontweight="bold",
                    color="white", zorder=5)
        else:
            ax.text(max(val, 0.0) + 0.6, i, fmt1(val), va="center", ha="left", fontsize=FS_CELL,
                    fontweight="bold" if bold else "normal", color=col if bold else INK, zorder=5)
    ax.axvline(loss, color=TERRA, lw=1.0, ls=(0, (3, 2)), zorder=2.5)
    ax.text(loss, -rh_pad, "loss", fontsize=FS_LAB, fontweight="bold", color=TERRA, ha="center", va="bottom")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS_LAB)
    ax.tick_params(axis="y", length=0, pad=3.0)
    ax.set_xticks([0, 10, 20, 30])


# b: Ironclad on v0.113
b0 = 52.4
shock_b = 62.1 - 52.4
rows_b = []
for i, (lab, fn_b) in enumerate([("Forgotten Ritual", "iron_fr"), ("Expect a Fight", "iron_eaf"),
                                 ("Relics", "iron_relics"), ("Shared cards", "iron_shared"),
                                 ("Power rules", "iron_powers"), ("All 78 changes", "iron_all")]):
    v, _ = rate(f"tmp/ablate1115/{fn_b}.json")
    col = BLUE if i < 2 else (DARK if fn_b == "iron_all" else SLATE)
    rows_b.append((lab, v - b0, col, i < 2 or fn_b == "iron_all"))
bar_panel(ax_b, rows_b, shock_b)
ax_b.set_xlabel("Points regained", labelpad=2.0)
new_content_gain = rate("tmp/ablate1115/iron_all.json")[0] - rate("tmp/cold1115/res/iron_v2_a10.json")[0]
ax_b.text(14.2, 5.0, "new content\nalso helps", fontsize=FS_LAB, fontweight="bold", color=DARK, ha="left",
          va="center", linespacing=1.0)
add_sprite(ax_b, ASSET_34_DIR, "fullcard_forgotten_ritual.png", (20.2, 2.25), 44.0, zorder=7)
add_sprite(ax_b, ASSET_34_DIR, "fullcard_expect_a_fight.png", (32.0, 2.25), 44.0, zorder=7)

# f: the Fool on v0.114
GROUPS_F = [("All rules", "all"), ("Fool's 40 cards", "cards"), ("Tarot cards", "tokens"), ("Powers", "powers"),
            ("Relics", "relics"), ("Agent's view", "obs")]
SINGLES_F = ["motif", "wah_cho_ki", "wah_symphony", "wah_ki", "cho_firm", "cho_ready", "wah_hammer",
             "melodic_shot", "bounce", "ki_rest", "cho_barbs"]
SINGLE_NAMES = {"motif": "Motif", "wah_cho_ki": "Wah Cho Ki", "wah_symphony": "Wah Symphony", "wah_ki": "Wah Ki",
                "cho_firm": "Cho Firm"}
singles = sorted(SINGLES_F, key=lambda t: -REV[t]["gain"])
named = [t for t in singles if t in SINGLE_NAMES]
others = [t for t in singles if t not in SINGLE_NAMES]
rows_f = []
for lab, tag in GROUPS_F:
    col = DARK if tag == "all" else (VIOLET if tag == "cards" else SLATE)
    rows_f.append((lab, REV[tag]["gain"], col, tag in ("all", "cards")))
for tag in named:
    rows_f.append((SINGLE_NAMES[tag], REV[tag]["gain"], VIOLET_LIGHT, False))
rows_f.append((f"{len(others)} other cards", [REV[t]["gain"] for t in others], VIOLET_LIGHT, False))
bar_panel(ax_f, rows_f, REV["all"]["gain"])
ax_f.set_xlabel("Points regained", labelpad=2.0)
largest_single = max(REV[t]["gain"] for t in SINGLES_F)
ax_f.text(13.5, len(GROUPS_F) + 0.5, f"no card\nabove {largest_single:.1f}", fontsize=FS_LAB, fontweight="bold",
          color=VIOLET, ha="left", va="center", linespacing=1.0)
for j, (fn, cx) in enumerate([("card_motif.png", 15.1), ("card_wah_cho_ki.png", 21.9), ("card_wah_ki.png", 28.7)]):
    add_sprite(ax_f, ASSET_3_DIR, fn, (cx, len(rows_f) - 2.1), 30.8, zorder=7 + j)

# =========================================================================
# Panel c: twelve random half-updates add up
# =========================================================================
clean_ax(ax_c)
out_semi = subprocess.run([sys.executable, os.path.join(R, "tmp", "semipatch1117", "analyze.py")],
                          capture_output=True, text=True, cwd=R).stdout
pts_c = [(int(m.group(1)), int(m.group(2)), float(m.group(4)), float(m.group(3)))
         for m in re.finditer(r"fr=(\d)\s+eaf=(\d).*?shock undone ([+-][\d.]+).*?additive prediction ([+-][\d.]+)",
                              out_semi)]
r_c = float(np.corrcoef([p for _, _, p, _ in pts_c], [o for _, _, _, o in pts_c])[0, 1])
LIM_C = (-1.5, 15.0)
ax_c.set_xlim(*LIM_C)
ax_c.set_ylim(*LIM_C)
for gv in (0, 5, 10):
    ax_c.axhline(gv, color=GRID, lw=0.55, zorder=0)
    ax_c.axvline(gv, color=GRID, lw=0.55, zorder=0)
t = np.linspace(*LIM_C, 50)
ax_c.fill_between(t, t - 1.4, t + 1.4, color="#94a3b8", alpha=0.12, lw=0, zorder=1)
ax_c.plot(LIM_C, LIM_C, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=2)
for fr in (0, 1):
    sub = [(p, o) for f, _, p, o in pts_c if f == fr]
    xs = []
    for i_p, (px, py) in enumerate(sub):
        nudge = 0.0
        for i_q, (qx, qy) in enumerate(sub):
            if i_p != i_q and abs(px - qx) < 0.15 and abs(py - qy) < 0.35:
                nudge = -0.2 if py < qy else 0.2
        xs.append(px + nudge)
    ax_c.scatter(xs, [o for _, o in sub], s=30, color=BLUE if fr else SLATE, edgecolor="white", linewidth=0.6,
                 zorder=4)
ax_c.text(-0.7, 13.0, "with Forgotten\nRitual", fontsize=FS_LAB, fontweight="bold", color=BLUE,
          ha="left", va="center", linespacing=1.0)
ax_c.text(14.6, 4.6, f"r = {r_c:.2f}", fontsize=FS_LAB + 0.6, fontweight="bold", color=INK, ha="right",
          va="center")
ax_c.set_xticks([0, 5, 10])
ax_c.set_yticks([0, 5, 10])
ax_c.set_xlabel("Sum of single restorations", labelpad=2.0)
ax_c.set_ylabel("Measured recovery (points)", labelpad=2.0)

# =========================================================================
# Panel g: six random complementary splits of v0.114 into two halves
# =========================================================================
clean_ax(ax_g)
whole = REV["all"]["gain"]
halves = [(REV[f"h{i}a"]["gain"], REV[f"h{i}b"]["gain"]) for i in range(6)]
share = np.mean([a + b for a, b in halves]) / whole
xs_g = [0.0] + [1.3 + i for i in range(6)]
ax_g.set_xlim(-0.7, 6.95)
ax_g.set_ylim(0.0, 41.0)
for gy in (10, 20, 30):
    ax_g.axhline(gy, color=GRID, lw=0.55, zorder=0)
ax_g.bar(xs_g[0], whole, width=0.72, color=DARK, edgecolor="none", zorder=3)
for x, (a, b) in zip(xs_g[1:], halves):
    lo_, hi_ = (a, b) if a >= b else (b, a)
    ax_g.bar(x, lo_, width=0.72, color=VIOLET, edgecolor="white", linewidth=0.5, zorder=3)
    ax_g.bar(x, hi_, bottom=lo_, width=0.72, color=VIOLET_LIGHT, edgecolor="white", linewidth=0.5, zorder=3)
ax_g.axhline(whole, color=TERRA, lw=1.0, ls=(0, (3, 2)), zorder=2.5)
ax_g.text(3.8, 37.6, f"halves sum to\n{100 * share:.0f}% of the whole", fontsize=FS_LAB, fontweight="bold",
          color=VIOLET, ha="center", va="center", linespacing=1.0)
ax_g.set_xticks(xs_g)
ax_g.set_xticklabels(["all"] + [str(i + 1) for i in range(6)])
ax_g.tick_params(axis="x", length=0)
ax_g.set_yticks([0, 10, 20, 30])
ax_g.set_xlabel("Random split in two halves", labelpad=2.0)
ax_g.set_ylabel("Points regained", labelpad=2.0)

# =========================================================================
# Panel d: absorbed locally (embedding drift of 124 cards) and without forgetting
# =========================================================================
clean_ax(ax_d)
dr = json.load(open(os.path.join(R, "paper", "figdata", "drift_u30000.json")))["rows"]
vals_d = [r["drift"] for r in dr]
order_d = sorted(range(len(dr)), key=lambda idx: -vals_d[idx])
sorted_vals_d = [vals_d[idx] for idx in order_d]
ax_d.set_xlim(-3.0, 126.0)
ax_d.set_ylim(0.0, 0.27)
for gy in (0.1, 0.2):
    ax_d.axhline(gy, color=GRID, lw=0.5, zorder=0)
ax_d.bar(range(len(order_d)), sorted_vals_d, width=0.82, color="#9aa8b9", edgecolor="none", zorder=2)
peaks_d = {}
for rank, idx in enumerate(order_d):
    cid = dr[idx]["card"]
    if cid in ("CARD.FORGOTTEN_RITUAL", "CARD.EXPECT_A_FIGHT"):
        ax_d.bar(rank, vals_d[idx], width=1.6, color=BLUE, edgecolor="none", zorder=4)
        ax_d.scatter([rank], [vals_d[idx]], s=20, color=BLUE, edgecolor="white", linewidth=0.5, zorder=5)
        peaks_d[cid] = (rank, vals_d[idx])
for cid, fn, cx, cy, ex, ey in [("CARD.FORGOTTEN_RITUAL", "fullcard_forgotten_ritual.png", 36.0, 0.205, 19.0, 0.225),
                                ("CARD.EXPECT_A_FIGHT", "fullcard_expect_a_fight.png", 86.0, 0.128, 69.0, 0.115)]:
    rk, val = peaks_d[cid]
    add_sprite(ax_d, ASSET_34_DIR, fn, (cx, cy), 35.0, zorder=7)
    ax_d.plot([rk + 1.2, ex], [val, ey], color=BLUE, lw=0.7, zorder=3)
ax_d.set_xticks([])
ax_d.set_yticks([0.0, 0.1, 0.2])
ax_d.set_yticklabels(["0", "0.1", "0.2"])
ax_d.set_xlabel("124 cards, ranked", labelpad=2.5)
ax_d.set_ylabel("Change of the card's\nlearned representation", labelpad=2.0, linespacing=1.0)

clean_ax(ax_d2)
pre_v113, ada_v113 = b0, 66.2
pre_v111 = rate("tmp/cold1115/res/iron_v2_a10.json")[0]
ada_v111 = rate("tmp/reverse1117/iron_new_v2engine.json")[0]
XL_D2 = (37.0, 72.0)
ax_d2.set_xlim(*XL_D2)
ax_d2.set_ylim(1.55, -0.55)
for gx in (40, 50, 60, 70):
    ax_d2.axvline(gx, color=GRID, lw=0.5, zorder=0)
for yy, (vo, vn) in enumerate([(pre_v113, ada_v113), (pre_v111, ada_v111)]):
    ax_d2.plot([vo, vn], [yy, yy], color=INK2, lw=1.0, zorder=3)
    ax_d2.scatter([vo], [yy], s=22, color=BLUE, edgecolor="white", linewidth=0.5, zorder=4)
    ax_d2.scatter([vn], [yy], s=22, color=TEAL, edgecolor="white", linewidth=0.5, zorder=5)
ax_d2.text(min(pre_v111, ada_v111) - 1.5, 1.0, "no forgetting", fontsize=7.0, fontweight="bold",
           color=TEAL, ha="right", va="center")
for x0, col, lab in ((XL_D2[0] + 1.0, BLUE, "before"), (XL_D2[0] + 16.0, TEAL, "adapted")):
    ax_d2.scatter([x0], [-0.95], s=16, color=col, edgecolor="white", linewidth=0.4, clip_on=False, zorder=5)
    ax_d2.text(x0 + 1.4, -0.95, lab, fontsize=FS_TICK, fontweight="bold", color=col, ha="left", va="center")
ax_d2.set_yticks([0, 1])
ax_d2.set_yticklabels(["v0.113", "v0.111"], fontsize=FS_TICK)
ax_d2.tick_params(axis="y", length=0, pad=2.0)
ax_d2.set_xticks([40, 50, 60, 70])
ax_d2.set_xlabel("Win rate (%)", labelpad=2.0)

# =========================================================================
# Panel h: top = recovery across 3.2e8 steps; bottom = 12 buffed Fool cards
# =========================================================================
clean_ax(ax_h)
SMOOTH = 50
STEPS_PER_UPDATE = 131072
iron_before = read_csv("paper/figdata/iron_v2_main_before_v0113.csv")[-500:]
iron_after = [q for q in read_csv("paper/figdata/iron_v3_main_after_v0113.csv")
              if int(q["w_episodes"]) >= 100]
iron_pre = 100.0 * np.mean([float(q["w_win_rate"]) for q in iron_before])
fork_iron = int(iron_before[-1]["env_steps"])
x_i = np.array([(int(q["env_steps"]) - fork_iron) / 1e8 for q in iron_after])
y_i = trailing_mean([100.0 * float(q["w_win_rate"]) for q in iron_after], SMOOTH)

fool_before = read_csv("paper/figdata/fool_v3_before_v0114.csv")
fool_pre = 100.0 * np.mean([float(q["w_win_rate"]) for q in fool_before])
fool = read_csv("paper/figdata/v4_fool_adapt.progress.csv")[2:]
FORK_FOOL = 12660
x_f = np.array([(int(q["update"]) - FORK_FOOL) * STEPS_PER_UPDATE / 1e8 for q in fool])
y_f = trailing_mean([100.0 * float(q["w_win_rate"]) for q in fool], SMOOTH)

x_end = float(x_f[-1])
keep = x_i <= x_end
x_i, y_i = x_i[keep], y_i[keep]

ax_h.set_xlim(0.0, 3.82)
ax_h.set_ylim(10.0, 72.0)
for gy in (20, 40, 60):
    ax_h.axhline(gy, color=GRID, lw=0.55, zorder=0)
ax_h.plot([0.0, float(x_i[-1])], [iron_pre, iron_pre], color=BLUE, lw=0.9, ls=(0, (3, 2)), zorder=2)
ax_h.plot([0.0, x_end], [fool_pre, fool_pre], color=VIOLET, lw=0.9, ls=(0, (3, 2)), zorder=2)

S0 = SMOOTH // 5
ax_h.plot(x_i[S0:], y_i[S0:], color=BLUE, lw=1.2, zorder=4)
ax_h.plot(x_f[S0:], y_f[S0:], color=VIOLET, lw=1.2, zorder=4)
badge(ax_h, "Ironclad", (float(x_i[-1]), float(y_i[-1])), 15.0, offset=(7.8, 0.0), zorder=8)
badge(ax_h, "Fool", (x_end, float(y_f[-1])), 15.0, offset=(7.8, 0.0), zorder=8)

fool_g_pre = v114["fool"]["v114"]
fool_g_post = mech114["summary"]["post_u15050_greedy_2000_win_pct"]
ax_h.text(0.08, 67.5, "v0.113: back in hours", fontsize=7.2, fontweight="bold", color=BLUE,
          ha="left", va="center")
ax_h.text(0.08, 35.5, f"v0.114: slow\n({fool_g_pre:.0f}% to {fool_g_post:.0f}% greedy)",
          fontsize=7.0, fontweight="bold", color=VIOLET, ha="left", va="center", linespacing=1.02)
ax_h.set_xticks([0, 1, 2, 3])
ax_h.set_xticklabels(["0", "1", "2", "3"])
ax_h.set_yticks([20, 40, 60])
ax_h.set_xlabel(r"Steps after update ($\mathregular{10^8}$)", labelpad=2.0)
ax_h.set_ylabel("Sampled win rate (%)", labelpad=2.0)

# Bottom of h: why v0.114 stalls — 12 buffed Fool pool cards before vs after 3.2e8 steps
clean_ax(ax_h2)
buffed_cards = mech114["buffed_cards"]
pre_pick = [c["pre_pick_pct"] for c in buffed_cards]
post_pick = [c["post_pick_pct"] for c in buffed_cards]
pre_overall = mech114["summary"]["buffed_pre_pick_pct"]
n_zero_pre = mech114["summary"]["n_buffed_zero_pick"]
n_lt1_post = sum(1 for p in post_pick if p < 1.0)
serrated = max(buffed_cards, key=lambda c: c["post_pick_pct"])

XL_H2 = (-2.8, 53.0)
ax_h2.set_xlim(*XL_H2)
ax_h2.set_ylim(1.60, -0.60)
for gx in (0, 20, 40):
    ax_h2.axvline(gx, color=GRID, lw=0.5, zorder=0)

xs_pre, ys_pre = beeswarm_2d(pre_pick, 0.0)
xs_post, ys_post = beeswarm_2d(post_pick, 1.0)
ax_h2.scatter(xs_pre, ys_pre, s=15, facecolor="white", edgecolor=BLUE, linewidth=0.85, zorder=4)
ax_h2.scatter(xs_post, ys_post, s=17, facecolor=VIOLET, edgecolor="white", linewidth=0.45, zorder=5)

ax_h2.text(3.6, -0.02, f"{pre_overall:.1f}% ({n_zero_pre} zero)", fontsize=6.8, fontweight="bold",
           color=BLUE, ha="left", va="center")
ax_h2.text(9.0, 1.05, f"{n_lt1_post}/12 <1%", fontsize=6.8, fontweight="bold",
           color=VIOLET, ha="left", va="center")
sx = serrated["post_pick_pct"]
ax_h2.text(52.5, 0.04, "Serrated\nArrows", fontsize=6.7, fontweight="bold", color=VIOLET,
           ha="right", va="center", linespacing=0.95)
ax_h2.plot([sx, sx], [0.52, 0.80], color=VIOLET, lw=0.65, zorder=3)

ax_h2.set_yticks([0, 1])
ax_h2.set_yticklabels(["before", "adapted"], fontsize=FS_TICK)
ax_h2.tick_params(axis="y", length=0, pad=2.0)
ax_h2.set_xticks([0, 20, 40])
ax_h2.set_xlabel("Strengthened cards taken (%)", labelpad=2.0, x=0.44)

# =========================================================================
# Panel letters
# =========================================================================
LET_Y1, LET_Y2 = PT_H - 5.0, R2_Y1 + 20.5
for lab, (gx_pt, gy_pt) in zip("abcdefgh", [
        (24.0, LET_Y1), (236.0, LET_Y1), (398.0, LET_Y1), (499.0, LET_Y1),
        (24.0, LET_Y2), (236.0, LET_Y2), (398.0, LET_Y2), (499.0, LET_Y2)]):
    fig.text(gx_pt / PT_W, gy_pt / PT_H, lab, fontsize=FS_LETTER, fontweight="bold", va="top", color=INK)

for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(R, "paper", "figs", f"fig3_patch.{ext}"), dpi=350)
print("written paper/figs/fig3_patch.{png,pdf,svg}")
print(f"a: rank correlation {rho_deck:.2f}; c: r = {r_c:.3f}; g: halves/whole = {share:.3f}; "
      f"b: new content {new_content_gain:+.1f}")
print(f"h: Ironclad pre {iron_pre:.1f}% -> end {y_i[-1]:.1f}% at {x_i[-1]:.2f}e8 steps; "
      f"Fool pre {fool_pre:.1f}% -> end {y_f[-1]:.1f}% at {x_end:.2f}e8 steps "
      f"(greedy 2000-seed: {fool_g_pre:.1f}% -> {fool_g_post:.1f}%)")
print(f"h2 (12 buffed Fool cards): pre {pre_overall:.2f}% ({n_zero_pre}/12 at 0%), "
      f"post {mech114['summary']['buffed_post_pick_pct']:.2f}% overall, "
      f"Serrated Arrows {serrated['pre_pick_pct']:.1f}% -> {serrated['post_pick_pct']:.1f}%, "
      f"{n_lt1_post}/12 <1% (9/12 <3%, 11/12 <7%)")
