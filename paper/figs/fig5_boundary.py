"""Figure 5: the combination boundary, dissected.

  a, Forgotten Ritual through five conditions of our network (taken when offered, plays giving +3 energy) with the
     A10 win rate of each condition above the bars; the card render and the two rules beside it.
  b, Abandonment after the v0.113 nerf against rediscovery after the v0.111 rule is restored (broken step axis).
  c, In-combat probe before and after the enabling play (three of our networks, AlphaDefect), and which input triggers
     AlphaDefect's play (energy preview vs condition highlight, removed or added).
  d, Forced exposure: voluntary pick rate of Stoke, Feel No Pain and Dark Embrace (with / without Stoke in the deck)
     at 0, 190M, 550M and 1.3B steps as a heat map, beside the conditional matrices of our 550M policy (by exhaust
     sources) and of AlphaDefect (by deck state); one colour scale.

Lettering follows paper/figs/STYLE.md (nothing under 6.5 pt, no notes on the figure).
Run:
    python3 paper/figs/fig5_boundary.py -> paper/figs/fig5_boundary.{pdf,svg,png}
"""
from decimal import Decimal, ROUND_HALF_UP
import json
import math
import os
import subprocess
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARDS_DIR = os.path.join(R, "tmp", "game_assets", "cards")
ASSET_V2_DIR = os.path.join(R, "tmp", "game_assets", "fig14_v2")
if not os.path.exists(os.path.join(ASSET_V2_DIR, "badge_char_ironclad.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_V2_DIR, "build_assets.py")], check=True, cwd=R)

ABANDON_PATH = os.path.join(R, "paper", "figdata", "abandon1124_curve.json")
AD_PROBE_PATH = os.path.join(R, "paper", "figdata", "alphadefect_causal_probe.json")

# Unified palette (shared across fig1, fig2, fig4, fig5)
BLUE = "#2166ac"
BLUE_MID = "#2b6cb0"
BLUE_PILL = "#dceaf7"
BLUE_SOFT = "#eef4fb"

TERRA = "#c85228"
TERRA_PILL = "#fbe6de"
TERRA_SOFT = "#fcf3ef"
MECH_BG = "#fcf6f2"
MECH_BORDER = "#e6cfc3"

TEAL = "#1b8a6b"
TEAL_PILL = "#ddf2eb"
TEAL_SOFT = "#edf8f4"

AMBER = "#c98a1b"
AMBER_PILL = "#faecd2"
AMBER_SOFT = "#fdf6ea"

VIOLET = "#5e4cc4"
VIOLET_PILL = "#e9e5f9"
VIOLET_SOFT = "#f6f4fc"

SLATE = "#64748b"
INK = "#14181f"
INK2 = "#4a5568"
MUTED = "#718096"
ZEBRA = "#f5f7fa"
CARD_BG = "#f7f9fc"
BORDER = "#dce3ec"
OUTER_BORDER = "#e2e8f0"
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 6.2,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.2, pad=1.8)


def r1_from_prob(prob_val: float) -> float:
    """Round probability [0,1] via report 2-decimal percentage then half-up to 1 decimal."""
    p2 = Decimal(str(prob_val * 100.0)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(p2.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def wilson(k, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, 100.0 * (c - h)), min(100.0, 100.0 * (c + h))


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def load_pil_rgba(folder, name):
    fn = name if name.endswith(".png") else f"{name}.png"
    return Image.open(os.path.join(folder, fn)).convert("RGBA")


def composite_on_bg(im_rgba: Image.Image, target_px_w: int, bg_rgb=(255, 255, 255)):
    """Composite RGBA onto solid background first, then Lanczos-resize so zero alpha-ringing occurs."""
    bg = Image.new("RGBA", im_rgba.size, (*bg_rgb, 255))
    comp = Image.alpha_composite(bg, im_rgba).convert("RGB")
    target_px_h = max(1, round(target_px_w * im_rgba.height / im_rgba.width))
    return np.asarray(comp.resize((target_px_w, target_px_h), Image.Resampling.LANCZOS))


def load_rgba_clean(folder, fn, target_h_px):
    im = load_pil_rgba(folder, fn)
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


def add_sprite(ax, folder, fn, xy, target_h_pt, xycoords="data", align=(0.5, 0.5), zorder=6, clip_on=False):
    arr, h_px = load_rgba_clean(folder, fn, target_h_pt * 5.0)
    oi = OffsetImage(arr, zoom=target_h_pt / h_px, interpolation="bilinear", resample=True)
    ab = AnnotationBbox(oi, xy, xycoords=xycoords, box_alignment=align, frameon=False, pad=0,
                        zorder=zorder, annotation_clip=clip_on)
    ab.set_clip_on(clip_on)
    ax.add_artist(ab)


def add_img_axes(fig, rect, im_pil, bg_hex="#ffffff", zorder=4):
    ax_i = fig.add_axes(rect, frameon=False, zorder=zorder)
    px_w = max(64, round(rect[2] * FIG_W * 420))
    ax_i.imshow(composite_on_bg(im_pil, px_w, bg_rgb=hex_to_rgb(bg_hex)), interpolation="bilinear")
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    return ax_i


# =========================================================================
# Load 100% Ground-Truth Datasets
# =========================================================================
ab_data = json.load(open(ABANDON_PATH))
ad_data = json.load(open(AD_PROBE_PATH))

# Dynamic AlphaDefect statistics from alphadefect_causal_probe.json
ad_rv1_std = ad_data["reproduction_rv1"]["rv1_std_summary"]
ad_rv1_rev = ad_data["probe1_card_pick_and_rule_revert"]["rv1_1000_revert_full"]
ad_rv1_pair = ad_data["probe1_card_pick_and_rule_revert"]["rv1_1000_paired_revert_vs_std"]

ad_fr_rev_target = ad_rv1_rev["target_cards"]["CARD.FORGOTTEN_RITUAL"]
ad_fr_rev_usage = ad_rv1_rev["forgotten_ritual_usage"]

ad_kd = ad_data["probe2_in_combat_know_do"]["alphadefect_reward_shop_forced"]
ad_kd_dist = ad_kd["distributions"]
ours_kd_ref = ad_data["probe2_in_combat_know_do"]["our_minimal_agent_comparison_ref"]



def register_helvetica_bold():
    """matplotlib reads only the first face of Helvetica.ttc, so bold lettering silently falls back to regular.
    The bold face is extracted once into tmp/ and registered; without the system font the figure stays regular."""
    dst = os.path.join(R, "tmp", "game_assets", "fig45_restyle", "Helvetica-Bold.ttf")
    try:
        if not os.path.exists(dst):
            from fontTools.ttLib import TTCollection
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            for face in TTCollection("/System/Library/Fonts/Helvetica.ttc").fonts:
                if face["name"].getDebugName(2) == "Bold":
                    face.save(dst)
                    break
        from matplotlib import font_manager
        font_manager.fontManager.addfont(dst)
    except Exception as exc:  # other platforms: keep the regular face
        print("Helvetica Bold not registered:", exc)


register_helvetica_bold()
plt.rcParams.update({"font.size": 6.8})

FS_AX = 7.6      # axis labels
FS_TICK = 6.8    # tick labels
FS_LAB = 7.4     # in-plot labels
FS_CNT = 6.6     # counts
GRID = "#edf2f7"


def pct_half_up(v):
    return int(math.floor(v + 0.5))


# =========================================================================
# Canvas, outer frame, and one overlay axes in points for card art and arrows
# =========================================================================
FIG_W, FIG_H = 8.7, 6.0
PT_W, PT_H = FIG_W * 72.0, FIG_H * 72.0
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

fig.add_artist(FancyBboxPatch(
    (0.006, 0.008), 0.988, 0.984, transform=fig.transFigure,
    boxstyle="round,pad=0.0,rounding_size=0.014",
    facecolor="none", edgecolor=OUTER_BORDER, linewidth=0.85, zorder=0
))

AXP = fig.add_axes([0, 0, 1, 1], zorder=8)
AXP.set_xlim(0, PT_W)
AXP.set_ylim(0, PT_H)
AXP.axis("off")
AXP.patch.set_visible(False)


def fx(x):
    return x * PT_W


def fy(y):
    return y * PT_H


def art(folder, fn, x_pt, y_pt, h_pt, align=(0.5, 0.5), zorder=6):
    add_sprite(AXP, folder, fn, (x_pt, y_pt), h_pt, xycoords="data", align=align, zorder=zorder)


def arrow_pt(x0, y0, x1, y1, color=INK2, lw=1.1, head=0.30, zorder=7):
    AXP.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle=f"-|>,head_width={head},head_length={head * 1.6}",
                                 lw=lw, color=color, shrinkA=0, shrinkB=0), zorder=zorder)


# =========================================================================
# Panel a: Forgotten Ritual through five conditions
# =========================================================================
BOX_A = (0.014, 0.578, 0.146, 0.400)               # card box in figure fractions
bx0, by0, bw_, bh_ = fx(BOX_A[0]), fy(BOX_A[1]), fx(BOX_A[2]), fy(BOX_A[3])
fig.add_artist(FancyBboxPatch(
    (BOX_A[0], BOX_A[1]), BOX_A[2], BOX_A[3], transform=fig.transFigure,
    boxstyle="round,pad=0.0,rounding_size=0.012", facecolor=CARD_BG, edgecolor=BORDER, linewidth=0.75, zorder=1
))
FR_H = 112.0
art(CARDS_DIR, "card_forgotten_ritual.png", bx0 + bw_ / 2.0, by0 + bh_ - 4.0, FR_H, align=(0.5, 1.0))

# the two rules: version and energy on the first line, the condition on the second
CAP_H, CAP_GAP = 24.0, 4.0
for k_rule, (ver, cond, col, bg) in enumerate([
    ("v0.111", "always", BLUE, BLUE_SOFT),
    ("v0.113", "after Exhaust", TERRA, TERRA_SOFT),
]):
    cy0 = by0 + 4.0 + (1 - k_rule) * (CAP_H + CAP_GAP)
    AXP.add_patch(FancyBboxPatch((bx0 + 4.0, cy0), bw_ - 8.0, CAP_H,
                                 boxstyle="round,pad=0.0,rounding_size=4.0",
                                 facecolor=bg, edgecolor=col, linewidth=0.8, zorder=3))
    AXP.text(bx0 + 8.5, cy0 + CAP_H * 0.70, ver, fontsize=7.4, fontweight="bold", color=col,
             ha="left", va="center", zorder=5)
    AXP.text(bx0 + 8.5, cy0 + CAP_H * 0.28, cond, fontsize=7.2, color=INK, ha="left", va="center", zorder=5)
    for k_orb in range(3):
        art(CARDS_DIR, "icon_energy_ironclad.png", bx0 + bw_ - 11.5 - k_orb * 9.5, cy0 + CAP_H * 0.70, 9.0,
            zorder=6)

AX_A = [0.230, 0.612, 0.294, 0.283]
ax_a = fig.add_axes(AX_A)
ax_a_top = fig.add_axes([AX_A[0], 0.906, AX_A[2], 0.048], frameon=False)
clean_ax(ax_a)

c4 = ab_data["four_conditions"]
_re_last = sorted(ab_data["rediscovery_curve"], key=lambda d: d["steps_m"])[-1]
conds_5 = [
    {"x": 0.0, "xtick": "Old net,\nv0.111", "offered": c4[0]["offered"], "picked": c4[0]["picked"],
     "plays": c4[0]["plays"], "energy_gained": c4[0]["energy_gained"], "wr_txt": "62.1%",
     "strip_bg": "#eef2f7", "strip_fg": INK},
    {"x": 1.0, "xtick": "Old net,\nv0.113", "offered": c4[1]["offered"], "picked": c4[1]["picked"],
     "plays": c4[1]["plays"], "energy_gained": c4[1]["energy_gained"], "wr_txt": "52.4% (-9.7)",
     "strip_bg": TERRA_PILL, "strip_fg": TERRA},
    {"x": 2.0, "xtick": "Adapted,\nv0.113", "offered": c4[2]["offered"], "picked": c4[2]["picked"],
     "plays": c4[2]["plays"], "energy_gained": c4[2]["energy_gained"], "wr_txt": "66.2%",
     "strip_bg": BLUE_SOFT, "strip_fg": BLUE},
    {"x": 3.0, "xtick": "Ours, rule\nrestored", "offered": c4[3]["offered"], "picked": c4[3]["picked"],
     "plays": c4[3]["plays"], "energy_gained": c4[3]["energy_gained"], "wr_txt": "65.7% (-0.5)",
     "strip_bg": BLUE_SOFT, "strip_fg": BLUE},
    {"x": 4.0, "xtick": f"Restored,\n+{_re_last['steps_m']:.0f}M steps", "offered": _re_last["offered"],
     "picked": _re_last["picked"], "plays": _re_last["plays"], "energy_gained": _re_last["energy_gained"],
     "wr_txt": f"{_re_last['win_pct']:.1f}%", "strip_bg": BLUE_SOFT, "strip_fg": BLUE},
]

A_XLIM = (-0.54, 4.54)
# A10 win rate of each condition, one cell per condition, with the Ironclad badge at the left
ax_a_top.set_xlim(*A_XLIM)
ax_a_top.set_ylim(0, 1)
ax_a_top.axis("off")
for c in conds_5:
    ax_a_top.add_patch(FancyBboxPatch((c["x"] - 0.43, 0.08), 0.86, 0.84,
                                      boxstyle="round,pad=0.0,rounding_size=0.14",
                                      facecolor=c["strip_bg"], edgecolor="none", zorder=2))
    ax_a_top.text(c["x"], 0.50, c["wr_txt"].split(" (")[0], fontsize=FS_LAB + 0.4, fontweight="bold",
                  color=c["strip_fg"], ha="center", va="center", zorder=4)
a_unit = fx(AX_A[2]) / (A_XLIM[1] - A_XLIM[0])
strip_left = fx(AX_A[0]) + (-0.43 - A_XLIM[0]) * a_unit
art(ASSET_V2_DIR, "badge_char_ironclad.png", strip_left - 10.0, fy(0.906 + 0.024), 17.0)
AXP.text(strip_left - 20.5, fy(0.906 + 0.024), "Win %", fontsize=FS_TICK, color=INK, ha="right", va="center")

for gy in (50, 100):
    ax_a.axhline(gy, color=GRID, lw=0.55, zorder=0)

BW_A = 0.36
for c in conds_5:
    xi = c["x"]
    xb, xe = xi - BW_A / 2.0, xi + BW_A / 2.0
    pk_pct = 100.0 * c["picked"] / c["offered"]
    lo_p, hi_p = wilson(c["picked"], c["offered"])
    en_pct = 100.0 * c["energy_gained"] / max(1, c["plays"])
    lo_e, hi_e = wilson(c["energy_gained"], c["plays"])
    is_nondraft = (c["picked"] == 0)

    ax_a.bar(xb, pk_pct, width=BW_A, color=BLUE, edgecolor="white", linewidth=0.45, zorder=3)
    ax_a.plot([xb, xb], [lo_p, hi_p], color=INK, lw=0.7, zorder=4)
    ax_a.bar(xe, en_pct, width=BW_A, color=AMBER_SOFT if is_nondraft else AMBER,
             edgecolor=AMBER if is_nondraft else "white", hatch="////" if is_nondraft else None,
             linewidth=0.6 if is_nondraft else 0.45, zorder=3)
    ax_a.plot([xe, xe], [lo_e, hi_e], color=INK2, lw=0.7, zorder=4)

    # taken when offered: share on top, picks/offers inside the unobstructed bar below lo_p or above a short bar
    pcol = TERRA if c["picked"] == 0 else BLUE
    if pk_pct >= 40:
        ax_a.text(xb, hi_p + 1.5, f"{pk_pct:.0f}", ha="center", va="bottom", fontsize=FS_LAB,
                  fontweight="bold", color=pcol, zorder=5)
        ax_a.text(xb, 0.5 * lo_p, f"{c['picked']}/{c['offered']}", ha="center", va="center", fontsize=FS_CNT,
                  fontweight="bold", color="white", rotation=90, zorder=5)
    else:
        ax_a.text(xb, hi_p + 1.5, f"{pk_pct:.0f}", ha="center", va="bottom", fontsize=FS_LAB,
                  fontweight="bold", color=pcol, zorder=5)
        ax_a.text(xb, hi_p + 9.5, f"{c['picked']}/{c['offered']}", ha="center", va="bottom", fontsize=FS_CNT,
                  fontweight="bold", color=pcol, rotation=90, zorder=5)
    # plays giving +3 energy
    ax_a.text(xe, min(101.5, hi_e + 1.5), f"{en_pct:.0f}", ha="center", va="bottom", fontsize=FS_LAB,
              fontweight="bold", color=AMBER, zorder=5)
    if is_nondraft:
        ax_a.text(xe, 48.0, f"{c['energy_gained']}/{c['plays']}", ha="center", va="center", fontsize=FS_CNT,
                  fontweight="bold", color=INK, rotation=90, path_effects=HALO, zorder=5)

# two direct labels in place of the legend
ax_a.text(0.66, 99.0, "Taken when\noffered", ha="left", va="center", fontsize=FS_LAB, fontweight="bold",
          color=BLUE, linespacing=1.0, zorder=6)
ax_a.text(2.10, 57.0, "Plays giving\n+3 energy", ha="center", va="center", fontsize=FS_LAB, fontweight="bold",
          color=AMBER, linespacing=1.0, zorder=6)

ax_a.set_xlim(*A_XLIM)
ax_a.set_ylim(0, 112)
ax_a.set_yticks([0, 50, 100])
ax_a.set_xticks([c["x"] for c in conds_5])
ax_a.set_xticklabels([c["xtick"] for c in conds_5], fontsize=FS_TICK, linespacing=1.0)
ax_a.tick_params(axis="x", length=0, pad=3.0)
ax_a.tick_params(axis="y", labelsize=FS_TICK)
ax_a.set_ylabel("Share (%)", fontsize=FS_AX, labelpad=2.0)

# =========================================================================
# Panel b: abandonment after the nerf against rediscovery after the rule is restored
# =========================================================================
AX_B1 = [0.597, 0.612, 0.300, 0.342]
AX_B2 = [0.912, 0.612, 0.068, 0.342]
ax_b1 = fig.add_axes(AX_B1)
ax_b2 = fig.add_axes(AX_B2, sharey=ax_b1)
clean_ax(ax_b1)
clean_ax(ax_b2)
ax_b2.spines["left"].set_visible(False)
ax_b2.tick_params(axis="y", left=False, labelleft=False)

ab_pts = sorted(ab_data["abandonment_curve"], key=lambda d: d["steps_m"])
ab_early = [d for d in ab_pts if d["steps_m"] <= 200.0]
ab_late = [d for d in ab_pts if d["steps_m"] > 200.0]
re_pts = sorted(ab_data["rediscovery_curve"], key=lambda d: d["steps_m"])
re_tot_p = sum(d["picked"] for d in re_pts)
re_tot_o = sum(d["offered"] for d in re_pts)

B1_XLIM, B1_YLIM = (-12, 622), (-6.0, 82)

for gy in (0, 20, 40, 60, 80):
    ax_b1.axhline(gy, color=GRID, lw=0.55, zorder=0)
    ax_b2.axhline(gy, color=GRID, lw=0.55, zorder=0)
pre_nerf_pct = 100.0 * c4[0]["picked"] / c4[0]["offered"]
for axx in (ax_b1, ax_b2):
    axx.axhline(pre_nerf_pct, color=SLATE, lw=0.7, ls=(0, (2.5, 2.0)), zorder=1)

x_ab = [d["steps_m"] for d in ab_early]
y_ab = [100.0 * d["picked"] / d["offered"] for d in ab_early]
lo_ab = [wilson(d["picked"], d["offered"])[0] for d in ab_early]
hi_ab = [wilson(d["picked"], d["offered"])[1] for d in ab_early]
ax_b1.fill_between(x_ab, lo_ab, hi_ab, color=TERRA, alpha=0.16, lw=0, zorder=2)
ax_b1.plot(x_ab, y_ab, "-o", color=TERRA, lw=1.7, ms=4.2, mec="white", mew=0.55, zorder=4)

x_re = [d["steps_m"] for d in re_pts]
y_re = [100.0 * d["picked"] / d["offered"] for d in re_pts]
lo_re = [wilson(d["picked"], d["offered"])[0] for d in re_pts]
hi_re = [wilson(d["picked"], d["offered"])[1] for d in re_pts]
ax_b1.fill_between(x_re, lo_re, hi_re, color=BLUE, alpha=0.20, lw=0, zorder=3)
ax_b1.plot(x_re, y_re, "-s", color=BLUE, lw=1.7, ms=3.9, mec="white", mew=0.55, zorder=5)

# dashed terracotta continuation from last early abandonment point (165.7M, 3.5%) across to ax_b2 (0.66B, 0%)
y_brk = y_ab[-1] * (ab_late[0]["steps_m"] - B1_XLIM[1]) / (ab_late[0]["steps_m"] - x_ab[-1])
ax_b1.plot([x_ab[-1], B1_XLIM[1]], [y_ab[-1], y_brk], "--", color=TERRA, lw=1.0, alpha=0.75, zorder=2)

ax_b1.text(66.0, 30.5, "abandoned:\nhalf-life ~35M", linespacing=1.02, fontsize=FS_LAB, fontweight="bold",
           color=TERRA, ha="left", va="center", path_effects=HALO, zorder=6)
ax_b1.text(312.0, 8.0, "restored rule: >12× slower", fontsize=FS_LAB, fontweight="bold",
           color=BLUE, ha="left", va="center", path_effects=HALO, zorder=6)

# the two rules as card renders in the open upper-middle sky, framed in the colour of their curve
def b1_pt(x, y):
    return (fx(AX_B1[0]) + (x - B1_XLIM[0]) / (B1_XLIM[1] - B1_XLIM[0]) * fx(AX_B1[2]),
            fy(AX_B1[1]) + (y - B1_YLIM[0]) / (B1_YLIM[1] - B1_YLIM[0]) * fy(AX_B1[3]))


RULE_CARD_H = 38.0
rc_w = RULE_CARD_H * 675.0 / 910.0
for fn_card, col, xc, ver in (("card_forgotten_ritual.png", TERRA, 214.0, "v0.113"),
                              ("card_forgotten_ritual_v111.png", BLUE, 338.0, "v0.111")):
    cx_pt, cy_pt = b1_pt(xc, 46.5)
    AXP.add_patch(FancyBboxPatch((cx_pt - rc_w / 2.0 - 2.0, cy_pt - 2.0), rc_w + 4.0, RULE_CARD_H + 4.0,
                                 boxstyle="round,pad=0.0,rounding_size=3.5", facecolor="white", edgecolor=col,
                                 linewidth=1.5, zorder=5))
    art(CARDS_DIR, fn_card, cx_pt, cy_pt, RULE_CARD_H, align=(0.5, 0.0), zorder=6)
    AXP.text(cx_pt, cy_pt - 8.0, ver, fontsize=FS_LAB, fontweight="bold", color=col, ha="center", va="center",
             zorder=6)

lx = np.arange(len(ab_late))
ly = [100.0 * d["picked"] / d["offered"] for d in ab_late]
llo = [wilson(d["picked"], d["offered"])[0] for d in ab_late]
lhi = [wilson(d["picked"], d["offered"])[1] for d in ab_late]
ax_b2.plot([-0.55] + list(lx), [y_brk] + ly, "--", color=TERRA, lw=1.0, alpha=0.75, zorder=2)
for j, d in enumerate(ab_late):
    ax_b2.plot([j, j], [llo[j], lhi[j]], color=TERRA, lw=1.0, zorder=3)
    ax_b2.scatter([j], [ly[j]], s=24, color=TERRA, edgecolor="white", linewidth=0.6, zorder=4)
    ax_b2.text(j, lhi[j] + 2.5, f"{d['picked']}/{d['offered']}", fontsize=FS_CNT, fontweight="bold",
               color=TERRA, ha="center", va="bottom", rotation=90)
ax_b2.set_xticks(lx)
ax_b2.set_xticklabels(["0.66B", "2.1B"][:len(ab_late)], fontsize=FS_TICK)
ax_b2.set_xlim(-0.55, len(ab_late) - 0.45)

d_brk = 0.018
ax_b1.plot((1 - d_brk, 1 + d_brk), (-d_brk, +d_brk), transform=ax_b1.transAxes, color=INK2, clip_on=False, lw=0.7)
ax_b2.plot((-d_brk * 4.4, +d_brk * 4.4), (-d_brk, +d_brk), transform=ax_b2.transAxes, color=INK2, clip_on=False,
           lw=0.7)

ax_b1.set_xlim(*B1_XLIM)
ax_b1.set_ylim(*B1_YLIM)
ax_b1.set_yticks([0, 20, 40, 60, 80])
ax_b1.set_xticks([0, 100, 200, 300, 400, 500, 600])
ax_b1.tick_params(labelsize=FS_TICK)
ax_b1.set_ylabel("Forgotten Ritual taken (%)", fontsize=FS_AX, labelpad=2.0)
fig.text(AX_B1[0] + (AX_B2[0] + AX_B2[2] - AX_B1[0]) / 2.0, 0.580, "Steps after the rule change (millions)",
         ha="center", va="top", fontsize=FS_AX, color=INK)

# =========================================================================
# Panel c left: probability of playing Forgotten Ritual before and after the enabling play
# =========================================================================
AX_C = [0.058, 0.088, 0.150, 0.388]
ax_c = fig.add_axes(AX_C)
clean_ax(ax_c)
for gy in (0, 20, 40, 60, 80):
    ax_c.axhline(gy, color=GRID, lw=0.55, zorder=0)

m_main = ours_kd_ref["iron_avg39800"]
m_ctl = ours_kd_ref["ctl_c"]
m_exh = ours_kd_ref["exh3_c"]


def probe_row(col, d_s, d_sp, dx, lw_c):
    return (col, 100 * d_s["mean"], 100 * d_s["median"], 100 * d_s["q25"], 100 * d_s["q75"],
            100 * d_sp["mean"], 100 * d_sp["median"], 100 * d_sp["q25"], 100 * d_sp["q75"], dx, lw_c)


OURS_SHADES = (BLUE, "#4f8fcf", "#8db7e2")
probe_rows = [
    probe_row(OURS_SHADES[0], m_main["p_ritual_s"], m_main["p_ritual_s_prime"], -0.045, 1.5),
    probe_row(OURS_SHADES[1], m_ctl["p_ritual_s"], m_ctl["p_ritual_s_prime"], -0.015, 1.5),
    probe_row(OURS_SHADES[2], m_exh["p_ritual_s"], m_exh["p_ritual_s_prime"], +0.015, 1.5),
    probe_row(VIOLET, ad_kd_dist["p_ritual_in_s"], ad_kd_dist["p_ritual_in_s_prime"], +0.048, 2.0),
]
for col, ms, mds, q1s, q3s, msp, mdsp, q1sp, q3sp, dx, lw_c in probe_rows:
    ax_c.plot([dx, dx], [q1s, q3s], color=col, lw=1.0, alpha=0.35, zorder=2)
    ax_c.plot([1 + dx, 1 + dx], [q1sp, min(86.0, q3sp)], color=col, lw=1.0, alpha=0.35, zorder=2)
    ax_c.plot([dx, 1 + dx], [ms, msp], "-o", color=col, lw=lw_c, ms=4.4, mec="white", mew=0.6, zorder=4)
    ax_c.scatter([dx, 1 + dx], [mds, mdsp], marker="D", s=16, facecolor="white", edgecolor=col, linewidth=0.9,
                 zorder=5)

ad_end = 100 * ad_kd_dist["p_ritual_in_s_prime"]["mean"]
ours_ends = [100 * m["p_ritual_s_prime"]["mean"] for m in (m_main, m_ctl, m_exh)]
ax_c.text(1.11, ad_end, "AlphaDefect", fontsize=FS_LAB, fontweight="bold", color=VIOLET, ha="left", va="center",
          clip_on=False)
ax_c.plot([1.10, 1.14, 1.14, 1.10], [min(ours_ends), min(ours_ends), max(ours_ends), max(ours_ends)],
          color=BLUE, lw=0.8, clip_on=False, zorder=3)
ax_c.text(1.18, 0.5 * (min(ours_ends) + max(ours_ends)), "Ours", fontsize=FS_LAB, fontweight="bold", color=BLUE,
          ha="left", va="center", clip_on=False)

ours_en_pct = m_main["greedy_any_exhaust_eligible_pct"]
ad_en_pct = ad_kd["greedy_choice_rates"]["any_exhaust_at_eligible_pct"]
en_txt = (f"{pct_half_up(ours_en_pct)}%" if pct_half_up(ours_en_pct) == pct_half_up(ad_en_pct)
          else f"{pct_half_up(ours_en_pct)}% / {pct_half_up(ad_en_pct)}%")
ax_c.text(-0.10, 87.0, f"{en_txt} choose the\nenabling play first", fontsize=FS_LAB, fontweight="bold",
          color=INK, ha="left", va="top", linespacing=1.05, zorder=6)
ax_c.scatter([0.98], [6.0], marker="D", s=16, facecolor="white", edgecolor=INK2, linewidth=0.9, zorder=5)
ax_c.text(1.05, 6.0, "median", fontsize=FS_TICK, color=INK, ha="left", va="center")

ax_c.set_xticks([0, 1])
ax_c.set_xticklabels(["Before", "After"], fontsize=FS_TICK)
ax_c.set_xlabel("Enabling play", fontsize=FS_AX, labelpad=2.0)
ax_c.set_xlim(-0.22, 1.42)
ax_c.set_ylim(-3.5, 87.5)
ax_c.tick_params(labelsize=FS_TICK)
ax_c.spines["bottom"].set_bounds(-0.1, 1.1)
ax_c.set_ylabel("Plays Forgotten Ritual (%)", fontsize=FS_AX, labelpad=2.0)

# =========================================================================
# Panel c right: which input triggers AlphaDefect's play (Burning Pact -> Forgotten Ritual)
# =========================================================================
BOX_CB = (0.262, 0.020, 0.230, 0.502)
ax_cb_bg = fig.add_axes([0, 0, 1, 1], zorder=1)
ax_cb_bg.set_xlim(0, PT_W)
ax_cb_bg.set_ylim(0, PT_H)
ax_cb_bg.axis("off")
ax_cb_bg.patch.set_visible(False)
ax_cb_bg.add_patch(FancyBboxPatch(
    (fx(BOX_CB[0]), fy(BOX_CB[1])), fx(BOX_CB[2]), fy(BOX_CB[3]),
    boxstyle="round,pad=0.0,rounding_size=5.2", facecolor=CARD_BG, edgecolor=BORDER, linewidth=0.75
))

AX_CB = [0.397, 0.088, 0.064, 0.244]
ax_cb = fig.add_axes(AX_CB, zorder=4)
ax_cb.patch.set_visible(False)
clean_ax(ax_cb)

p_sp_intact = r1_from_prob(ad_kd_dist["p_ritual_in_s_prime"]["mean"])
p_sp_no_gain = r1_from_prob(ad_kd_dist["p_ritual_s_prime_ablate_no_gain_preview"]["mean"])
p_sp_no_glow = r1_from_prob(ad_kd_dist["p_ritual_s_prime_ablate_no_glow"]["mean"])
p_s_intact = r1_from_prob(ad_kd_dist["p_ritual_in_s"]["mean"])
p_s_sp_gain = r1_from_prob(ad_kd_dist["p_ritual_s_spoof_gain_preview"]["mean"])
p_s_sp_glow = r1_from_prob(ad_kd_dist["p_ritual_s_spoof_glow"]["mean"])

mech_rows = [
    ("after the enabling play", p_sp_intact, VIOLET),
    ("energy preview removed", p_sp_no_gain, AMBER),
    ("condition highlight removed", p_sp_no_glow, TERRA),
    ("before the enabling play", p_s_intact, VIOLET),
    ("energy preview added", p_s_sp_gain, AMBER),
    ("condition highlight added", p_s_sp_glow, TERRA),
]
y_pos_m = np.array([0.0, 1.0, 2.0, 3.35, 4.35, 5.35])
ax_cb.axhline(2.675, color=BORDER, lw=0.85, zorder=1)
for ym, (mlab, mval, mcol) in zip(y_pos_m, mech_rows):
    ax_cb.barh(ym, mval, height=0.68, color=mcol, edgecolor="none", zorder=3)
    ax_cb.text(-2.6, ym, mlab, fontsize=FS_TICK, color=INK, ha="right", va="center", clip_on=False)
    ax_cb.text(mval + 2.0, ym, f"{mval:.1f}", fontsize=FS_LAB, fontweight="bold", color=mcol, ha="left",
               va="center", clip_on=False)
ax_cb.set_ylim(5.85, -0.55)
ax_cb.set_xlim(0, 64)
ax_cb.set_xticks([0, 30, 60])
ax_cb.tick_params(axis="x", labelsize=FS_TICK)
ax_cb.set_yticks([])
ax_cb.spines["left"].set_visible(False)
ax_cb.set_xlabel("Plays Forgotten Ritual (%)", fontsize=FS_AX, labelpad=2.0, x=0.14)

# the two cards with the arrow, centered in the intervention container above the bars
CARD_H = 64.0
card_w = CARD_H * 675.0 / 910.0
cards_cx = fx(BOX_CB[0] + BOX_CB[2] / 2.0)
cards_y0 = fy(AX_CB[1] + AX_CB[3]) + 11.0
c1x = cards_cx - card_w / 2.0 - 13.0
c2x = cards_cx + card_w / 2.0 + 13.0
art(CARDS_DIR, "card_burning_pact.png", c1x, cards_y0, CARD_H, align=(0.5, 0.0))
AXP.add_patch(FancyBboxPatch((c2x - card_w / 2.0 - 2.4, cards_y0 - 2.4), card_w + 4.8, CARD_H + 4.8,
                             boxstyle="round,pad=0.0,rounding_size=4.0", facecolor="#fef3c7", edgecolor=AMBER,
                             linewidth=1.6, zorder=5))
art(CARDS_DIR, "card_forgotten_ritual.png", c2x, cards_y0, CARD_H, align=(0.5, 0.0), zorder=6)
arrow_pt(c1x + card_w / 2.0 + 2.5, cards_y0 + CARD_H / 2.0, c2x - card_w / 2.0 - 4.5, cards_y0 + CARD_H / 2.0,
         color=TERRA, lw=1.4, head=0.34)

# =========================================================================
# Panel d: forced exposure heat map, card column, and the two conditional matrices
# =========================================================================
steps_d = [0, 190, 550, 1300]
y_stoke = [3.96, 69.16, 89.89, 58.00]
y_fnp_with = [0.59, 14.29, 80.00, 83.33]
y_fnp_no = [0.59, 10.87, 45.59, 22.97]
y_de_with = [0.00, 2.11, 28.57, 20.00]
y_de_no = [0.00, 2.35, 7.95, 3.70]

CMAP_D = plt.get_cmap("YlGnBu")
NORM_D = matplotlib.colors.Normalize(vmin=0, vmax=100)
ROW_EDGES5 = np.arange(6)                     # Stoke; FNP + / -; DE + / -
ROW_EDGES3 = np.array([0, 1, 3, 5])           # the same bands, one row per card

D_Y0, D_Y1 = 0.088, 0.464                     # rows, figure fractions
D_COL0 = 0.503                                # left edge of the card column
CARD_COL_W, SUB_COL_W = 47.0, 31.4            # points
MAIN_X0 = fx(D_COL0) + CARD_COL_W + SUB_COL_W
MAIN_W, OURS_W, AD_W = 4 * 22.8, 3 * 22.8, 2 * 25.8
OURS_X0 = MAIN_X0 + MAIN_W + 6.0
AD_X0 = OURS_X0 + OURS_W + 6.0
rows_h_pt = fy(D_Y1 - D_Y0)
row_pt = rows_h_pt / 5.0


def d_axes(x0_pt, w_pt):
    return fig.add_axes([x0_pt / PT_W, D_Y0, w_pt / PT_W, D_Y1 - D_Y0])


def cell_text(ax, x, y, val, cnt=None, big=FS_LAB + 0.2):
    tc = "white" if val >= 55 else INK
    if cnt is None:
        ax.text(x, y, f"{pct_half_up(val)}", ha="center", va="center", fontsize=big, fontweight="bold", color=tc,
                zorder=4)
    else:
        ax.text(x, y - 0.17, f"{pct_half_up(val)}", ha="center", va="center", fontsize=big, fontweight="bold",
                color=tc, zorder=4)
        ax.text(x, y + 0.24, f"{cnt[0]}/{cnt[1]}", ha="center", va="center", fontsize=FS_CNT, color=tc, zorder=4)


def frame_heat(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=3.0, labelsize=FS_TICK)
    for s_ in ax.spines.values():
        s_.set_visible(False)


# main heat map: voluntary pick rate in free runs across forced-exposure steps
ax_dm = d_axes(MAIN_X0, MAIN_W)
mat_main = np.array([y_stoke, y_fnp_with, y_fnp_no, y_de_with, y_de_no])
ax_dm.pcolormesh(np.arange(5), ROW_EDGES5, mat_main, cmap=CMAP_D, norm=NORM_D, edgecolors="white", linewidth=0.9)
for r_i in range(5):
    for c_j in range(4):
        cell_text(ax_dm, c_j + 0.5, r_i + 0.5, mat_main[r_i, c_j])
for yb in (1, 3):
    ax_dm.axhline(yb, color="white", lw=2.6, zorder=3)
ax_dm.add_patch(Rectangle((2.03, 0.04), 0.94, 4.92, facecolor="none", edgecolor=BLUE, linewidth=1.5, zorder=5,
                          clip_on=False))
ax_dm.set_ylim(5, 0)
ax_dm.set_xlim(0, 4)
ax_dm.set_xticks(np.arange(4) + 0.5)
ax_dm.set_xticklabels(["0", "190M", "550M", "1.3B"])
ax_dm.set_xlabel("Forced-exposure steps", fontsize=FS_AX, labelpad=2.0)
frame_heat(ax_dm)

# with / without Stoke beside the Feel No Pain and Dark Embrace rows
for r_i, lab in ((1, "+ Stoke"), (2, "− Stoke"), (3, "+ Stoke"), (4, "− Stoke")):
    ax_dm.text(-0.12, r_i + 0.5, lab, fontsize=7.0, color=INK, ha="right", va="center", clip_on=False)

# card column with crisp bold 6.8 pt card names and dependency arrows
y_top_pt = fy(D_Y1)
badge_h = 33.0
card_cx = fx(D_COL0) + CARD_COL_W / 2.0
card_specs = [
    ("badge_stoke.png", "Stoke", y_top_pt - 10.5),
    ("badge_feel_no_pain.png", "Feel No Pain", y_top_pt - 2.02 * row_pt),
    ("badge_dark_embrace.png", "Dark Embrace", y_top_pt - 3.92 * row_pt),
]
name_bot_ys = []
badge_top_ys = []
for bname, cname, by_c in card_specs:
    art(CARDS_DIR, bname, card_cx, by_c, badge_h)
    ny_top = by_c - badge_h / 2.0 - 1.2
    AXP.text(card_cx, ny_top, cname, fontsize=6.8, fontweight="bold", color=INK, ha="center", va="top", zorder=6)
    badge_top_ys.append(by_c + badge_h / 2.0)
    name_bot_ys.append(ny_top - 7.2)

for k_arr in range(2):
    arrow_pt(card_cx, name_bot_ys[k_arr] - 1.6, card_cx, badge_top_ys[k_arr + 1] + 1.8,
             color=INK2, lw=1.15, head=0.28)

# ours at 550M by number of exhaust sources (same bands)
hm1_counts = [
    [(63, 69), (16, 19), (1, 1)],      # Stoke
    [(41, 113), (33, 40), (8, 8)],     # Feel No Pain
    [(5, 74), (3, 22), (3, 6)],        # Dark Embrace
]
hm1_pct = np.array([[100.0 * k / n for k, n in row] for row in hm1_counts])
ax_d1 = d_axes(OURS_X0, OURS_W)
ax_d1.pcolormesh(np.arange(4), ROW_EDGES3, hm1_pct, cmap=CMAP_D, norm=NORM_D, edgecolors="white", linewidth=0.9)
for r_i in range(3):
    yc = 0.5 * (ROW_EDGES3[r_i] + ROW_EDGES3[r_i + 1])
    for c_j in range(3):
        cell_text(ax_d1, c_j + 0.5, yc, hm1_pct[r_i, c_j], hm1_counts[r_i][c_j])
ax_d1.set_ylim(5, 0)
ax_d1.set_xlim(0, 3)
ax_d1.set_xticks(np.arange(3) + 0.5)
ax_d1.set_xticklabels(["0", "1", "≥2"])
ax_d1.set_xlabel("Exhaust sources", fontsize=FS_AX, labelpad=2.0)
frame_heat(ax_d1)

# AlphaDefect by deck state
ad_tc = ad_rv1_std["target_cards"]
hm2_counts = []
for cid in ("CARD.STOKE", "CARD.FEEL_NO_PAIN", "CARD.DARK_EMBRACE"):
    cond_d = ad_tc[cid]["conditional_by_deck_state"]
    hm2_counts.append([
        (cond_d["no_stoke"]["picked"], cond_d["no_stoke"]["offered"]),
        (cond_d["stoke_in_deck"]["picked"], cond_d["stoke_in_deck"]["offered"]),
    ])
hm2_pct = np.array([[100.0 * k / n for k, n in row] for row in hm2_counts])
ax_d2 = d_axes(AD_X0, AD_W)
ax_d2.pcolormesh(np.arange(3), ROW_EDGES3, hm2_pct, cmap=CMAP_D, norm=NORM_D, edgecolors="white", linewidth=0.9)
for r_i in range(3):
    yc = 0.5 * (ROW_EDGES3[r_i] + ROW_EDGES3[r_i + 1])
    for c_j in range(2):
        cell_text(ax_d2, c_j + 0.5, yc, hm2_pct[r_i, c_j], hm2_counts[r_i][c_j])
ax_d2.set_ylim(5, 0)
ax_d2.set_xlim(0, 2)
ax_d2.set_xticks(np.arange(2) + 0.5)
ax_d2.set_xticklabels(["−", "+"], fontsize=FS_AX)
ax_d2.set_xlabel("Stoke in deck", fontsize=FS_AX, labelpad=2.0)
frame_heat(ax_d2)

# headers: one colour bar over the main heat map, the owners of the two matrices
HEAD_Y = D_Y1 + 0.018
ax_cbar = fig.add_axes([MAIN_X0 / PT_W, HEAD_Y, MAIN_W / PT_W, 0.013])
cb_d = fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM_D, cmap=CMAP_D), cax=ax_cbar, orientation="horizontal")
cb_d.set_ticks([0, 50, 100])
cb_d.outline.set_linewidth(0.5)
cb_d.outline.set_edgecolor(INK2)
ax_cbar.xaxis.set_ticks_position("top")
ax_cbar.tick_params(labelsize=FS_TICK, length=2.0, pad=1.5)
AXP.text(MAIN_X0 - 4.0, fy(HEAD_Y + 0.0065), "Taken (%)", fontsize=7.0, color=INK, ha="right", va="center")
AXP.text(OURS_X0 + OURS_W / 2.0, fy(HEAD_Y + 0.0065), "Ours, 550M", fontsize=FS_LAB, fontweight="bold", color=BLUE,
         ha="center", va="center")
AXP.text(AD_X0 + AD_W / 2.0, fy(HEAD_Y + 0.0065), "AlphaDefect", fontsize=FS_LAB, fontweight="bold", color=VIOLET,
         ha="center", va="center")

# =========================================================================
# Panel letters
# =========================================================================
for lab, (gx, gy) in zip("abcd", [(0.012, 0.986), (0.545, 0.986), (0.012, 0.532), (0.500, 0.532)]):
    fig.text(gx, gy, lab, fontsize=11, fontweight="bold", va="top", color=INK)

for ext in ("pdf", "svg", "png"):
    fig.savefig(os.path.join(R, "paper", "figs", f"fig5_boundary.{ext}"), dpi=350)

_re_peak = max(re_pts, key=lambda d: d["picked"] / d["offered"])
print(
    f"written paper/figs/fig5_boundary.{{pdf,svg,png}} at 350 DPI | "
    f"panel a cond5: u{_re_last['update']} (+{_re_last['steps_m']:.1f}M), "
    f"win={_re_last['win_pct']:.1f}%, picked={_re_last['picked']}/{_re_last['offered']} "
    f"({100*_re_last['picked']/_re_last['offered']:.1f}%), "
    f"energy={_re_last['energy_gained']}/{_re_last['plays']} ({_re_last['energy_pct']:.1f}%) | "
    f"panel b rediscovery: {len(re_pts)} pts (0..{_re_last['steps_m']:.1f}M), "
    f"total={re_tot_p}/{re_tot_o}, peak=u{_re_peak['update']} (+{_re_peak['steps_m']:.1f}M) "
    f"{_re_peak['picked']}/{_re_peak['offered']} ({100*_re_peak['picked']/_re_peak['offered']:.1f}%)"
)
