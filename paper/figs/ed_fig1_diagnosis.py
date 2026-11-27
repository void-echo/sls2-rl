"""Extended Data Figure 1: Counterfactual room-entry death diagnosis, Act-3 double-boss health trajectories,
and terminal-win objective ablation.

  a, Counterfactual room-entry replay diagnosis (8x entry-HP and 8x full-HP replays across 400 Ascension 10 runs)
     across three Ironclad training generations (13.0%, 55.8% and 69.3% win rate) at Act-3 bosses, alongside the
     partition of all 117 losses across all rooms at 69.3% (0 normal-combat deaths).
  b, Entering-health trajectories across the four boss doors (Acts 1-3 and the consecutive Act-3 Boss 1 -> Boss 2
     door; -30 HP vs -55 HP in Boss 1), paired with win rates across all six ordered Act-3 boss pairs (78-88%).
  c, Pure terminal-win objective (v3w_iron_win, milestones removed) versus standard shaped control (v3h_iron_ctl)
     from a common ~72% fork (u44300) across 2,000 paired fresh1105 seeds per reading (+200M, +270M, +360M, and
     +860M when win_50850.json is present), pooled posterior comparison, and the stage-of-death trade-off.

Writes paper/figs/ed_fig1_diagnosis.{png,pdf,svg} (350 DPI).
"""
import glob
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
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(R, "tmp", "game_assets", "fig14_v2")
if not os.path.exists(os.path.join(ASSET_DIR, "badge_char_ironclad.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_DIR, "build_assets.py")], check=True, cwd=R)


def register_helvetica_bold():
    """Register Helvetica Bold face from macOS Helvetica.ttc so bold text renders as true bold."""
    dst = os.path.join(R, "tmp", "game_assets", "ed_figs", "Helvetica-Bold.ttf")
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
STEP = 131072

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


def wilson(k, n, z=1.96):
    p = k / max(1, n)
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return 100.0 * max(0.0, c - h), 100.0 * min(1.0, c + h)


def draw_beta_ridge(ax, y_base, k, n, col_line, col_fill, max_h=0.52, alpha_fill=0.38, lw=0.95, zorder=3):
    a_p, b_p = k + 1.0, (n - k) + 1.0
    mu = a_p / (a_p + b_p)
    sd = math.sqrt(a_p * b_p / (((a_p + b_p) ** 2) * (a_p + b_p + 1.0)))
    xg = np.linspace(max(0.001, mu - 4.3 * sd), min(0.999, mu + 4.3 * sd), 160)
    dens = beta_dist.pdf(xg, a_p, b_p)
    dens = dens / dens.max() * max_h
    xp = xg * 100.0
    ax.fill_between(xp, y_base, y_base - dens, color=col_fill, alpha=alpha_fill, lw=0, zorder=zorder)
    ax.plot(xp, y_base - dens, color=col_line, lw=lw, zorder=zorder + 0.2)


def fmt_diff(val):
    r = float(Decimal(str(val)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    if abs(r) < 0.05:
        return "+0.0"
    return f"+{r:.1f}" if r > 0 else f"{MINUS}{abs(r):.1f}"


# =========================================================================
# 1. Load Ground-Truth Data From Repository Reports & JSONs
# =========================================================================
def parse_death_fork_row(md_path, row_prefix):
    for line in open(md_path, encoding="utf-8"):
        if line.startswith(f"| {row_prefix} |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            tot = int(cells[1].split("/")[0])
            c_exec = int(cells[2].split("/")[0])
            c_hp = int(cells[3].split("/")[0])
            c_deck = int(cells[4].split("/")[0])
            assert c_exec + c_hp + c_deck == tot, (row_prefix, tot, c_exec, c_hp, c_deck)
            return tot, c_exec, c_hp, c_deck
    raise KeyError(f"{row_prefix} not found in {md_path}")


# Generation 1 (11-05, 13.0% A10): docs/results/v2r_kd_same_asc10_kl1/notes.md
notes_g1 = open(os.path.join(R, "docs", "results", "v2r_kd_same_asc10_kl1", "notes.md"), encoding="utf-8").read()
m_g1 = re.search(r"失败归类：不稳 / 血量 / 状态 \| (\d+) / (\d+) / (\d+) \| (\d+) / (\d+) / (\d+) \|", notes_g1)
assert m_g1 is not None
g1_vals = [int(x) for x in m_g1.groups()]
g1_exec, g1_hp, g1_deck = g1_vals[0] + g1_vals[3], g1_vals[1] + g1_vals[4], g1_vals[2] + g1_vals[5]
g1_tot = g1_exec + g1_hp + g1_deck  # 255

# Generation 2 (11-11, 55.8% A10): docs/results/v2r_kd_s1_main/death_fork_a10_u14050.md
p_df2 = os.path.join(R, "docs", "results", "v2r_kd_s1_main", "death_fork_a10_u14050.md")
g2_tot, g2_exec, g2_hp, g2_deck = parse_death_fork_row(p_df2, "第3幕 boss")

# Generation 3 (11-20, 69.3% A10): docs/results/v3_iron/death_fork_a10_u39800.md
p_df3 = os.path.join(R, "docs", "results", "v3_iron", "death_fork_a10_u39800.md")
g3_tot, g3_exec, g3_hp, g3_deck = parse_death_fork_row(p_df3, "第3幕 boss")
all3_tot, all3_exec, all3_hp, all3_deck = parse_death_fork_row(p_df3, "全部")

# HP timeline & ordered boss pairs: docs/results/v3_iron/hp_timeline_u39800.md
hp_txt = open(os.path.join(R, "docs", "results", "v3_iron", "hp_timeline_u39800.md"), encoding="utf-8").read()


def parse_hp_line(label):
    for line in hp_txt.splitlines():
        if line.strip().startswith(label):
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+", line[len(label):])]
            return nums  # [win, a3:lowhp, a3:deck, a3:unsteady, (earlier)]
    raise KeyError(label)


hp_pct_a1 = parse_hp_line("act 1 boss: HP at the door / max HP (%)")
hp_pct_a2 = parse_hp_line("act 2 boss: HP at the door / max HP (%)")
hp_pct_b1 = parse_hp_line("act 3 boss: HP at the door / max HP (%)")
hp_pct_b2 = parse_hp_line("act 3 boss 2: HP at the door / max HP (%)")
hp_loss_b1 = parse_hp_line("act 3 boss 1: HP lost in it (survivors)")

boss_pairs = []
BOSS_ICON_MAP = {
    "Queen": "boss_queen.png",
    "Test_Subject": "boss_test_subject.png",
    "Aeonglass": "boss_aeonglass.png",
}
for m in re.finditer(r"^\s*(Queen|Test_Subject|Aeonglass)\+(Queen|Test_Subject|Aeonglass)\s+n=\s*(\d+)\s+won\s+(\d+)",
                     hp_txt, flags=re.M):
    b1_nm, b2_nm, n_p, w_p = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    boss_pairs.append((b1_nm, b2_nm, w_p, n_p, 100.0 * w_p / n_p))
boss_pairs.sort(key=lambda x: x[4], reverse=True)

# Terminal-win objective ablation: tmp/fresh1125/{win,ctl}_*.json (auto-includes 50850 if present)
FORK_U = 44300
fresh_dir = os.path.join(R, "tmp", "fresh1125")
cand_us = sorted(
    int(os.path.basename(p).split("_")[1].split(".")[0])
    for p in glob.glob(os.path.join(fresh_dir, "win_*.json"))
    if os.path.exists(os.path.join(fresh_dir, f"ctl_{os.path.basename(p).split('_')[1]}"))
)
readings = []
for u in cand_us:
    c_eps = json.load(open(os.path.join(fresh_dir, f"ctl_{u}.json")))["episodes"]
    w_eps = json.load(open(os.path.join(fresh_dir, f"win_{u}.json")))["episodes"]
    C = {e["seed"]: e for e in c_eps}
    W = {e["seed"]: e for e in w_eps}
    ks = [k for k in C if k in W]
    n_k = len(ks)
    cw = sum(bool(C[k]["victory"]) for k in ks)
    ww = sum(bool(W[k]["victory"]) for k in ks)
    only_w = sum(1 for k in ks if W[k]["victory"] and not C[k]["victory"])
    only_c = sum(1 for k in ks if C[k]["victory"] and not W[k]["victory"])
    c_early = sum(1 for e in c_eps if not e["victory"] and e.get("act", 0) < 3)
    c_late = sum(1 for e in c_eps if not e["victory"] and e.get("act", 0) >= 3)
    w_early = sum(1 for e in w_eps if not e["victory"] and e.get("act", 0) < 3)
    w_late = sum(1 for e in w_eps if not e["victory"] and e.get("act", 0) >= 3)
    steps_M = round((u - FORK_U) * STEP / 1e6 / 10.0) * 10  # 200, 270, 360, 860
    readings.append({
        "u": u, "steps_M": int(steps_M), "n": n_k,
        "cw": cw, "ww": ww, "cp": 100.0 * cw / n_k, "wp": 100.0 * ww / n_k,
        "only_w": only_w, "only_c": only_c,
        "diff": 100.0 * (ww - cw) / n_k,
        "z": (only_w - only_c) / math.sqrt(max(1, only_w + only_c)),
        "c_early": c_early, "c_late": c_late,
        "w_early": w_early, "w_late": w_late,
    })

# =========================================================================
# 2. Figure Layout (8.5 x 5.25 in, 350 DPI)
# =========================================================================
FIG_W, FIG_H = 8.5, 5.25
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

fig.add_artist(FancyBboxPatch(
    (0.006, 0.008), 0.988, 0.984, transform=fig.transFigure,
    boxstyle="round,pad=0.0,rounding_size=0.014",
    facecolor="none", edgecolor=OUTER_BORDER, linewidth=0.8, zorder=0
))

# Top row: Panel a (left), Panel b (middle trajectory + right boss pairs)
ax_a_top = fig.add_axes([0.062, 0.904, 0.365, 0.060])
ax_a = fig.add_axes([0.062, 0.548, 0.365, 0.314])

ax_b1 = fig.add_axes([0.492, 0.548, 0.246, 0.366])
ax_b2 = fig.add_axes([0.818, 0.548, 0.160, 0.366])

# Bottom row: Panel c (left paired readings + pooled, right stage-of-death trade-off)
ax_c1 = fig.add_axes([0.088, 0.098, 0.488, 0.355])
ax_c2 = fig.add_axes([0.648, 0.098, 0.330, 0.355])

# =========================================================================
# Panel a: Counterfactual room-entry replay diagnosis
# =========================================================================
cols_a = [
    ("13.0%", g1_tot, g1_deck, g1_hp, g1_exec),
    ("55.8%", g2_tot, g2_deck, g2_hp, g2_exec),
    ("69.3%", g3_tot, g3_deck, g3_hp, g3_exec),
    ("All rooms\n(69.3%)", all3_tot, all3_deck, all3_hp, all3_exec),
]
xs_a = [0.0, 1.0, 2.0, 3.25]

# Top count strip (losses out of 400 runs)
ax_a_top.set_xlim(-0.52, 3.80)
ax_a_top.set_ylim(0.0, 1.95)
ax_a_top.axis("off")
add_sprite(ax_a_top, "badge_char_ironclad.png", (-0.34, 1.48), target_h_pt=16.5, zorder=6)
ax_a_top.text(1.12, 1.48, "Act-3 boss losses (per 400 games)", fontsize=FS_TICK, fontweight="bold",
              color=INK, ha="center", va="center")
ax_a_top.text(3.25, 1.48, "All losses", fontsize=FS_TICK, fontweight="bold",
              color=INK, ha="center", va="center")
for xp, (_, tot, _, _, _), is_all in zip(xs_a, cols_a, [False, False, False, True]):
    bg = BLUE_SOFT if is_all else TERRA_PILL
    fg = BLUE if is_all else TERRA
    ax_a_top.add_patch(FancyBboxPatch(
        (xp - 0.35, 0.10), 0.70, 0.86,
        boxstyle="round,pad=0.0,rounding_size=0.14",
        facecolor=bg, edgecolor="none", zorder=2
    ))
    ax_a_top.text(xp, 0.53, f"n = {tot}", fontsize=FS_LAB - 0.2, fontweight="bold",
                  color=fg, ha="center", va="center", zorder=3)

clean_ax(ax_a)
for gy in (25, 50, 75, 100):
    ax_a.axhline(gy, color=GRID, lw=0.55, zorder=0)

bar_w = 0.58
seg_specs = [
    ("Deck (≤50% full HP)", TERRA),
    ("Entry HP (>50% full HP)", AMBER),
    ("Execution (>50% entry HP)", TEAL),
]

for idx_c, (xp, (_, tot, c_dk, c_hp, c_ex)) in enumerate(zip(xs_a, cols_a)):
    pcts = [100.0 * c_dk / tot, 100.0 * c_hp / tot, 100.0 * c_ex / tot]
    bot = 0.0
    for j, (p_val, (_, col)) in enumerate(zip(pcts, seg_specs)):
        ax_a.bar(xp, p_val, width=bar_w, bottom=bot, color=col, edgecolor="white", linewidth=0.8, zorder=2)
        if p_val >= 11.5:
            txt = f"{p_val:.1f}%" if idx_c == 3 else f"{p_val:.0f}%"
            ax_a.text(xp, bot + p_val / 2.0, txt, ha="center", va="center",
                      fontsize=FS_CELL, fontweight="bold", color="white", zorder=4)
        else:
            ax_a.text(xp + bar_w / 2.0 + 0.05, bot + p_val / 2.0, f"{p_val:.0f}%",
                      ha="left", va="center", fontsize=FS_CELL, fontweight="bold", color=col, zorder=4)
        bot += p_val

ax_a.axvline(2.625, color=BORDER, lw=0.9, ls=(0, (3.0, 2.0)), zorder=3)
ax_a.set_xlim(-0.52, 3.80)
ax_a.set_ylim(0, 104)
ax_a.set_xticks(xs_a)
ax_a.set_xticklabels([c[0] for c in cols_a], fontsize=FS_TICK)
ax_a.set_yticks([0, 25, 50, 75, 100])
ax_a.set_xlabel("Agent's win rate at Ascension 10", labelpad=2.5)
ax_a.set_ylabel("Share of losses (%)", labelpad=2.0)

# Compact horizontal legend right above y=104 or inside clean upper margin
leg_handles = [
    Rectangle((0, 0), 1, 1, facecolor=TEAL, edgecolor="none", label="Execution"),
    Rectangle((0, 0), 1, 1, facecolor=AMBER, edgecolor="none", label="Entry HP"),
    Rectangle((0, 0), 1, 1, facecolor=TERRA, edgecolor="none", label="Deck"),
]
ax_a.legend(handles=leg_handles, loc="lower center", bbox_to_anchor=(0.48, 0.99),
            ncol=3, frameon=False, fontsize=FS_TICK, handlelength=0.9, handletextpad=0.35, columnspacing=0.85)

# =========================================================================
# Panel b1: Entering-HP trajectory across boss doors
# =========================================================================
clean_ax(ax_b1)
for gy in (40, 60, 80, 100):
    ax_b1.axhline(gy, color=GRID, lw=0.55, zorder=0)

# Highlight Boss 1 -> Boss 2 double-boss transition
ax_b1.axvspan(2.0, 3.0, color="#fdf4f0", alpha=0.85, zorder=0.2)

x_doors = np.arange(4)
y_win = [hp_pct_a1[0], hp_pct_a2[0], hp_pct_b1[0], hp_pct_b2[0]]
y_lowhp = [hp_pct_a1[1], hp_pct_a2[1], hp_pct_b1[1], hp_pct_b2[1]]
y_deck = [hp_pct_a1[2], hp_pct_a2[2], hp_pct_b1[2], hp_pct_b2[2]]

ax_b1.plot(x_doors, y_deck, "^--", color=AMBER, ms=4.8, mec="white", mew=0.5, lw=1.25,
           label="Deck loss (n = 27)", zorder=3)
ax_b1.plot(x_doors, y_lowhp, "s-", color=TERRA, ms=5.0, mec="white", mew=0.55, lw=1.65,
           label="Boss-2 HP loss (n = 42)", zorder=4)
ax_b1.plot(x_doors, y_win, "o-", color=BLUE, ms=5.2, mec="white", mew=0.55, lw=1.80,
           label="Wins (n = 283)", zorder=5)

# Key HP-drop callouts in Boss 1
ax_b1.text(2.58, 88.2, f"{MINUS}{hp_loss_b1[0]:.0f} HP", fontsize=FS_LAB, fontweight="bold",
           color=BLUE, ha="left", va="bottom", zorder=6)
ax_b1.text(2.36, 53.0, f"{MINUS}{hp_loss_b1[1]:.0f} HP", fontsize=FS_LAB, fontweight="bold",
           color=TERRA, ha="right", va="center", zorder=6)

# End-point values at Boss 2 door
ax_b1.text(3.08, y_win[3], f"{y_win[3]:.0f}%", fontsize=FS_LAB, fontweight="bold",
           color=BLUE, ha="left", va="center")
ax_b1.text(3.08, y_deck[3] + 2.2, f"{y_deck[3]:.0f}%", fontsize=FS_TICK, fontweight="bold",
           color=AMBER, ha="left", va="center")
ax_b1.text(3.08, y_lowhp[3] - 2.2, f"{y_lowhp[3]:.0f}%", fontsize=FS_LAB, fontweight="bold",
           color=TERRA, ha="left", va="center")

ax_b1.set_xlim(-0.22, 3.56)
ax_b1.set_ylim(24, 105)
ax_b1.set_xticks(x_doors)
ax_b1.set_xticklabels(["Act 1\nboss", "Act 2\nboss", "Act 3\nBoss 1", "Act 3\nBoss 2"], fontsize=FS_TICK)
ax_b1.set_yticks([40, 60, 80, 100])
ax_b1.set_ylabel("Entering health (% of max)", labelpad=2.0)
ax_b1.legend(loc="lower left", bbox_to_anchor=(0.01, 0.01), frameon=True, facecolor="white",
             edgecolor=BORDER, framealpha=0.95, fontsize=FS_TICK - 0.2, handlelength=1.4,
             borderpad=0.32, labelspacing=0.22)

# =========================================================================
# Panel b2: Win rates across the 6 ordered Act-3 boss pairs (with boss icons)
# =========================================================================
clean_ax(ax_b2)
for gx in (50, 75, 100):
    ax_b2.axvline(gx, color=GRID, lw=0.55, zorder=0)

for i, (b1_nm, b2_nm, w_p, n_p, wr_p) in enumerate(boss_pairs):
    ry = i
    lo_ci, hi_ci = wilson(w_p, n_p)
    ax_b2.barh(ry, wr_p, height=0.58, color=BLUE, alpha=0.86, edgecolor="white", linewidth=0.5, zorder=2)
    ax_b2.plot([lo_ci, hi_ci], [ry, ry], color=INK, lw=1.0, zorder=4)
    ax_b2.text(hi_ci + 2.5, ry, f"{wr_p:.0f}%", fontsize=FS_CELL, fontweight="bold",
               color=BLUE, va="center", ha="left", zorder=5)
    # Place the two real boss sprites to the left of x=0 with a vector arrow between them
    add_sprite(ax_b2, BOSS_ICON_MAP[b1_nm], (-37.0, ry), target_h_pt=15.0, zorder=6)
    ax_b2.annotate("", xy=(-16.2, ry), xytext=(-26.5, ry),
                   arrowprops=dict(arrowstyle="->,head_width=0.14,head_length=0.18", lw=0.75, color=INK2),
                   annotation_clip=False, zorder=6)
    add_sprite(ax_b2, BOSS_ICON_MAP[b2_nm], (-8.5, ry), target_h_pt=15.0, zorder=6)

ax_b2.set_xlim(0, 118)
ax_b2.set_ylim(5.55, -0.62)
ax_b2.set_xticks([0, 50, 100])
ax_b2.set_yticks([])
ax_b2.spines["left"].set_color(INK2)
ax_b2.set_xlabel("Pair win rate (%)", labelpad=2.5)
ax_b2.text(-0.32, 1.03, "Boss 1 & 2", transform=ax_b2.transAxes, fontsize=FS_TICK,
           fontweight="bold", color=INK, ha="left", va="bottom")
ax_b2.text(0.54, 1.03, "78–88% for all six", transform=ax_b2.transAxes, fontsize=FS_TICK,
           fontweight="bold", color=BLUE, ha="center", va="bottom")

# =========================================================================
# Panel c1: Pure win-only reward vs standard control across 2,000-seed readings
# =========================================================================
clean_ax(ax_c1)
for gx in (68, 70, 72, 74, 76):
    ax_c1.axvline(gx, color=GRID, lw=0.55, zorder=0)

# Pooled over first 3 readings (6,000 seeds, cited in manuscript) and over all available readings
r3 = readings[:3]
pool3_n = sum(r["n"] for r in r3)
pool3_cw = sum(r["cw"] for r in r3)
pool3_ww = sum(r["ww"] for r in r3)
pool3_ow = sum(r["only_w"] for r in r3)
pool3_oc = sum(r["only_c"] for r in r3)
pool3_diff = 100.0 * (pool3_ww - pool3_cw) / pool3_n
pool3_z = (pool3_ow - pool3_oc) / math.sqrt(max(1, pool3_ow + pool3_oc))

pool_all_n = sum(r["n"] for r in readings)
pool_all_cw = sum(r["cw"] for r in readings)
pool_all_ww = sum(r["ww"] for r in readings)
pool_all_ow = sum(r["only_w"] for r in readings)
pool_all_oc = sum(r["only_c"] for r in readings)
pool_all_diff = 100.0 * (pool_all_ww - pool_all_cw) / pool_all_n
pool_all_z = (pool_all_ow - pool_all_oc) / math.sqrt(max(1, pool_all_ow + pool_all_oc))

rows_c1 = []
for r in readings:
    rows_c1.append((f"+{r['steps_M']}M", r["cw"], r["ww"], r["n"], r["diff"], r["z"], False))
if len(readings) > 3:
    rows_c1.append(("Three readings\npooled", pool3_cw, pool3_ww, pool3_n, pool3_diff, pool3_z, True))
    rows_c1.append(("Four readings\npooled", pool_all_cw, pool_all_ww, pool_all_n, pool_all_diff, pool_all_z, True))
else:
    rows_c1.append(("Three readings\npooled", pool3_cw, pool3_ww, pool3_n, pool3_diff, pool3_z, True))

X_DELTA = 78.6
for i, (lab, cw, ww, n_tot, d_pts, z_v, is_pool) in enumerate(rows_c1):
    ry = i + (0.28 if is_pool else 0.0)
    if i % 2 == 0 and not is_pool:
        ax_c1.axhspan(ry - 0.46, ry + 0.46, color=ZEBRA, zorder=0.1)
    if is_pool:
        ax_c1.axhspan(ry - 0.44, ry + 0.44, color=BLUE_SOFT, alpha=0.65, zorder=0.1)
    base = ry + 0.26
    cp, wp = 100.0 * cw / n_tot, 100.0 * ww / n_tot
    draw_beta_ridge(ax_c1, base, cw, n_tot, SLATE, SLATE, max_h=0.52, alpha_fill=0.25, lw=0.85, zorder=2)
    draw_beta_ridge(ax_c1, base, ww, n_tot, BLUE, BLUE, max_h=0.52, alpha_fill=0.38, lw=0.95, zorder=3)
    ax_c1.plot([cp, wp], [base, base], color=INK, lw=1.2, zorder=4)
    ax_c1.scatter([cp], [base], s=24, facecolor="white", edgecolor=SLATE, linewidth=1.25, zorder=5)
    ax_c1.scatter([wp], [base], s=24, color=BLUE, edgecolor="white", linewidth=0.6, zorder=6)

    # Delta pill on the right
    pill_bg = BLUE_PILL if d_pts >= 0 else SLATE_PILL
    pill_fg = BLUE if d_pts >= 0 else INK2
    ax_c1.add_patch(FancyBboxPatch(
        (X_DELTA - 1.55, ry - 0.34), 3.10, 0.68,
        boxstyle="round,pad=0.0,rounding_size=0.14",
        facecolor=pill_bg, edgecolor="none", zorder=3
    ))
    ax_c1.text(X_DELTA, ry, fmt_diff(d_pts), fontsize=FS_CELL, fontweight="bold",
               color=pill_fg, ha="center", va="center", zorder=4)

sep_y = len(readings) - 0.36
ax_c1.plot([66.3, 80.3], [sep_y, sep_y], color=BORDER, lw=0.85, zorder=4)

yt_pos = [i + (0.28 if r[-1] else 0.0) for i, r in enumerate(rows_c1)]
ax_c1.set_xlim(66.3, 80.5)
ax_c1.set_ylim(yt_pos[-1] + 0.60, -0.78)
ax_c1.set_xticks([68, 70, 72, 74, 76])
ax_c1.set_yticks(yt_pos)
ax_c1.set_yticklabels([r[0] for r in rows_c1], fontsize=FS_TICK, linespacing=1.0)
ax_c1.tick_params(axis="y", length=0)
ax_c1.set_xlabel("Ascension 10 win rate on 2,000 paired seeds (%)", labelpad=2.5)
ax_c1.set_ylabel("Steps after branching", labelpad=2.0)
# centre the axis label on the four step rows so it clears the pooled rows below
ax_c1.yaxis.set_label_coords(-0.105, 1.0 - (1.5 + 0.78) / (yt_pos[-1] + 0.60 + 0.78))

# Legend header above ridges & delta column
ax_c1.scatter([68.4], [-0.46], s=22, facecolor="white", edgecolor=SLATE, linewidth=1.25, zorder=5, clip_on=False)
ax_c1.text(68.85, -0.46, "Standard reward", fontsize=FS_LAB - 0.2, fontweight="bold", color=SLATE, va="center")
ax_c1.scatter([73.1], [-0.46], s=22, color=BLUE, edgecolor="white", linewidth=0.6, zorder=5, clip_on=False)
ax_c1.text(73.55, -0.46, "Win-only reward", fontsize=FS_LAB - 0.2, fontweight="bold", color=BLUE, va="center")
ax_c1.text(X_DELTA, -0.46, "Δ (points)", fontsize=FS_LAB - 0.2, fontweight="bold", color=INK, ha="center", va="center")

# =========================================================================
# Panel c2: Stage-of-death trade-off (Acts 1-2 vs Act-3 double-boss losses)
# =========================================================================
clean_ax(ax_c2)
for gy in (100, 200, 300, 400):
    ax_c2.axhline(gy, color=GRID, lw=0.55, zorder=0)

c_e_arr = np.array([r["c_early"] for r in readings], dtype=float)
w_e_arr = np.array([r["w_early"] for r in readings], dtype=float)
c_l_arr = np.array([r["c_late"] for r in readings], dtype=float)
w_l_arr = np.array([r["w_late"] for r in readings], dtype=float)

# Sums for first 3 readings (manuscript text) and all readings
sum3_ce, sum3_we = int(c_e_arr[:3].sum()), int(w_e_arr[:3].sum())
sum3_cl, sum3_wl = int(c_l_arr[:3].sum()), int(w_l_arr[:3].sum())

xc = np.array([0.0, 1.18])
bw = 0.34

means_ctl = [c_e_arr.mean(), c_l_arr.mean()]
means_win = [w_e_arr.mean(), w_l_arr.mean()]

ax_c2.bar(xc - bw / 2.0, means_ctl, width=bw, color=SLATE, alpha=0.82, edgecolor="white",
          linewidth=0.6, label="Standard", zorder=2)
ax_c2.bar(xc + bw / 2.0, means_win, width=bw, color=BLUE, alpha=0.90, edgecolor="white",
          linewidth=0.6, label="Win-only", zorder=2)

# Overlay paired reading trajectories so every single JSON file is directly visible
jitter = np.linspace(-0.045, 0.045, len(readings))
for idx_r, r in enumerate(readings):
    dx = jitter[idx_r]
    ax_c2.plot([xc[0] - bw / 2.0 + dx, xc[0] + bw / 2.0 + dx], [r["c_early"], r["w_early"]],
               color=INK2, lw=0.55, alpha=0.55, zorder=3)
    ax_c2.scatter([xc[0] - bw / 2.0 + dx], [r["c_early"]], s=14, facecolor="white", edgecolor=SLATE,
                  linewidth=0.9, zorder=4)
    ax_c2.scatter([xc[0] + bw / 2.0 + dx], [r["w_early"]], s=14, color=BLUE, edgecolor="white",
                  linewidth=0.5, zorder=4)

    ax_c2.plot([xc[1] - bw / 2.0 + dx, xc[1] + bw / 2.0 + dx], [r["c_late"], r["w_late"]],
               color=INK2, lw=0.55, alpha=0.55, zorder=3)
    ax_c2.scatter([xc[1] - bw / 2.0 + dx], [r["c_late"]], s=14, facecolor="white", edgecolor=SLATE,
                  linewidth=0.9, zorder=4)
    ax_c2.scatter([xc[1] + bw / 2.0 + dx], [r["w_late"]], s=14, color=BLUE, edgecolor="white",
                  linewidth=0.5, zorder=4)

# Mean + cumulative callouts across all plotted readings
sum_all_ce, sum_all_we = int(c_e_arr.sum()), int(w_e_arr.sum())
sum_all_cl, sum_all_wl = int(c_l_arr.sum()), int(w_l_arr.sum())
d_early = means_win[0] - means_ctl[0]
d_late = means_win[1] - means_ctl[1]
ax_c2.text(xc[0], max(w_e_arr) + 22, f"+{d_early:.0f} per 2,000",
           fontsize=FS_CELL, fontweight="bold", color=TERRA, ha="center", va="bottom", linespacing=1.08)
ax_c2.text(xc[1], max(c_l_arr) + 18, f"{MINUS}{abs(d_late):.0f} per 2,000",
           fontsize=FS_CELL, fontweight="bold", color=TEAL, ha="center", va="bottom", linespacing=1.08)

add_sprite(ax_c2, "badge_char_ironclad.png", (0.52, 448), target_h_pt=18.5, zorder=6)

ax_c2.set_xlim(-0.48, 1.66)
ax_c2.set_ylim(0, 520)
ax_c2.set_xticks(xc)
ax_c2.set_xticklabels(["Acts 1–2\nlosses", "Act-3 boss\nlosses"], fontsize=FS_TICK)
ax_c2.set_yticks([0, 150, 300, 450])
ax_c2.set_ylabel("Losses per 2,000 games", labelpad=2.0)
ax_c2.legend(loc="upper left", bbox_to_anchor=(0.01, 1.00), frameon=False, fontsize=FS_TICK,
             handlelength=1.0, handletextpad=0.35)

# =========================================================================
# Panel letters
# =========================================================================
for lab, (gx, gy) in zip("abc", [(0.014, 0.980), (0.446, 0.980), (0.014, 0.490)]):
    fig.text(gx, gy, lab, fontsize=FS_LETTER, fontweight="bold", va="top", color=INK)

out_base = os.path.join(R, "paper", "figs", "ed_fig1_diagnosis")
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out_base}.{ext}", dpi=350)
print(f"written {out_base}.{{png,pdf,svg}}")
print("c readings:", [(r["u"], r["steps_M"], round(r["wp"], 2), round(r["cp"], 2), round(r["diff"], 2)) for r in readings])
print(f"c pooled 3x2k: {100*pool3_ww/pool3_n:.2f}% vs {100*pool3_cw/pool3_n:.2f}% ({pool3_diff:+.2f}, z={pool3_z:.2f}); "
      f"4x2k: {100*pool_all_ww/pool_all_n:.2f}% vs {100*pool_all_cw/pool_all_n:.2f}% ({pool_all_diff:+.2f}, z={pool_all_z:.2f})")
