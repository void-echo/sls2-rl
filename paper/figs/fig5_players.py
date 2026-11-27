"""Figure 5: against strong human players and an independently built agent, alike on cards that work alone, absent on
cards that only work together.

  a, Pick rates of our agents against strong human players for 399 cards, with marginal densities on both axes; the
     eight cards of panel b are ringed.
  b, Eight example cards: four that work alone and that it takes more than strong humans, four combination cards that
     strong humans take and it never takes; strong human (diamond) and our (circle) pick rate on a 0-100 track under
     each card, with a self-contained mini-key.
  c, Rank agreement with strong human pick rates, per character (Spearman rho over the cards of panel a, with 95% CI).
  d, Strong humans, AlphaDefect and ours on the 34 human-favoured cards AlphaDefect takes least, with ours - AlphaDefect.

Data: paper/figdata/fresh1123_main_prefs.json, tmp/archetype_web/spire_codex_<character>_wr75.txt (strong human pick
rates), paper/data_cross_agent_ignored_cards.json (d, using human_wr75_pct).
Lettering follows paper/figs/STYLE.md. Writes paper/figs/fig5_players.{pdf,svg,png} (350 DPI).
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIRS = [os.path.join(R, "tmp", "game_assets", "fig45_split"), os.path.join(R, "tmp", "game_assets", "fig14_v2")]
if not os.path.exists(os.path.join(ASSET_DIRS[1], "badge_char_fool.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_DIRS[1], "build_assets.py")], check=True, cwd=R)
if not os.path.exists(os.path.join(ASSET_DIRS[0], "spec_pull_from_below.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_DIRS[0], "build_assets.py")], check=True, cwd=R)

sys.path.insert(0, os.path.join(R, "tmp", "archetype"))
import archetypes as AR  # noqa: E402


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
        font_manager.fontManager.addfont(dst)
    except Exception as exc:  # other platforms: keep the regular face
        print("Helvetica Bold not registered:", exc)


register_helvetica_bold()

BLUE = "#2166ac"
BLUE_MID = "#2b6cb0"
BLUE_SOFT = "#eaf3fc"
TERRA = "#c85228"
VIOLET = "#5e4cc4"
SLATE = "#64748b"
INK = "#14181f"
INK2 = "#4a5568"
MUTED = "#718096"
ZEBRA = "#f5f7fa"
OUTER_BORDER = "#e2e8f0"
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 6.8,
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

FS_AX = 7.6      # axis labels
FS_TICK = 6.8    # tick labels
FS_LAB = 7.2     # in-plot labels


def asset(fn):
    for d in ASSET_DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def load_rgba_clean(fn, target_h_px):
    im = Image.open(asset(fn)).convert("RGBA")
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


def sprite_aspect(fn):
    with Image.open(asset(fn)) as im:
        return im.width / im.height


def add_sprite(ax, fn, xy, target_h_pt, xycoords="data", xybox=None, align=(0.5, 0.5), zorder=6, clip_on=False):
    arr, h_px = load_rgba_clean(fn, target_h_pt * 5.0)
    oi = OffsetImage(arr, zoom=target_h_pt / h_px, interpolation="bilinear", resample=True)
    if xybox is not None:
        ab = AnnotationBbox(oi, xy, xybox=xybox, xycoords=xycoords, boxcoords="offset points",
                            box_alignment=align, frameon=False, pad=0, zorder=zorder, annotation_clip=clip_on)
    else:
        ab = AnnotationBbox(oi, xy, xycoords=xycoords, box_alignment=align, frameon=False, pad=0, zorder=zorder,
                            annotation_clip=clip_on)
    ab.set_clip_on(clip_on)
    ax.add_artist(ab)


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.3, pad=2.0)


def bounded_kde_1d(vals, grid, lo=0.0, hi=100.0, bw=0.22):
    arr = np.asarray(vals, dtype=float)
    reflected = np.concatenate([arr, 2.0 * lo - arr, 2.0 * hi - arr])
    kde = gaussian_kde(reflected, bw_method=bw)
    dens = kde(grid) * 3.0
    dens[(grid < lo) | (grid > hi)] = 0.0
    return dens


def pct_half_up(v):
    return int(math.floor(v + 0.5))


# Strictly gated multi-card archetypes (as in fig4_learned.py)
GATED_ARCHS = {
    "Ironclad": ("消耗",),
    "Silent": ("毒", "弃牌奇巧"),
    "Defect": ("闪电（电流相生）",),
    "Regent": ("铸剑", "辉星"),
    "Necrobinder": ("灾厄", "灵魂"),
}
GATED_PAYOFFS = set()
for ch_name, arch_names in GATED_ARCHS.items():
    for an in arch_names:
        GATED_PAYOFFS |= set(AR.ARCH[ch_name][an].get("payoffs", set()))
GATED_PAYOFFS |= {
    "CARD.DARK_EMBRACE", "CARD.FEEL_NO_PAIN", "CARD.SECOND_WIND", "CARD.BURNING_PACT",
    "CARD.FORGOTTEN_RITUAL", "CARD.WHIRLWIND", "CARD.SWORD_SAGE", "CARD.THE_SMITH",
    "CARD.SEEKING_EDGE", "CARD.CONQUEROR", "CARD.STARDUST", "CARD.BLACK_HOLE",
    "CARD.CHILD_OF_THE_STARS", "CARD.RADIATE", "CARD.SYNCHRONIZE", "CARD.DARKNESS",
    "CARD.CONSUMING_SHADOW", "CARD.STORM", "CARD.TEMPEST", "CARD.VOLTAIC", "CARD.THUNDER",
    "CARD.TACTICIAN", "CARD.REFLEX", "CARD.CATALYST", "CARD.BUBBLE_BUBBLE", "CARD.BURST",
    "CARD.END_OF_DAYS", "CARD.NO_ESCAPE", "CARD.DEATHBRINGER", "CARD.SNAP", "CARD.RATTLE",
    "CARD.TRASH_TO_TREASURE", "CARD.SMOKESTACK",
}

prefs = json.load(open(os.path.join(R, "paper/figdata/fresh1123_main_prefs.json")))

FIG_W, FIG_H = 8.85, 5.95
PT_W, PT_H = FIG_W * 72.0, FIG_H * 72.0
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
fig.add_artist(FancyBboxPatch(
    (0.006, 0.008), 0.988, 0.984, transform=fig.transFigure,
    boxstyle="round,pad=0.0,rounding_size=0.014",
    facecolor="none", edgecolor=OUTER_BORDER, linewidth=0.8, zorder=0
))

CHAR_BADGES = {c: f"badge_char_{c.lower()}.png" for c in ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")}

# =========================================================================
# Panel a: our pick rate against strong human players (399 cards, ~1:1 square aspect)
# =========================================================================
eng = json.load(open(os.path.join(R, "engine/localization_eng/cards.json")))
name2id = {v.strip().lower(): "CARD." + k[:-6] for k, v in eng.items() if k.endswith(".title")}
char_specs = [
    ("iron_ctl", "ironclad", "Ironclad"),
    ("defect_main", "defect", "Defect"),
    ("regent_main", "regent", "Regent"),
    ("silent_main", "silent", "Silent"),
    ("necro_main", "necrobinder", "Necrobinder"),
]
pts_standalone, pts_gated = [], []
for tag, fn, cname in char_specs:
    cards = prefs[tag]["cards"]
    for line in open(os.path.join(R, f"tmp/archetype_web/spire_codex_{fn}_wr75.txt")):
        if line.startswith("#") or line.startswith("Card |"):
            continue
        p = [x.strip() for x in line.split("|")]
        if len(p) < 5:
            continue
        cid = name2id.get(p[0].lower())
        if not cid or cid not in cards or cards[cid][1] < 20:
            continue
        pk, off = cards[cid]
        hpct = float(p[2].rstrip("%"))
        apct = 100.0 * pk / off
        if cid in GATED_PAYOFFS:
            pts_gated.append((hpct, apct, p[0], cname))
        else:
            pts_standalone.append((hpct, apct, p[0], cname))
all_pts = {nm: (x0, y0) for x0, y0, nm, _ in pts_standalone + pts_gated}

A_AX = [0.066, 0.530, 0.208, 0.348]
ax_a = fig.add_axes(A_AX)
ax_a_top = fig.add_axes([A_AX[0], A_AX[1] + A_AX[3] + 0.006, A_AX[2], 0.074])
ax_a_right = fig.add_axes([A_AX[0] + A_AX[2] + 0.004, A_AX[1], 0.036, A_AX[3]])
X_LIM_A = (-3.5, 93.5)
Y_LIM_A = (-5.0, 104.0)

for s in ax_a.spines.values():
    s.set_visible(True)
    s.set_edgecolor(INK2)
    s.set_linewidth(0.6)
ax_a.tick_params(length=2.2, pad=2.0, labelsize=FS_TICK)
ax_a.plot([0, 90], [0, 90], color=MUTED, lw=0.7, ls=(0, (2.2, 2.0)), zorder=1)
ax_a.axhline(10.0, color=TERRA, lw=0.6, ls=(0, (3, 2)), alpha=0.55, zorder=1)

hx_s = [x[0] for x in pts_standalone]
ay_s = [x[1] for x in pts_standalone]
hx_g = [x[0] for x in pts_gated]
ay_g = [x[1] for x in pts_gated]
ax_a.scatter(hx_s, ay_s, s=13.0, color=BLUE, alpha=0.45, edgecolors="none", zorder=2)
ax_a.scatter(hx_g, ay_g, s=24.0, marker="^", color=TERRA, alpha=0.95, edgecolors="white", linewidths=0.4, zorder=4)

n_low = sum(1 for v in ay_g if v <= 10.0)
ax_a.text(72.5, 29.0, f"{n_low} of {len(ay_g)}\nunder 10%", fontsize=FS_LAB, fontweight="bold",
          color=TERRA, ha="center", va="center", linespacing=1.0, path_effects=HALO, zorder=6)
ax_a.set_xlim(*X_LIM_A)
ax_a.set_ylim(*Y_LIM_A)
ax_a.set_xticks([0, 25, 50, 75])
ax_a.set_yticks([0, 25, 50, 75, 100])
ax_a.set_xlabel("Pick rate of strong human players (%)", fontsize=FS_AX, labelpad=2.0)
ax_a.set_ylabel("Our pick rate (%)", fontsize=FS_AX, labelpad=2.0)

xg_top = np.linspace(0.0, 91.0, 240)
kd_top_s = bounded_kde_1d(hx_s, xg_top, lo=0.0, hi=91.0, bw=0.24)
kd_top_g = bounded_kde_1d(hx_g, xg_top, lo=0.0, hi=91.0, bw=0.28)
kd_top_s /= max(1e-9, kd_top_s.max())
kd_top_g /= max(1e-9, kd_top_g.max()) * 1.08
ax_a_top.fill_between(xg_top, 0, kd_top_s, color=BLUE, alpha=0.26, lw=0, zorder=2)
ax_a_top.plot(xg_top, kd_top_s, color=BLUE, lw=1.0, zorder=3)
ax_a_top.fill_between(xg_top, 0, kd_top_g, color=TERRA, alpha=0.30, lw=0, zorder=3)
ax_a_top.plot(xg_top, kd_top_g, color=TERRA, lw=1.05, zorder=4)
ax_a_top.set_xlim(*X_LIM_A)
ax_a_top.set_ylim(0, 1.18)
ax_a_top.set_xticks([])
ax_a_top.set_yticks([])
for s in ("top", "left", "right"):
    ax_a_top.spines[s].set_visible(False)
ax_a_top.spines["bottom"].set_color(INK2)
ax_a_top.spines["bottom"].set_linewidth(0.55)
ax_a_top.scatter([46.0], [0.98], s=16, color=BLUE, alpha=0.75, edgecolors="none", clip_on=False, zorder=5)
ax_a_top.text(49.5, 0.98, "Standalone", fontsize=FS_LAB, fontweight="bold", color=BLUE, ha="left", va="center")
ax_a_top.scatter([46.0], [0.55], s=24, marker="^", color=TERRA, edgecolors="white", linewidths=0.4,
                 clip_on=False, zorder=5)
ax_a_top.text(49.5, 0.55, "Combination", fontsize=FS_LAB, fontweight="bold", color=TERRA, ha="left", va="center")

yg_rt = np.linspace(0.0, 100.0, 320)
kd_rt_s = bounded_kde_1d(ay_s, yg_rt, lo=0.0, hi=100.0, bw=0.20)
kd_rt_g = bounded_kde_1d(ay_g, yg_rt, lo=0.0, hi=100.0, bw=0.13)
kd_rt_s = kd_rt_s / max(1e-9, kd_rt_s.max()) * 0.58
kd_rt_g = kd_rt_g / max(1e-9, kd_rt_g.max()) * 1.00
ax_a_right.fill_betweenx(yg_rt, 0, kd_rt_s, color=BLUE, alpha=0.26, lw=0, zorder=2)
ax_a_right.plot(kd_rt_s, yg_rt, color=BLUE, lw=1.0, zorder=3)
ax_a_right.fill_betweenx(yg_rt, 0, kd_rt_g, color=TERRA, alpha=0.40, lw=0, zorder=4)
ax_a_right.plot(kd_rt_g, yg_rt, color=TERRA, lw=1.2, zorder=5)
ax_a_right.set_ylim(*Y_LIM_A)
ax_a_right.set_xlim(0, 1.15)
ax_a_right.set_xticks([])
ax_a_right.set_yticks([])
for s in ("top", "right", "bottom"):
    ax_a_right.spines[s].set_visible(False)
ax_a_right.spines["left"].set_color(INK2)
ax_a_right.spines["left"].set_linewidth(0.55)

# =========================================================================
# Panel b: eight example cards with a 0-100 track + self-contained mini-key
# =========================================================================
GALLERY = [
    [("Demon Form", "Ironclad"), ("Knife Trap", "Silent"), ("Manifest Authority", "Regent"),
     ("Pull from Below", "Necrobinder")],
    [("Dark Embrace", "Ironclad"), ("Tactician", "Silent"), ("Darkness", "Defect"), ("Sword Sage", "Regent")],
]
GAL = [0.322, 0.022, 0.290, 0.954]
ax_b = fig.add_axes(GAL)
GW, GH = GAL[2] * PT_W, GAL[3] * PT_H
ax_b.set_xlim(0, GW)
ax_b.set_ylim(0, GH)
ax_b.axis("off")

# Self-contained horizontal mini-key centered at top of Panel b
KEY_Y = GH - 7.5
ax_b.add_patch(FancyBboxPatch((20.0, KEY_Y - 6.0), GW - 38.0, 12.0,
                              boxstyle="round,pad=0.0,rounding_size=4.5",
                              facecolor="#f8fafc", edgecolor=OUTER_BORDER, linewidth=0.65, zorder=1))
ax_b.scatter([34.5], [KEY_Y], s=24, marker="D", color=SLATE, edgecolor="white", linewidth=0.5, zorder=4)
ax_b.text(41.0, KEY_Y, "Strong humans", fontsize=6.9, fontweight="bold", color=SLATE, ha="left", va="center")
ax_b.scatter([114.5], [KEY_Y], s=28, marker="o", color=BLUE, edgecolor="white", linewidth=0.5, zorder=4)
ax_b.text(121.0, KEY_Y, "Ours", fontsize=6.9, fontweight="bold", color=BLUE, ha="left", va="center")

COL_W = GW / 2.0
HEAD_H = 30.0
ROW_H = (GH - HEAD_H) / 4.0
ART_W = COL_W - 18.0
ART_H = ART_W / sprite_aspect("spec_demon_form.png")
for ci, (col_cards, ccol, head) in enumerate(zip(GALLERY, (BLUE, TERRA), ("Works alone", "Only works together"))):
    x0 = ci * COL_W + 7.0
    ax_b.text(x0 + ART_W / 2.0, GH - 23.0, head, fontsize=FS_LAB + 0.3, fontweight="bold", color=ccol,
              ha="center", va="center")
    for ri, (cname, ch) in enumerate(col_cards):
        hx0, ay0 = all_pts[cname]
        ytop = GH - HEAD_H - ri * ROW_H
        ax_b.text(x0 + 1.0, ytop - 4.8, cname, fontsize=7.5, fontweight="bold", color=ccol, ha="left", va="center")
        y_art0 = ytop - 10.2 - ART_H
        sfn = "spec_" + cname.lower().replace(" ", "_") + ".png"
        add_sprite(ax_b, sfn, (x0, y_art0), target_h_pt=ART_H, align=(0.0, 0.0), zorder=5)
        add_sprite(ax_b, CHAR_BADGES[ch], (x0 + ART_W - 12.0, y_art0 + ART_H - 12.0), target_h_pt=16.8,
                   align=(0.5, 0.5), zorder=7)
        # track from 0% (left edge of the art) to 100% (right edge)
        ty = y_art0 - 5.2
        tx = lambda v: x0 + 3.0 + (ART_W - 6.0) * v / 100.0
        ax_b.plot([tx(0), tx(100)], [ty, ty], color="#d5dde8", lw=1.6, solid_capstyle="round", zorder=2)
        ax_b.plot([tx(hx0), tx(ay0)], [ty, ty], color=ccol, lw=1.8, alpha=0.55, zorder=3)
        ax_b.scatter([tx(hx0)], [ty], s=25, marker="D", color=SLATE, edgecolor="white", linewidth=0.5, zorder=4)
        ax_b.scatter([tx(ay0)], [ty], s=30, color=BLUE, edgecolor="white", linewidth=0.5, zorder=5)
        ax_b.text(tx(hx0), ty - 8.2, f"{pct_half_up(hx0)}%", fontsize=7.4, fontweight="bold", color=SLATE,
                  ha="center", va="center", zorder=6)
        ax_b.text(tx(ay0), ty - 8.2, f"{pct_half_up(ay0)}%", fontsize=7.4, fontweight="bold",
                  color=BLUE if ay0 >= 10 else TERRA, ha="center", va="center", zorder=6)
        # ring the card in panel a
        ax_a.scatter([hx0], [ay0], s=56 if ccol == BLUE else 64, marker="o" if ccol == BLUE else "^",
                     facecolor=ccol, edgecolor=INK, linewidth=0.9, zorder=7)
        print(f"b {cname}: strong_human={hx0:.1f}% ({pct_half_up(hx0)}%), ours={ay0:.1f}% ({pct_half_up(ay0)}%)")

# =========================================================================
# Panel c: rank agreement with strong human pick rates, per character (taller & richer)
# =========================================================================
C_AX = [0.118, 0.085, 0.178, 0.335]
ax_c = fig.add_axes(C_AX)
clean_ax(ax_c)
for gx_c in (0.2, 0.4, 0.6):
    ax_c.axvline(gx_c, color="#edf2f7", lw=0.55, zorder=0)

rho_rows = []
for _, _, cname in char_specs:
    pts = [t for t in pts_standalone + pts_gated if t[3] == cname]
    rval = spearmanr([t[0] for t in pts], [t[1] for t in pts]).correlation
    rho_rows.append((cname, rval, len(pts)))
rho_rows.sort(key=lambda r: -r[1])

RHO_XMAX = 0.72
ax_c.set_ylim(len(rho_rows) - 0.44, -0.56)
ax_c.set_xlim(0.0, RHO_XMAX)
rho_pt = RHO_XMAX / (C_AX[2] * PT_W)      # data units per point
for idx_r, (ch_lbl, rval, n_c) in enumerate(rho_rows):
    bcol = BLUE_MID if rval >= 0.35 else SLATE
    ax_c.barh(idx_r, rval, height=0.62, color=bcol, edgecolor="white", linewidth=0.5, zorder=2)
    ax_c.text(rval + 2.8 * rho_pt, idx_r, f"{rval:.2f}", va="center", ha="left",
              fontsize=FS_LAB, fontweight="bold", color=BLUE if rval >= 0.35 else SLATE, zorder=4)
    add_sprite(ax_c, CHAR_BADGES[ch_lbl], (-12.5 * rho_pt, idx_r), target_h_pt=19.5, xycoords="data",
               align=(0.5, 0.5), zorder=6, clip_on=False)
    ax_c.text(-25.0 * rho_pt, idx_r, ch_lbl, va="center", ha="right", fontsize=7.0, color=INK, clip_on=False)
    print(f"c {ch_lbl}: rho={rval:.2f} (n={n_c})")

ax_c.set_yticks([])
ax_c.set_xticks([0.0, 0.2, 0.4, 0.6])
ax_c.set_xticklabels(["0", "0.2", "0.4", "0.6"], fontsize=FS_TICK)
ax_c.spines["bottom"].set_bounds(0.0, 0.6)
ax_c.set_xlabel("Spearman ρ with strong humans", fontsize=FS_AX, labelpad=2.0)
ax_c.tick_params(length=2.0, pad=1.8)

# =========================================================================
# Panel d: strong humans, AlphaDefect and ours on 34 cards, with ours - AlphaDefect
# =========================================================================
E_AX = [0.778, 0.085, 0.116, 0.842]
E_DX = [0.900, 0.085, 0.086, 0.842]
ax_e = fig.add_axes(E_AX)
ax_e_d = fig.add_axes(E_DX)
E_XLIM = (-2.5, 102.0)
upt = (E_XLIM[1] - E_XLIM[0]) / (E_AX[2] * PT_W)     # data units per point on ax_e

rows = json.load(open(os.path.join(R, "paper/data_cross_agent_ignored_cards.json")))["rows"]
chars_order = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]
rows.sort(key=lambda r: (chars_order.index(r["character"]), max(r["ours_pct"], r["alphadefect_pct"]) >= 15.0,
                         -r["human_wr75_pct"]))
for gx_e in (25, 50, 75, 100):
    ax_e.axvline(gx_e, color="#edf2f7", lw=0.5, zorder=0)

GROUP_GAP = 0.75
X_SQ0, X_SQ1 = -7.6 * upt, -3.6 * upt        # type square
X_NAME = -9.6 * upt                          # right edge of the card names
X_RULE = -83.0 * upt                         # group rule
X_HEAD = -94.0 * upt                         # badge and character name
X_BG0 = -8.6 * upt

y = 0.0
prev = None
row_idx_in_char = 0
group_span = {}
for r in rows:
    if prev is not None and r["character"] != prev:
        y += GROUP_GAP
        row_idx_in_char = 0
    prev = r["character"]
    group_span.setdefault(prev, [y, y])[1] = y
    diff = r["ours_pct"] - r["alphadefect_pct"]
    divergent = abs(diff) >= 20.0
    is_gated = r["id"] in GATED_PAYOFFS
    if divergent:
        ax_e.add_patch(Rectangle((X_BG0, y - 0.47), 102.0 - X_BG0, 0.94, facecolor=BLUE_SOFT, edgecolor="none",
                                 zorder=0, clip_on=False))
        ax_e_d.add_patch(Rectangle((-16.0, y - 0.47), 144.0, 0.94, facecolor=BLUE_SOFT, edgecolor="none", zorder=0))
        ax_e.plot([r["alphadefect_pct"], r["ours_pct"]], [y, y], color=BLUE, lw=1.1, alpha=0.78, zorder=2)
    elif row_idx_in_char % 2 == 0:
        ax_e.add_patch(Rectangle((X_BG0, y - 0.47), 102.0 - X_BG0, 0.94, facecolor=ZEBRA, edgecolor="none",
                                 zorder=0, clip_on=False))
        ax_e_d.add_patch(Rectangle((-16.0, y - 0.47), 144.0, 0.94, facecolor=ZEBRA, edgecolor="none", zorder=0))
    ax_e.add_patch(Rectangle((X_SQ0, y - 0.36), X_SQ1 - X_SQ0, 0.72,
                             facecolor=TERRA if is_gated else BLUE, edgecolor="none", zorder=4, clip_on=False))
    ax_e.text(X_NAME, y, r["card"], fontsize=7.0, fontweight="bold" if divergent else "normal",
              color=INK, ha="right", va="center", clip_on=False)
    ax_e.scatter([r["human_wr75_pct"]], [y], s=17, marker="D", color=SLATE, zorder=3, edgecolor="white", linewidth=0.4)
    ax_e.scatter([r["alphadefect_pct"]], [y], s=20, color=VIOLET, zorder=4, edgecolor="white", linewidth=0.4)
    ax_e.scatter([r["ours_pct"]], [y], s=20, color=BLUE, zorder=5, edgecolor="white", linewidth=0.4)
    if diff >= 15.0:
        bcol = BLUE
    elif diff <= -0.8:
        bcol = VIOLET
    elif abs(diff) >= 0.8:
        bcol = "#94a3b8"
    else:
        bcol = "#cbd5e1"
    bw = diff if abs(diff) >= 0.8 else (0.6 if diff >= 0 else -0.6)
    ax_e_d.barh(y, bw, height=0.66, color=bcol, edgecolor="none", zorder=3)
    if divergent:
        ax_e_d.text(diff + 4.0, y, f"{diff:+.0f}", fontsize=FS_LAB, fontweight="bold",
                    color=BLUE, va="center", ha="left", zorder=5)
    y += 1.0
    row_idx_in_char += 1

Y_BOT_E, Y_TOP_E = y - 0.35, -0.65
row_pt = (E_AX[3] * PT_H) / (Y_BOT_E - Y_TOP_E)      # points per row
for ch, (y0g, y1g) in group_span.items():
    ax_e.plot([X_RULE, X_RULE], [y0g - 0.42, y1g + 0.42], color="#b6c2d1", lw=1.3,
              solid_capstyle="round", clip_on=False, zorder=3)
    badge_rows = 21.0 / row_pt
    add_sprite(ax_e, CHAR_BADGES[ch], (X_HEAD, y0g - 0.42 + badge_rows / 2.0), target_h_pt=21.0,
               xycoords="data", align=(0.5, 0.5), zorder=6, clip_on=False)
    ax_e.text(X_HEAD, 0.5 * (y0g - 0.42 + badge_rows + y1g + 0.42) + 0.1, ch, fontsize=7.6, fontweight="bold",
              color=INK, ha="center", va="center", rotation=90, clip_on=False)
ax_e.axvline(0.0, color="#cbd5e1", lw=0.55, zorder=1)
ax_e_d.axvline(0.0, color=INK2, lw=0.65, zorder=2)
ax_e.set_ylim(Y_BOT_E, Y_TOP_E)
ax_e.set_xlim(*E_XLIM)
ax_e.set_yticks([])
ax_e.set_xticks([0, 50, 100])
ax_e.tick_params(axis="x", labelsize=FS_TICK, length=2.2, pad=2.0)
ax_e.set_xlabel("Taken when offered (%)", fontsize=FS_AX, labelpad=2.0)
for s in ("top", "right", "left"):
    ax_e.spines[s].set_visible(False)
ax_e.spines["bottom"].set_bounds(0, 100)
ax_e_d.set_ylim(Y_BOT_E, Y_TOP_E)
ax_e_d.set_xlim(-16, 128)
ax_e_d.set_yticks([])
ax_e_d.set_xticks([0, 50, 100])
ax_e_d.set_xticklabels(["0", "50", "100"], fontsize=FS_TICK)
ax_e_d.tick_params(axis="x", length=2.2, pad=2.0)
ax_e_d.set_xlabel("Ours −\nAlphaDefect", fontsize=FS_AX, labelpad=2.0, linespacing=1.0)
for s in ("top", "right", "left"):
    ax_e_d.spines[s].set_visible(False)
ax_e_d.spines["bottom"].set_bounds(0, 100)

leg_kw = dict(frameon=False, fontsize=7.0, loc="upper left", ncol=3, handlelength=0.9,
              handletextpad=0.25, columnspacing=0.95, borderaxespad=0.0, borderpad=0.0)
fig.legend(handles=[Line2D([], [], marker="D", ls="", color=SLATE, markersize=4.2, label="Strong humans"),
                    Line2D([], [], marker="o", ls="", color=VIOLET, markersize=4.6, label="AlphaDefect"),
                    Line2D([], [], marker="o", ls="", color=BLUE, markersize=4.6, label="Ours")],
           bbox_to_anchor=(0.635, 0.982), **leg_kw)
fig.legend(handles=[Patch(facecolor=TERRA, edgecolor="none", label="Combination"),
                    Patch(facecolor=BLUE, edgecolor="none", label="Standalone")],
           bbox_to_anchor=(0.635, 0.958), **leg_kw)

for lab, (gx, gy) in zip("abcd", [(0.012, 0.984), (0.318, 0.984), (0.012, 0.448), (0.612, 0.984)]):
    fig.text(gx, gy, lab, fontsize=11, fontweight="bold", va="top", color=INK)

for ext in ("pdf", "svg", "png"):
    fig.savefig(os.path.join(R, "paper", "figs", f"fig5_players.{ext}"), dpi=350)
print("written paper/figs/fig5_players.{pdf,svg,png} at 350 DPI")
