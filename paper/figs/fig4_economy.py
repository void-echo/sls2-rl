"""Figure 4: from a win signal alone, the agent learns an economy of health, gold and cards.

  a, Campfire upgrade share by Act x entering health (heat map) with the per-Act mean beside it.
  b, Implied marginal value of a potion in HP equivalents across Acts 1-3.
  c, Events with an option that costs health: share taking it by health on arrival (same colour scale as a), and
     three events where it makes a near-fixed choice (>94%).
  d, The shop: gold per visit by Act, split into cards, relics, card removal and potions, with the unspent gold.
  e, Most-removed card at shop (% of removals) for each character, with k/n counts.
  f, Relics: the six taken most and six never taken when an ancient offers three, and when affordable in a shop.
  g, Win rate at the Act-3 boss door by decile of the value estimate (Wilson 95% ribbon).

Data: paper/figdata/fresh1123_main_prefs.json (a), paper/figdata/behaviour_v0113.json (c-f; 300 greedy A10 runs per
character on v0.113.1), docs/results/v3_iron/death_fork_a10_u39800.md (b, g; values as in fig4_learned.py).
Game art: tmp/game_assets/fig14_v2/ and tmp/game_assets/fig45_split/ (built by build_assets.py there).
Lettering follows paper/figs/STYLE.md. Writes paper/figs/fig4_economy.{pdf,svg,png} (350 DPI).
"""
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
from matplotlib import font_manager
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIRS = [os.path.join(R, "tmp", "game_assets", "fig45_split"), os.path.join(R, "tmp", "game_assets", "fig14_v2")]
if not os.path.exists(os.path.join(ASSET_DIRS[1], "badge_char_fool.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_DIRS[1], "build_assets.py")], check=True, cwd=R)
if not os.path.exists(os.path.join(ASSET_DIRS[0], "card_strike_regent.png")):
    subprocess.run([sys.executable, os.path.join(ASSET_DIRS[0], "build_assets.py")], check=True, cwd=R)


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
TERRA = "#c85228"
TEAL = "#1b8a6b"
AMBER = "#c98a1b"
PLUM = "#b85682"
VIOLET = "#5e4cc4"
SLATE = "#64748b"
INK = "#14181f"
INK2 = "#4a5568"
MUTED = "#718096"
NAVY = "#253494"     # dark end of the YlGnBu heat scale
OUTER_BORDER = "#e2e8f0"
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]

CHAR_COLORS = {"Ironclad": BLUE, "Defect": TERRA, "Regent": TEAL, "Silent": AMBER, "Necrobinder": PLUM,
               "Fool": VIOLET}
CHAR_BADGES = {c: f"badge_char_{c.lower()}.png" for c in CHAR_COLORS}

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
HEAT_CMAP = "YlGnBu"
HEAT_NORM = Normalize(vmin=0, vmax=96)


def asset(fn):
    for d in ASSET_DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


_SPRITE_CACHE = {}


def load_rgba_clean(fn, target_h_px):
    key = (fn, int(round(target_h_px)))
    if key in _SPRITE_CACHE:
        return _SPRITE_CACHE[key]
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
    out = (np.concatenate([rgb, ra_clean], axis=-1).astype(np.uint8), nh)
    _SPRITE_CACHE[key] = out
    return out


def sprite_aspect(fn):
    with Image.open(asset(fn)) as im:
        return im.width / im.height


def add_sprite(ax, fn, xy, target_h_pt, xycoords="data", xybox=None, align=(0.5, 0.5), zorder=6, clip_on=False,
               alpha=1.0):
    arr, h_px = load_rgba_clean(fn, target_h_pt * 5.0)
    oi = OffsetImage(arr, zoom=target_h_pt / h_px, interpolation="bilinear", resample=True, alpha=alpha)
    if xybox is not None:
        ab = AnnotationBbox(oi, xy, xybox=xybox, xycoords=xycoords, boxcoords="offset points",
                            box_alignment=align, frameon=False, pad=0, zorder=zorder, annotation_clip=clip_on)
    else:
        ab = AnnotationBbox(oi, xy, xycoords=xycoords, box_alignment=align, frameon=False, pad=0, zorder=zorder,
                            annotation_clip=clip_on)
    ab.set_clip_on(clip_on)
    ax.add_artist(ab)


def wilson_pct(p_pct, n, z=1.96):
    p = p_pct / 100.0
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return 100.0 * max(0.0, c - h), 100.0 * min(1.0, c + h)


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.3, pad=2.0)


def wrap_relic_name(raw):
    """Compact 1- or 2-line English relic name from localization, stripped of leading 'The '."""
    s = raw.strip()
    if s.startswith("The "):
        s = s[4:]
    if " " not in s or len(s) <= 8:
        return s
    words = s.split()
    if len(words) == 2:
        return f"{words[0]}\n{words[1]}"
    # Balance 3+ words (e.g. 'Sling of Courage' -> 'Sling of\nCourage')
    best_i, best_w = 1, 999
    for i in range(1, len(words)):
        w = max(len(" ".join(words[:i])), len(" ".join(words[i:])))
        if w < best_w:
            best_i, best_w = i, w
    return " ".join(words[:best_i]) + "\n" + " ".join(words[best_i:])


FIG_W, FIG_H = 8.85, 5.95
PT_W, PT_H = FIG_W * 72.0, FIG_H * 72.0
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")


def pt_axes(rect):
    """Axes whose data units are points of the printed figure (for placing art and cells exactly)."""
    ax = fig.add_axes(rect)
    w, h = rect[2] * PT_W, rect[3] * PT_H
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return ax, w, h


fig.add_artist(FancyBboxPatch(
    (0.006, 0.008), 0.988, 0.984, transform=fig.transFigure,
    boxstyle="round,pad=0.0,rounding_size=0.014",
    facecolor="none", edgecolor=OUTER_BORDER, linewidth=0.8, zorder=0
))

prefs = json.load(open(os.path.join(R, "paper/figdata/fresh1123_main_prefs.json")))
beh = json.load(open(os.path.join(R, "paper/figdata/behaviour_v0113.json")))

# =========================================================================
# Panel a: campfire heat map (Act x entering health) + per-Act mean + colour bar with bed / anvil
# =========================================================================
ax_a = fig.add_axes([0.060, 0.748, 0.142, 0.200])
ax_a_bar = fig.add_axes([0.206, 0.748, 0.054, 0.200])
ax_a_cbar = fig.add_axes([0.083, 0.652, 0.110, 0.016])

hp_bins = [(0.0, 0.40), (0.40, 0.60), (0.60, 0.75), (0.75, 0.90), (0.90, 1.01)]
all_rest = [x for v in prefs.values() for x in v["rest"]]
mat = np.zeros((3, len(hp_bins)))
act_means = []
for act_idx, act_num in enumerate((1, 2, 3)):
    act_sub = [x for x in all_rest if x[1] == act_num]
    act_means.append(100.0 * sum(1 for x in act_sub if x[2] == "SMITH") / max(1, len(act_sub)))
    for b_idx, (lo, hi) in enumerate(hp_bins):
        sub = [x for x in act_sub if lo <= x[0] < hi]
        mat[act_idx, b_idx] = 100.0 * sum(1 for x in sub if x[2] == "SMITH") / max(1, len(sub))

im_a = ax_a.imshow(mat, cmap=HEAT_CMAP, norm=HEAT_NORM, aspect="auto")
ax_a.set_xticks(np.arange(-0.5, len(hp_bins), 1), minor=True)
ax_a.set_yticks(np.arange(-0.5, 3, 1), minor=True)
ax_a.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
ax_a.tick_params(which="minor", bottom=False, left=False)
for i in range(3):
    for j in range(len(hp_bins)):
        val = mat[i, j]
        ax_a.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7.8, fontweight="bold",
                  color="white" if val >= 50 else INK)
ax_a.add_patch(FancyBboxPatch(
    (1.54, -0.46), 0.92, 2.92, boxstyle="round,pad=0.0,rounding_size=0.08",
    facecolor="none", edgecolor=TERRA, linewidth=1.3, linestyle=(0, (3.0, 1.5)), zorder=5
))
ax_a.set_xticks(np.arange(-0.5, len(hp_bins) + 0.5, 1.0))
ax_a.set_xticklabels(["0", "40", "60", "75", "90", "100"], fontsize=FS_TICK)
ax_a.set_yticks([0, 1, 2])
ax_a.set_yticklabels(["Act 1", "Act 2", "Act 3"], fontsize=7.0)
ax_a.set_xlabel("Health at the rest site (%)", fontsize=FS_AX, labelpad=2.0)
ax_a.tick_params(length=2.0, pad=1.8)
ax_a.tick_params(axis="y", length=0, pad=2.5)
for s in ax_a.spines.values():
    s.set_edgecolor(INK2)
    s.set_linewidth(0.55)

ax_a_bar.set_ylim(2.5, -0.5)
ax_a_bar.set_xlim(0, 100)
for i_act, mval in enumerate(act_means):
    ax_a_bar.barh(i_act, mval, height=0.62, color=BLUE_MID, edgecolor="white", linewidth=0.5, zorder=2)
    ax_a_bar.text(mval + 4.0, i_act, f"{mval:.0f}", va="center", ha="left",
                  fontsize=FS_LAB, fontweight="bold", color=BLUE, zorder=4)
for s in ("top", "right", "bottom"):
    ax_a_bar.spines[s].set_visible(False)
ax_a_bar.spines["left"].set_color(INK2)
ax_a_bar.spines["left"].set_linewidth(0.55)
ax_a_bar.set_yticks([])
ax_a_bar.set_xticks([])
ax_a_bar.text(0.0, 1.03, "Act mean", transform=ax_a_bar.transAxes, fontsize=7.0, color=INK, ha="left", va="bottom")

cb_a = fig.colorbar(im_a, cax=ax_a_cbar, orientation="horizontal")
cb_a.set_ticks([])
cb_a.outline.set_linewidth(0.5)
cb_a.outline.set_edgecolor(INK2)
cb_a.set_label("Upgrade (%)", fontsize=FS_AX, labelpad=3.0)
add_sprite(ax_a_cbar, "rest_heal_badge.png", (0.0, 0.5), target_h_pt=17.0,
           xycoords="axes fraction", xybox=(-4.0, 0), align=(1.0, 0.5), zorder=7)
add_sprite(ax_a_cbar, "rest_smith_badge.png", (1.0, 0.5), target_h_pt=17.0,
           xycoords="axes fraction", xybox=(4.0, 0), align=(0.0, 0.5), zorder=7)

# =========================================================================
# Panel b: implied marginal value of a potion (HP equivalents) across Acts 1-3
# =========================================================================
ax_b = fig.add_axes([0.322, 0.668, 0.084, 0.282])
clean_ax(ax_b)
for gy in (5, 10, 15):
    ax_b.axhline(gy, color="#edf2f7", lw=0.55, zorder=0)
# slopes of the value change on potions and on HP (docs/results/v3_iron/death_fork_a10_u39800.md)
pot_hp = [0.0097 / 0.0015, 0.0093 / 0.0010, 0.0148 / 0.0014]  # [6.467, 9.300, 10.571]
pot_icons = ["potion_blood.png", "potion_regen.png", "potion_attack.png"]
pot_cols = [SLATE, BLUE, TERRA]
pot_fills = ["#718399", BLUE_MID, TERRA]
ax_b.bar([0, 1, 2], pot_hp, width=0.62, color=pot_fills, edgecolor="white", linewidth=0.65, zorder=3)
for idx_b, (val, p_icon, pcol) in enumerate(zip(pot_hp, pot_icons, pot_cols)):
    add_sprite(ax_b, p_icon, (idx_b, val + 0.25), target_h_pt=16.0, xycoords="data", align=(0.5, 0.0), zorder=6)
    ax_b.text(idx_b, val + 3.05, f"{val:.1f}", ha="center", va="bottom",
              fontsize=FS_LAB, fontweight="bold", color=pcol, path_effects=HALO, zorder=7)
premium = 100.0 * (pot_hp[2] / pot_hp[0] - 1.0)
ax_b.annotate("", xy=(2.0, 17.6), xytext=(-0.1, 12.6),
              arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.36", mutation_scale=7.0,
                              lw=1.1, color=TERRA, connectionstyle="arc3,rad=-0.20"), zorder=5)
ax_b.text(0.50, 18.6, f"+{premium:.0f}%", fontsize=8.6, fontweight="bold", color=TERRA,
          ha="center", va="center", zorder=7)
ax_b.set_xlim(-0.52, 2.52)
ax_b.set_ylim(0, 21.0)
ax_b.set_yticks([0, 5, 10, 15, 20])
ax_b.set_xticks([0, 1, 2])
ax_b.set_xticklabels(["Act 1", "Act 2", "Act 3"], fontsize=FS_TICK)
ax_b.tick_params(axis="y", labelsize=FS_TICK)
ax_b.tick_params(axis="x", length=0, pad=3.0)
ax_b.set_ylabel("Potion value (HP)", fontsize=FS_AX, labelpad=2.0)

# =========================================================================
# Panel c: events that cost health, by health on arrival; and three near-fixed choices (>94%)
# =========================================================================
ax_c, CW_, CH_ = pt_axes([0.432, 0.622, 0.560, 0.362])
EV_HP = [
    ("ROOM_FULL_OF_CHEESE|INITIAL|SEARCH", "Room Full\nof Cheese", "event_room_full_of_cheese.png", "shop_relic.png"),
    ("BRAIN_LEECH|INITIAL|RIP", "Brain Leech", "event_brain_leech.png", "shop_card.png"),
    ("SUNKEN_STATUE|INITIAL|DIVE_INTO_WATER", "Sunken\nStatue", "event_sunken_statue.png", "icon_gold.png"),
]
EV_FIX = [
    ("THIS_OR_THAT|INITIAL|ORNATE", "This or\nThat?", "event_this_or_that.png", "shop_relic.png"),
    ("POTION_COURIER|INITIAL|RANSACK", "Potion\nCourier", "event_potion_courier.png", "shop_potion.png"),
    ("ROUND_TEA_PARTY|INITIAL|PICK_FIGHT", "Round Tea\nParty", "event_round_tea_party.png", "shop_relic.png"),
]
N_SMALL = 40           # bins with fewer offers are drawn faded
RH = 36.0              # row height (pt)
ART_H = RH - 3.0
ART_W = ART_H * 4.0 / 3.0
TOP = CH_ - 15.0       # top of the first row
cmap_heat = plt.get_cmap(HEAT_CMAP)


def row_y(i):
    return TOP - RH * (i + 0.5)


def event_row(x_name, x_art, i, label, art, reward):
    y = row_y(i)
    ax_c.text(x_name, y, label, ha="right", va="center", fontsize=7.0, color=INK, linespacing=1.0)
    add_sprite(ax_c, art, (x_art, y), target_h_pt=ART_H, align=(0.0, 0.5), zorder=5)
    add_sprite(ax_c, reward, (x_art + ART_W + 3.0, y - ART_H / 2.0 - 1.5), target_h_pt=15.0,
               align=(1.0, 0.0), zorder=7)


def cell(x0, y, w, rate, txt, faded=False, fs=7.8):
    col = cmap_heat(HEAT_NORM(rate))
    ax_c.add_patch(Rectangle((x0 + 0.8, y - RH / 2.0 + 0.8), w - 1.6, RH - 1.6, facecolor=col,
                             edgecolor="none", alpha=0.38 if faded else 1.0, zorder=3))
    tcol = MUTED if faded else ("white" if rate >= 50 else INK)
    ax_c.text(x0 + w / 2.0, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold", color=tcol, zorder=4)


# block 1: lose HP for a reward, by health on arrival
XN1, XA1 = 47.0, 50.0
XC1 = XA1 + ART_W + 6.0
CWD = 36.0
for i, (key, label, art, reward) in enumerate(EV_HP):
    ev = beh["events"][key]
    event_row(XN1, XA1, i, label, art, reward)
    for j, (n_off, n_ch) in enumerate(ev["by_hp_below50_50to75_above75"]):
        rate = 100.0 * n_ch / n_off
        cell(XC1 + j * CWD, row_y(i), CWD, rate, f"{rate:.0f}", faded=n_off < N_SMALL)
y_tick = row_y(2) - RH / 2.0 - 6.0
for j, t in enumerate(["<50", "50–75", ">75"]):
    ax_c.text(XC1 + (j + 0.5) * CWD, y_tick, t, ha="center", va="center", fontsize=FS_TICK, color=INK2)
ax_c.text(XC1 + 1.5 * CWD + 6.0, y_tick - 11.0, "Health on arrival (%)", ha="center", va="center",
          fontsize=FS_AX, color=INK)
add_sprite(ax_c, "icon_heart.png", (XC1 + 1.5 * CWD - 41.0, y_tick - 11.0), target_h_pt=10.0, zorder=6)
ax_c.text(XA1, CH_ - 6.0, "Lose HP for a reward", ha="left", va="center", fontsize=FS_LAB, fontweight="bold",
          color=BLUE)

# block 2: near-fixed choice (>94%)
XN2 = XC1 + 3 * CWD + 47.0
XA2 = XN2 + 3.0
XC2 = XA2 + ART_W + 6.0
CW2 = CW_ - XC2 - 1.0
for i, (key, label, art, reward) in enumerate(EV_FIX):
    ev = beh["events"][key]
    event_row(XN2, XA2, i, label, art, reward)
    rate = 100.0 * ev["chosen"] / ev["offered"]
    cell(XC2, row_y(i), CW2, rate, f"{ev['chosen']}/{ev['offered']}", fs=7.4)
ax_c.text(XC2 + CW2 / 2.0, y_tick, "taken / offered", ha="center", va="center", fontsize=FS_TICK, color=INK2)
ax_c.text(XA2 - 3.0, CH_ - 6.0, "Near-fixed choice (>94%)", ha="left", va="center", fontsize=FS_LAB, fontweight="bold",
          color=NAVY)

# =========================================================================
# Panel d: the shop, gold per visit by Act
# =========================================================================
ax_d = fig.add_axes([0.066, 0.378, 0.318, 0.152])
SHOP_KINDS = [("buy_card", "Cards", "shop_card.png", BLUE_MID), ("buy_relic", "Relics", "shop_relic.png", AMBER),
              ("remove", "Removal", "shop_remove.png", TERRA), ("buy_potion", "Potions", "shop_potion.png", TEAL)]
pooled = beh["shop_pooled_by_act"]
for gx in (100, 200, 300):
    ax_d.axvline(gx, color="#edf2f7", lw=0.55, zorder=0)
for ia, act in enumerate(("1", "2", "3")):
    row = pooled[act]
    x = 0.0
    for kind, _, _, col in SHOP_KINDS:
        w = row["spend_per_visit_by_kind"][kind]
        ax_d.barh(ia, w, left=x, height=0.66, color=col, edgecolor="white", linewidth=0.6, zorder=3)
        x += w
    spent, entry = row["spent_per_visit"], row["gold_at_entry"]
    ax_d.barh(ia, entry - spent, left=spent, height=0.66, color="#eef2f7", edgecolor="#c7d2df",
              linewidth=0.6, linestyle=(0, (2, 1.2)), zorder=2)
    ax_d.text(spent + 3.0, ia, f"{spent:.0f}", ha="left", va="center", fontsize=FS_LAB, fontweight="bold",
              color=INK, zorder=5)
    ax_d.text(entry + 3.0, ia, f"{entry:.0f}", ha="left", va="center", fontsize=FS_LAB, color=MUTED, zorder=5)
ax_d.set_ylim(2.55, -0.55)
ax_d.set_xlim(0, 342)
ax_d.set_yticks([0, 1, 2])
ax_d.set_yticklabels(["Act 1", "Act 2", "Act 3"], fontsize=7.0)
ax_d.set_xticks([0, 100, 200, 300])
ax_d.tick_params(axis="x", labelsize=FS_TICK, length=2.2, pad=2.0)
ax_d.tick_params(axis="y", length=0, pad=2.5)
for s in ("top", "right"):
    ax_d.spines[s].set_visible(False)
ax_d.spines["left"].set_visible(False)
ax_d.set_xlabel("Gold per shop visit", fontsize=FS_AX, labelpad=2.0)
add_sprite(ax_d, "icon_gold.png", (0.5, -0.30), target_h_pt=10.0, xycoords="axes fraction", xybox=(-46.0, 0.0),
           zorder=6)

# legend: the four things a shop sells, with their icons
ax_dl, DLW, DLH = pt_axes([0.020, 0.540, 0.385, 0.052])
renderer = fig.canvas.get_renderer()


def text_w_pt(t):
    return t.get_window_extent(renderer).width * 72.0 / fig.dpi


xl = 8.0
for kind, name, icon, col in SHOP_KINDS:
    add_sprite(ax_dl, icon, (xl, DLH / 2.0), target_h_pt=17.0, align=(0.5, 0.5), zorder=6)
    t = ax_dl.text(xl + 10.5, DLH / 2.0, name, ha="left", va="center", fontsize=FS_LAB, fontweight="bold", color=col)
    xl += 10.5 + text_w_pt(t) + 16.0
ax_dl.add_patch(Rectangle((xl - 6.0, DLH / 2.0 - 5.0), 12.0, 10.0, facecolor="#eef2f7", edgecolor="#c7d2df",
                          linewidth=0.6, linestyle=(0, (2, 1.2))))
ax_dl.text(xl + 9.0, DLH / 2.0, "Kept", ha="left", va="center", fontsize=FS_LAB, fontweight="bold", color=MUTED)

# =========================================================================
# Panel e: most-removed card at shop (% of removals) for each character
# =========================================================================
ax_e, EW, EH = pt_axes([0.428, 0.312, 0.564, 0.286])
E_CHARS = [("Ironclad", "card_defend_ironclad.png"), ("Silent", "card_strike_silent.png"),
           ("Defect", "card_dualcast.png"), ("Regent", "card_strike_regent.png"),
           ("Necrobinder", "card_strike_necrobinder.png"), ("Fool", "card_strike_fool.png")]
card_names = json.load(open(os.path.join(R, "engine/localization_eng/cards.json")))
E_MARG_L, E_MARG_R = 12.0, 4.0
pitch_e = (EW - E_MARG_L - E_MARG_R) / len(E_CHARS)
CARD_H = 73.0
card_w = CARD_H * sprite_aspect(E_CHARS[0][1])

add_sprite(ax_e, "shop_remove.png", (16.0, EH - 7.0), target_h_pt=13.5, align=(0.5, 0.5), zorder=6)
ax_e.text(26.0, EH - 7.0, "Most-removed card at shop (% of removals)", ha="left", va="center",
          fontsize=7.4, fontweight="bold", color=INK)

for k, (ch, card_png) in enumerate(E_CHARS):
    rem = beh["removed"][ch]
    top_id = max(rem, key=rem.get)
    tot_rem = sum(rem.values())
    share = 100.0 * rem[top_id] / tot_rem
    cx = E_MARG_L + pitch_e * (k + 0.5) - 4.0
    y_card0 = 30.5
    add_sprite(ax_e, card_png, (cx + 4.0, y_card0), target_h_pt=CARD_H, align=(0.5, 0.0), zorder=5)
    add_sprite(ax_e, CHAR_BADGES[ch], (cx + 4.0 - card_w / 2.0 - 1.0, y_card0 + CARD_H - 4.0), target_h_pt=18.5,
               align=(0.5, 0.5), zorder=7)
    name = card_names[top_id.split(".", 1)[1] + ".title"]
    ax_e.text(cx + 4.0, 22.0, name, ha="center", va="center", fontsize=7.0, color=INK)
    ax_e.text(cx + 4.0, 11.8, f"{share:.0f}%", ha="center", va="center", fontsize=8.2,
              fontweight="bold", color=CHAR_COLORS[ch])
    ax_e.text(cx + 4.0, 2.5, f"{rem[top_id]}/{tot_rem}", ha="center", va="center", fontsize=6.6, color=INK2)
    print(f"e {ch}: {name} {share:.1f}% ({rem[top_id]}/{tot_rem})")

# =========================================================================
# Panel f: relics it always / never takes (ancient's choice of three; shop when affordable)
# =========================================================================
ax_f, FW, FH = pt_axes([0.012, 0.012, 0.802, 0.292])
N_END = 6
MIN_N = 50


def ranked(table, k_key, n_key):
    rows = [(v[k_key] / v[n_key], v[n_key], rid) for rid, v in table.items() if v[n_key] >= MIN_N]
    top = sorted(rows, key=lambda r: (-r[0], -r[1]))[:N_END]
    never = sorted([r for r in rows if r[0] == 0.0], key=lambda r: -r[1])[:N_END]
    return top, never


relic_names = json.load(open(os.path.join(R, "engine/localization_eng/relics.json")))
strips = [("Ancient", "node_ancient.png", ranked(beh["relics_event"], "chosen", "offered")),
          ("Shop", "merchant.png", ranked(beh["relics_shop"], "bought", "affordable"))]
X0F = 44.0
GAP_F = 16.0
pitch_f = (FW - X0F - GAP_F - 2.0) / (2 * N_END)
ICON_F = 24.5
ROW_Y_ICON = [97.5, 42.0]
for (lab, node_icon, (top, never)), y_ic in zip(strips, ROW_Y_ICON):
    add_sprite(ax_f, node_icon, (21.0, y_ic - 2.0), target_h_pt=21.5, zorder=6)
    ax_f.text(21.0, y_ic - 21.0, lab, ha="center", va="center", fontsize=7.0, color=INK)
    for gi, group in enumerate((top, never)):
        for m, (rate, n, rid) in enumerate(group):
            xc = X0F + pitch_f * (m + 0.5) + gi * (N_END * pitch_f + GAP_F)
            icon = f"relic_{rid.split('.', 1)[1].lower()}.png"
            add_sprite(ax_f, icon, (xc, y_ic), target_h_pt=min(ICON_F, (pitch_f - 4.0) / sprite_aspect(icon)),
                       zorder=6)
            ax_f.text(xc, y_ic - 17.0, f"{100.0 * rate:.0f}%", ha="center", va="center", fontsize=6.9,
                      fontweight="bold", color=BLUE if rate >= 0.5 else (TERRA if rate == 0 else SLATE))
            rtitle = wrap_relic_name(relic_names[rid.split(".", 1)[1] + ".title"])
            ax_f.text(xc, y_ic - 22.0, rtitle, ha="center", va="top", fontsize=6.5, color=INK, linespacing=0.90)

for lab, _, groups in strips:   # names for log verification, in the order drawn
    for gname, group in zip(("most", "never"), groups):
        print(f"f {lab} {gname}:", ", ".join(f"{relic_names[r[2].split('.', 1)[1] + '.title']} {100 * r[0]:.0f}% "
                                                f"(n={r[1]})" for r in group))
x_mid = X0F + N_END * pitch_f + GAP_F / 2.0
ax_f.plot([x_mid, x_mid], [5.5, FH - 13.5], color="#cbd5e1", lw=0.8, ls=(0, (2, 2)))
ax_f.plot([X0F + 2.0, FW - 2.0], [58.8, 58.8], color="#e2e8f0", lw=0.7)
ax_f.text(X0F + 2.0, FH - 6.5, "Taken most", ha="left", va="center", fontsize=FS_LAB, fontweight="bold",
          color=BLUE)
ax_f.text(FW - 2.0, FH - 6.5, "Never taken", ha="right", va="center", fontsize=FS_LAB, fontweight="bold",
          color=TERRA)

# =========================================================================
# Panel g: value estimate against the actual win rate at the Act-3 boss door
# =========================================================================
ax_g = fig.add_axes([0.868, 0.082, 0.118, 0.205])
clean_ax(ax_g)
for gy in (40, 60, 80, 100):
    ax_g.axhline(gy, color="#edf2f7", lw=0.55, zorder=0)
# Exact decile counts & win rates from docs/results/v3_iron/death_fork_a10_u39800.md
dec_x = np.arange(1, 11)
dec_n = [71, 70, 70, 70, 70, 71, 70, 70, 70, 71]
dec_wr = [35.21, 77.14, 85.71, 94.29, 100.0, 97.18, 100.0, 100.0, 100.0, 100.0]
dec_lo = [wilson_pct(w, n)[0] for w, n in zip(dec_wr, dec_n)]
dec_hi = [wilson_pct(w, n)[1] for w, n in zip(dec_wr, dec_n)]
MEAN_WR = 88.90   # 625 / 703 entries
AUC = 0.93
ax_g.axhline(MEAN_WR, color=SLATE, lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
ax_g.fill_between(dec_x, dec_lo, dec_hi, color=BLUE, alpha=0.16, lw=0, zorder=2)
ax_g.plot(dec_x, dec_wr, "o-", color=BLUE, ms=3.6, mec="white", mew=0.5, lw=1.4, zorder=4)
ax_g.scatter([1], [dec_wr[0]], s=28, color=TERRA, edgecolor="white", linewidth=0.7, zorder=5)
ax_g.text(1.7, dec_wr[0] - 2.5, f"{dec_wr[0]:.0f}%", fontsize=FS_LAB, fontweight="bold", color=TERRA,
          ha="left", va="center", path_effects=HALO, zorder=6)
ax_g.text(6.9, 70.0, f"AUC {AUC:.2f}", fontsize=8.0, fontweight="bold", color=BLUE, ha="center", va="center",
          zorder=4)
for bx, b_icon in ((4.6, "boss_queen.png"), (6.9, "boss_test_subject.png"), (9.2, "boss_aeonglass.png")):
    add_sprite(ax_g, b_icon, (bx, 48.0), target_h_pt=14.0, xycoords="data", zorder=6)
ax_g.set_xticks([1, 4, 7, 10])
ax_g.set_xticklabels(["D1", "D4", "D7", "D10"], fontsize=FS_TICK)
ax_g.set_yticks([40, 60, 80, 100])
ax_g.tick_params(axis="y", labelsize=FS_TICK)
ax_g.set_ylim(21, 106.5)
ax_g.set_xlim(0.42, 10.62)
ax_g.set_xlabel("Value estimate (decile)", fontsize=FS_AX, labelpad=2.0)
ax_g.set_ylabel("Boss win rate (%)", fontsize=FS_AX, labelpad=2.0)

# Panel letters
for lab, (gx, gy) in zip("abcdefg", [(0.012, 0.984), (0.272, 0.984), (0.415, 0.984), (0.012, 0.600),
                                     (0.415, 0.600), (0.012, 0.305), (0.818, 0.305)]):
    fig.text(gx, gy, lab, fontsize=11, fontweight="bold", va="top", color=INK)

for ext in ("pdf", "svg", "png"):
    fig.savefig(os.path.join(R, "paper", "figs", f"fig4_economy.{ext}"), dpi=350)
print("written paper/figs/fig4_economy.{pdf,svg,png} at 350 DPI")
