"""Figure 1: one connected schematic of the world, the agent and the updates (no results).

a  The game: six characters, one run (three acts as real map nodes, the final gate with the three final bosses),
   the combat scene and the decisions a run is made of (card reward, rest site, shop, relics and potions).
b  The agent: the visible state as an unordered set of hashed key-value entity tokens, a four-layer set transformer,
   a value (probability of winning the run) and a pointer over the legal actions; the chosen action goes back into the game.
c  Updates become experiments: the version timeline of the real game and what an update does, then the paired
   experiment (same network, same seeds, old and new version, one change switched back and the games replayed).

Art is decoded from the game package by tmp/game_assets/fig1_world/build_assets.py and extra_assets.py,
plus v0.114 Fool cards in tmp/game_assets/fig3_restyle/.
Writes paper/figs/fig1_world.{png,pdf,svg} (png at 350 dpi).
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Wedge

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(R, "tmp", "game_assets", "fig1_world")
X = os.path.join(A, "extra")
BADGES = os.path.join(R, "tmp", "game_assets", "fig14_v2")
FOOL_CARDS = os.path.join(R, "tmp", "game_assets", "fig3_restyle")
if not os.path.exists(os.path.join(A, "combat_scene.png")):
    subprocess.run([sys.executable, os.path.join(A, "build_assets.py")], check=True, cwd=R)
if not os.path.exists(os.path.join(X, "merchant_sprite.png")):
    subprocess.run([sys.executable, os.path.join(A, "extra_assets.py")], check=True, cwd=R)


def register_helvetica_bold():
    """matplotlib reads only the first face of Helvetica.ttc, so bold lettering would fall back to regular."""
    dst = os.path.join(A, "fonts", "Helvetica-Bold.ttf")
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

# palette (STYLE.md)
BLUE = "#2166ac"
BLUE_SOFT = "#eef4fb"
BLUE_PILL = "#d6e6f5"
TERRA = "#c85228"
TERRA_SOFT = "#fcf3ef"
TERRA_PILL = "#fae6dd"
TEAL = "#1b8a6b"
AMBER = "#c98a1b"
VIOLET = "#5e4cc4"
PLUM = "#b85682"
INK = "#1f2937"
INK2 = "#475569"
GREY = "#94a3b8"
EDGE = "#9bb3ce"
GAME_BG = "#f3f5f8"
OUTER_BORDER = "#e2e8f0"
CHAR_COLORS = {"Ironclad": BLUE, "Defect": TERRA, "Regent": TEAL, "Silent": AMBER, "Necrobinder": PLUM, "Fool": VIOLET}

plt.rcParams.update({"font.family": "Helvetica", "pdf.fonttype": 42, "svg.fonttype": "none"})

W_IN, H_IN = 8.5, 6.0
U = 100.0                 # drawing units per inch (1 unit = 0.72 pt)
PX_PER_U = 7.0            # embedded art at 700 px per inch
fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

_cache = {}


def path_of(name):
    for d in (A, X, BADGES, FOOL_CARDS):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def load(name):
    if name not in _cache:
        _cache[name] = Image.open(path_of(name)).convert("RGBA")
    return _cache[name]


def halo(pim, grow=10, blur=9, strength=0.85):
    """A soft white glow around a sprite, so dark art stands out on a dark backdrop."""
    pad = grow + 2 * blur
    big = Image.new("RGBA", (pim.width + 2 * pad, pim.height + 2 * pad), (0, 0, 0, 0))
    a = Image.new("L", big.size, 0)
    a.paste(pim.split()[3], (pad, pad))
    a = a.filter(ImageFilter.MaxFilter(2 * (grow // 2) + 1)).filter(ImageFilter.GaussianBlur(blur))
    a = a.point(lambda v: int(v * strength))
    glow = Image.new("RGBA", big.size, (255, 255, 255, 0))
    glow.putalpha(a)
    glow.alpha_composite(pim, (pad, pad))
    return glow


def put(name, cx, cy, h=None, w=None, rot=0.0, z=5, crop=None, enhance=None, anchor="c", clip=None, glow=None):
    """Place an image at (cx, cy) with height h (or width w) in drawing units; returns its extent."""
    im = load(name)
    if crop is not None:
        im = im.crop(crop)
    asp = im.width / im.height
    if h is None:
        h = w / asp
    w = h * asp
    wp, hp = max(2, int(round(w * PX_PER_U))), max(2, int(round(h * PX_PER_U)))
    pim = im.convert("RGBa").resize((wp, hp), Image.Resampling.LANCZOS)
    if rot:
        pim = pim.rotate(rot, expand=True, resample=Image.Resampling.BICUBIC)
    pim = pim.convert("RGBA")
    if enhance is not None:
        pim = enhance(pim)
    off_x = off_y = 0.0
    if glow is not None:
        g = halo(pim, **glow)
        off_x, off_y = (g.width - pim.width) / 2 / PX_PER_U, (g.height - pim.height) / 2 / PX_PER_U
        pim = g
    ew, eh = pim.width / PX_PER_U, pim.height / PX_PER_U
    if anchor == "c":
        x0, y0 = cx - ew / 2, cy - eh / 2
    elif anchor == "b":          # cy is the bottom edge of the art itself
        x0, y0 = cx - ew / 2, cy - off_y
    else:
        raise ValueError(anchor)
    art = ax.imshow(np.asarray(pim), extent=[x0, x0 + ew, y0, y0 + eh], zorder=z, interpolation="antialiased",
                    aspect="auto")
    if clip is not None:
        art.set_clip_path(clip)
    return (x0 + off_x, x0 + ew - off_x, y0 + off_y, y0 + eh - off_y)


def txt(x, y, s, size=7.0, color=INK, weight="bold", ha="center", va="center", z=20, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, **kw)


def rbox(x0, y0, w, h, fc="white", ec="none", lw=0.6, r=6.0, z=1, ls="-"):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec,
                       linewidth=lw, zorder=z, linestyle=ls)
    ax.add_patch(p)
    return p


def arrow(p0, p1, color=INK2, lw=1.0, z=15, rad=0.0, head=6.0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head, color=color, linewidth=lw, zorder=z,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0,
                        capstyle="round", joinstyle="round")
    ax.add_patch(a)
    return a


def poly_arrow(pts, color=BLUE, lw=1.4, z=15, head=7.0):
    """A polyline with an arrow head on its last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round")
    arrow(pts[-2], pts[-1], color=color, lw=lw, z=z, head=head)


def text_width(s, size, weight="bold"):
    """Width of a string in drawing units."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    tp = TextPath((0, 0), s, size=size, prop=FontProperties(family="Helvetica", weight=weight))
    return tp.get_extents().width * U / 72.0


def pill(x, y, s, fc, ec, tc, size=7.0, padx=4.5, h=12.5, z=18):
    w = text_width(s, size) + 2 * padx
    rbox(x - w / 2, y - h / 2, w, h, fc=fc, ec=ec, lw=0.8, r=h / 2, z=z)
    txt(x, y, s, size=size, color=tc, z=z + 1)
    return w


def letter(x, y, s, title):
    txt(x, y, s, size=11, color=INK, ha="left", va="center")
    txt(x + 12, y - 0.3, title, size=8.5, color=INK, ha="left", va="center")


def mini_net(x0, y0, w, h, z=6):
    """A small three-column network glyph in a rounded box (the same network as in b)."""
    rbox(x0, y0, w, h, fc="white", ec=BLUE, lw=0.9, r=5, z=z)
    cols = [x0 + w * f for f in (0.24, 0.5, 0.76)]
    rows = [y0 + h * f for f in (0.2, 0.4, 0.6, 0.8)]
    for li, lx in enumerate(cols):
        for k, ry in enumerate(rows):
            ax.add_patch(Circle((lx, ry), 2.6, facecolor="white", edgecolor=BLUE, linewidth=0.8, zorder=z + 2))
            if li < 2:
                for ry2 in rows:
                    ax.plot([lx, cols[li + 1]], [ry, ry2], color=EDGE, lw=0.35, zorder=z + 1)


# ---------------------------------------------------------------- frame and orthogonal zones (uniform 6-unit gutters)
fig.add_artist(FancyBboxPatch((0.006, 0.008), 0.988, 0.984, transform=fig.transFigure,
                              boxstyle="round,pad=0.0,rounding_size=0.015", facecolor="none",
                              edgecolor=OUTER_BORDER, linewidth=0.8, zorder=0))

# Top band (a): y = 446..590 (h=144)
# Gutter: y = 440..446 (6 units)
# Middle-Left (part of a): x = 8..498, y = 236..440 (h=204)
# Vertical gutter: x = 498..504 (6 units)
# Middle-Right (b): x = 504..842, y = 236..440 (h=204)
# Gutter: y = 230..236 (6 units)
# Bottom band (c): x = 8..842, y = 10..230 (h=220)
rbox(8, 446, 834, 144, fc=GAME_BG, ec="#e1e7f0", lw=0.55, r=8, z=0.2)
rbox(8, 236, 490, 204, fc=GAME_BG, ec="#e1e7f0", lw=0.55, r=8, z=0.2)
rbox(504, 236, 338, 204, fc=BLUE_SOFT, ec="#d4e3f4", lw=0.55, r=8, z=0.2)
rbox(8, 10, 834, 220, fc=TERRA_SOFT, ec="#f1ddd4", lw=0.55, r=8, z=0.2)

letter(16, 579, "a", "The game")
letter(512, 428, "b", "The agent")
letter(16, 217, "c", "Updates become experiments")

# ================================================================= a: characters
CHARS = ["Ironclad", "Silent", "Regent", "Defect", "Necrobinder", "Fool"]
HC = 95.0
widths = [HC * load(f"char_{c.lower()}.png").width / load(f"char_{c.lower()}.png").height for c in CHARS]
x_lo, x_hi = 18.0, 404.0
gap = (x_hi - x_lo - sum(widths)) / (len(CHARS) - 1)
xc = x_lo
char_x = {}
for c, w in zip(CHARS, widths):
    char_x[c] = xc + w / 2
    put(f"char_{c.lower()}.png", xc + w / 2, 464, h=HC, anchor="b", z=4)
    xc += w + gap
for c in CHARS:
    txt(char_x[c], 455, c, size=7.5, color=CHAR_COLORS[c])
pill(char_x["Fool"] + 2, 573, "new in v0.113", fc="white", ec=VIOLET, tc=VIOLET, size=7.0, h=12.5)

# ================================================================= a: three acts drawn with the real map nodes
ACT_X0, ACT_W, ACT_GAP = 418.0, 62.0, 9.0
ACT_Y0, ACT_H = 452.0, 132.0
acts = [
    ("mapbg_overgrowth.png", ["map_monster.png", "map_unknown.png", "map_rest.png", "map_elite.png"]),
    ("mapbg_hive.png", ["map_monster.png", "map_shop.png", "map_elite.png", "map_chest.png"]),
    ("mapbg_glory.png", ["map_unknown.png", "map_monster.png", "map_rest.png", "map_elite.png"]),
]
for i, (bg, nodes) in enumerate(acts):
    x0 = ACT_X0 + i * (ACT_W + ACT_GAP)
    clip = rbox(x0, ACT_Y0, ACT_W, ACT_H, fc="none", ec="#8a7350", lw=0.7, r=5, z=3.5)
    im = load(bg)
    cw = int(im.height * ACT_W / ACT_H)
    put(bg, x0 + ACT_W / 2, ACT_Y0 + ACT_H / 2, h=ACT_H, z=2, clip=clip,
        crop=(int(im.width * 0.3), 0, int(im.width * 0.3) + cw, im.height))
    txt(x0 + ACT_W / 2, ACT_Y0 + ACT_H - 10, f"Act {i + 1}", size=7.5, color="#3b2f22")
    pts = []
    for j, nd in enumerate(nodes):
        pts.append((x0 + ACT_W / 2 + (-13 if j % 2 == 0 else 13), ACT_Y0 + 17 + j * 27.0))
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="#5b4630", lw=1.2, ls=(0, (1.6, 1.6)), zorder=5)
    for (nx, ny), nd in zip(pts, nodes):
        put(nd, nx, ny, h=25.5, z=6)
    if i < 2:
        arrow((x0 + ACT_W + 0.5, ACT_Y0 + ACT_H / 2), (x0 + ACT_W + ACT_GAP - 0.5, ACT_Y0 + ACT_H / 2),
              color=INK, lw=1.1, head=6)

# ================================================================= a: final gate with the three final bosses
BX0, BY0, BW, BH = 634.0, 452.0, 202.0, 132.0
arrow((ACT_X0 + 3 * ACT_W + 2 * ACT_GAP + 0.5, ACT_Y0 + ACT_H / 2), (BX0 - 0.5, ACT_Y0 + ACT_H / 2),
      color=INK, lw=1.1, head=6)
gate = rbox(BX0, BY0, BW, BH, fc="none", ec="#2d2440", lw=0.9, r=6, z=3.5)
bgq = load("bg_queen_room.png")
cw = int(bgq.height * BW / BH)
put("bg_queen_room.png", BX0 + BW / 2, BY0 + BH / 2, h=BH, z=2, clip=gate,
    crop=((bgq.width - cw) // 2, 0, (bgq.width - cw) // 2 + cw, bgq.height),
    enhance=lambda p: ImageEnhance.Brightness(ImageEnhance.Color(p).enhance(0.55)).enhance(0.62))
txt(BX0 + BW / 2, BY0 + BH - 10, "Final bosses: two in a row", size=7.5, color="white")
GL = dict(grow=8, blur=10, strength=0.55)
put("boss_queen.png", BX0 + 42, BY0 + 4, h=104, anchor="b", z=5, glow=GL)
put("boss_aeonglass.png", BX0 + BW - 32, BY0 + 7, h=94, anchor="b", z=5, glow=GL)
put("boss_test_subject.png", BX0 + BW / 2 + 4, BY0 + 3, h=65, anchor="b", z=6, glow=GL)


# ================================================================= a: the decisions a run is made of
def vignette(x0, y0, w, h, label, node=None):
    rbox(x0, y0, w, h, fc="white", ec=OUTER_BORDER, lw=0.65, r=6, z=1)
    tw = text_width(label, 7.0)
    off = 7.5 if node else 0.0
    txt(x0 + w / 2 + off, y0 + 8.5, label, size=7.0, color=INK)
    if node:
        put(node, x0 + w / 2 + off - tw / 2 - 9, y0 + 9.0, h=13, z=8)


vignette(14, 341, 130, 93, "Card reward", node="map_monster.png")
for cx, rot, name in ((46, 8, "card_inflame.png"), (112, -8, "card_demon_form.png"), (79, 0, "card_true_grit.png")):
    put(name, cx, 394, h=68, rot=rot, z=5 if rot == 0 else 4)

vignette(150, 341, 96, 93, "Rest site", node="map_rest.png")
put("rest_heal.png", 198, 409, h=33, z=4)
put("rest_smith.png", 198, 372, h=33, z=4)

vignette(14, 242, 130, 93, "Shop", node="map_shop.png")
put("merchant_sprite.png", 45, 255, h=64, anchor="b", z=4)
for i, (name, hh) in enumerate((("card_bodyguard.png", 41), ("relic_kunai.png", 20), ("potion_regen.png", 21))):
    cx = 92 + i * 21
    put(name, cx, 300 if i == 0 else 293, h=hh, z=5)
    put("icon_gold.png", cx, 268, h=10, z=5)

vignette(150, 242, 96, 93, "Relics · potions")
for i, name in enumerate(["relic_burning_blood.png", "relic_vajra.png", "relic_anchor.png", "relic_lantern.png"]):
    put(name, 169 + i * 19.5, 306, h=21, z=5)
for i, name in enumerate(["potion_blood.png", "potion_energy.png", "potion_fire.png"]):
    put(name, 178 + i * 20, 276, h=21, z=5)

# ================================================================= a: the combat scene
CS_X0, CS_W = 252.0, 240.0
cs = load("combat_scene.png")
CS_H = CS_W * cs.height / cs.width
CS_Y0 = 434.0 - CS_H
put("combat_scene.png", CS_X0 + CS_W / 2, CS_Y0 + CS_H / 2, w=CS_W, z=4)
BASH_X = CS_X0 + CS_W * 760 / 1400.0

# ================================================================= b: state as an unordered set of entity tokens
BAG = (518.0, 262.0, 92.0, 139.0)
rbox(*BAG, fc="white", ec=BLUE, lw=0.95, r=18, z=2, ls=(0, (3, 2)))
txt(BAG[0] + BAG[2] / 2, 410, "unordered set", size=7.0, color=BLUE)
pill(BAG[0] + BAG[2] / 2, 248.5, "hashed key–value tokens", fc="white", ec=BLUE, tc=BLUE,
     size=6.8, padx=4.0, h=12.5, z=18)

CARD_TOK = [("card_bash.png", (543, 375), 8), ("card_strike.png", (586, 358), -6), ("card_defend.png", (546, 325), 4)]
for name, (tx, ty), rot in CARD_TOK:
    put(name, tx, ty, h=38.5, rot=rot, z=7)
ICON_TOK = [("relic_burning_blood.png", AMBER, (588, 318)), ("potion_blood.png", TEAL, (589, 289)),
            ("node_queen.png", TERRA, (542, 285)), ("map_elite.png", INK2, (566, 276))]
for name, col, (tx, ty) in ICON_TOK:
    ax.add_patch(Circle((tx, ty), 9.6, facecolor="white", edgecolor=col, linewidth=1.0, zorder=6))
    put(name, tx, ty, h=13.2, z=7)
arrow((CS_X0 + CS_W + 1, 355), (BAG[0] - 1, 355), color=INK, lw=1.5, head=7.5)
txt(505, 366, "state", size=7.0, color=INK)

# ================================================================= b: set transformer, four clean attention layers
N_TOK = 7
X_IN = 625.0
COLS_X = [647.0, 670.0, 692.0, 714.0]   # 4 crisp attention-layer columns
NODE_Y = [268.0 + k * 19.0 for k in range(N_TOK)]
# Bottom-to-top: map_elite, potion, relic, enemy, Bash(4), Defend(5), Strike(6)
NODE_COL = [INK2, TEAL, AMBER, TERRA, BLUE, BLUE, BLUE]
arrow((BAG[0] + BAG[2] + 1, 325), (X_IN - 5.5, 325), color=BLUE, lw=1.4, head=7.5)

# Clean horizontal token tracks across input + 4 attention layers
for k, ny in enumerate(NODE_Y):
    ax.plot([X_IN, COLS_X[-1]], [ny, ny], color="#cbd5e1", lw=0.95, zorder=2.2, solid_capstyle="round")

# 4 rounded attention-layer columns with subtle vertical self-attention bus
COL_W_ATT = 13.0
for lx in COLS_X:
    rbox(lx - COL_W_ATT / 2, NODE_Y[0] - 8.5, COL_W_ATT, NODE_Y[-1] - NODE_Y[0] + 17.0,
         fc="white", ec=BLUE, lw=0.75, r=4.8, z=2.5)
    ax.plot([lx, lx], [NODE_Y[0], NODE_Y[-1]], color="#dbeafe", lw=2.0, zorder=2.8, solid_capstyle="round")

# Highlighted multi-token attention fan in BLUE (Layer 1 -> Layer 2 into Bash token k=4)
HL_TOK = 4
ATT_WEIGHTS = [0.42, 0.50, 0.72, 0.95, 1.00, 0.58, 0.65]
for k, (ny, wt) in enumerate(zip(NODE_Y, ATT_WEIGHTS)):
    ax.plot([COLS_X[0], COLS_X[1]], [ny, NODE_Y[HL_TOK]],
            color=BLUE, lw=0.7 + 0.95 * wt, alpha=0.35 + 0.65 * wt, zorder=4.2, solid_capstyle="round")

# Token nodes at input and inside the 4 attention columns
for k, ny in enumerate(NODE_Y):
    ax.add_patch(Circle((X_IN, ny), 3.9, facecolor=NODE_COL[k], edgecolor="white", linewidth=0.6, zorder=8))
for li, lx in enumerate(COLS_X):
    for k, ny in enumerate(NODE_Y):
        is_hl = (li == 1 and k == HL_TOK)
        if is_hl:
            ax.add_patch(Circle((lx, ny), 5.4, facecolor=BLUE_PILL, edgecolor="none", zorder=7.5))
        ax.add_patch(Circle((lx, ny), 3.6,
                            facecolor=BLUE if is_hl else "white",
                            edgecolor=BLUE if is_hl else NODE_COL[k],
                            linewidth=1.0 if is_hl else 0.9, zorder=8))

txt(680.5, 415, "Set transformer", size=7.0, color=BLUE)
txt(680.5, 404, "4 attention layers", size=7.0, color=BLUE)

# ================================================================= b: value, probability of winning the run
SX, SY = 740.0, 405.0
ax.add_patch(Circle((SX, SY), 6.5, facecolor="white", edgecolor=TEAL, linewidth=1.0, zorder=8))
txt(SX, SY - 0.3, "Σ", size=7.5, color=TEAL)
# Clean top pooling connection from final attention column into Σ (no crossing with Pointer lines)
ax.plot([COLS_X[-1] - 4.5, COLS_X[-1] + 4.5], [NODE_Y[-1] + 8.8, NODE_Y[-1] + 8.8],
        color=TEAL, lw=1.3, solid_capstyle="round", zorder=5)
arrow((COLS_X[-1], NODE_Y[-1] + 9.2), (SX - 6.8, SY), color=TEAL, lw=1.1, head=5.8, rad=-0.18)
GX, GY, GR = 798.0, 397.0, 18.5
ax.add_patch(Wedge((GX, GY), GR, 0, 180, width=6, facecolor="#d5dee8", edgecolor="none", zorder=6))
ax.add_patch(Wedge((GX, GY), GR, 58, 180, width=6, facecolor=TEAL, edgecolor="none", zorder=7))
ang = np.deg2rad(58)
ax.plot([GX, GX + (GR - 2) * np.cos(ang)], [GY, GY + (GR - 2) * np.sin(ang)], color=INK, lw=1.3, zorder=8,
        solid_capstyle="round")
ax.add_patch(Circle((GX, GY), 2.6, facecolor=INK, edgecolor="none", zorder=9))
arrow((SX + 7, SY), (GX - GR - 2, SY - 1), color=TEAL, lw=1.1, head=6)
txt(GX, 424, "P(win)", size=7.5, color=TEAL)

# ================================================================= b: relational pointer over the legal actions
CANDS = [("card_strike.png", 0.18, False, 6), ("card_defend.png", 0.11, False, 5), ("card_bash.png", 0.71, True, 4)]
CARD_H = 40.0
CX = 753.0
CYS = [355.0, 312.0, 268.0]
for (name, p, chosen, k), cy in zip(CANDS, CYS):
    if chosen:
        rbox(CX - 17, cy - CARD_H / 2 - 3, 87, CARD_H + 6, fc=BLUE_PILL, ec=BLUE, lw=1.1, r=5, z=4)
    put(name, CX, cy, h=CARD_H, z=6)
    ax.plot([COLS_X[-1] + 4.0, CX - 15.5], [NODE_Y[k], cy], color=BLUE if chosen else GREY,
            lw=1.35 if chosen else 0.85, zorder=3.6, solid_capstyle="round")
    blen = 47 * p / 0.71
    rbox(CX + 18, cy - 4.5, 47, 9, fc="white", ec=GREY if not chosen else BLUE, lw=0.55, r=2, z=5)
    rbox(CX + 18, cy - 4.5, max(blen, 3), 9, fc=BLUE if chosen else GREY, ec="none", r=2, z=6)
txt(CX + 41.5, CYS[0] + 15.5, "Pointer", size=7.0, color=BLUE)

# the chosen action goes back into the game cleanly without colliding with text
poly_arrow([(CX, CYS[2] - CARD_H / 2 - 3), (CX, 233.0), (BASH_X, 233.0), (BASH_X, CS_Y0 - 0.5)], color=BLUE, lw=1.6,
           head=8)
txt(BASH_X - 6, 266, "action", size=7.0, color=BLUE, ha="right")


# ================================================================= a/b: it learns only from its own games
def no_sign(cx, cy, r=9.6):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=TERRA, linewidth=1.1, zorder=9))
    d = r * 0.7071
    ax.plot([cx - d, cx + d], [cy + d, cy - d], color=TERRA, lw=1.1, zorder=11)


def icon_human(cx, cy):
    ax.add_patch(Circle((cx, cy + 2.9), 2.5, facecolor=INK2, edgecolor="none", zorder=10))
    ax.add_patch(Wedge((cx, cy - 4.6), 5.0, 0, 180, facecolor=INK2, edgecolor="none", zorder=10))


def icon_tree(cx, cy):
    pts = {"r": (cx, cy + 4.8), "a": (cx - 3.8, cy), "b": (cx + 3.8, cy),
           "a1": (cx - 5.8, cy - 4.8), "a2": (cx - 1.9, cy - 4.8),
           "b1": (cx + 1.9, cy - 4.8), "b2": (cx + 5.8, cy - 4.8)}
    for p_, c_ in (("r", "a"), ("r", "b"), ("a", "a1"), ("a", "a2"), ("b", "b1"), ("b", "b2")):
        ax.plot([pts[p_][0], pts[c_][0]], [pts[p_][1], pts[c_][1]], color=INK2, lw=0.6, zorder=10)
    for v in pts.values():
        ax.add_patch(Circle(v, 1.2, facecolor=INK2, edgecolor="none", zorder=10))


def icon_bubble(cx, cy):
    rbox(cx - 6.2, cy - 2.4, 12.4, 8.0, fc=INK2, ec="none", r=2.4, z=10)
    ax.add_patch(Polygon([(cx - 3.3, cy - 2.4), (cx - 5.2, cy - 5.8), (cx - 0.5, cy - 2.4)], closed=True,
                         facecolor=INK2, edgecolor="none", zorder=10))
    for dx in (-2.8, 0, 2.8):
        ax.add_patch(Circle((cx + dx, cy + 1.6), 0.85, facecolor="white", edgecolor="none", zorder=10.5))


def icon_eye(cx, cy):
    t = np.linspace(0, np.pi, 30)
    xs = np.concatenate([cx - 6.2 * np.cos(t), cx + 6.2 * np.cos(t)])
    ys = np.concatenate([cy + 3.6 * np.sin(t), cy - 3.6 * np.sin(t)])
    ax.add_patch(Polygon(np.c_[xs, ys], closed=True, facecolor="white", edgecolor=INK2, linewidth=0.8, zorder=10))
    ax.add_patch(Circle((cx, cy), 2.0, facecolor=INK2, edgecolor="none", zorder=10.5))


for fn, nx in zip((icon_human, icon_tree, icon_bubble, icon_eye), (408, 430, 452, 474)):
    no_sign(nx, 276)
    fn(nx, 276)
txt(441.0, 252, "own games only", size=7.0, color=TERRA)

# ================================================================= c: version timeline of the real game
TL_Y = 54.0
ax.plot([22, 474], [TL_Y, TL_Y], color=INK2, lw=1.4, zorder=5, solid_capstyle="round")
arrow((470, TL_Y), (484, TL_Y), color=INK2, lw=1.4, head=8)
VERS = [("v0.107.1", 54.0, INK), ("v0.111", 134.0, INK), ("v0.113", 274.0, TERRA), ("v0.114", 438.0, TERRA)]
for v, vx, col in VERS:
    pill(vx, TL_Y, v, fc="white", ec=col, tc=col, size=7.5, h=13.5)
    ax.plot([vx, vx], [TL_Y + 6.8, TL_Y + 15.5], color=col, lw=0.9, zorder=4)

# v0.107.1: the five original characters
for (c, x_, y_) in (("ironclad", 23, 162), ("silent", 54, 162), ("regent", 85, 162), ("defect", 38.5, 126),
                    ("necrobinder", 69.5, 126)):
    put(f"badge_char_{c}.png", x_, y_, h=32.5, z=6)

# v0.111: the agent starts learning
mini_net(112, 96, 44, 58)
txt(134, 166, "agent starts", size=7.0, color=BLUE)

# v0.113: Forgotten Ritual rule change + two crisp mini-pills + the Fool arrives
FR_Y = 146.0
put("card_forgotten_ritual_before.png", 206, FR_Y, h=76, z=6)
arrow((238, FR_Y), (255, FR_Y), color=TERRA, lw=1.6, head=7.5)
put("card_forgotten_ritual_after.png", 288, FR_Y, h=76, z=6)

# High-contrast rule-edit mini-pills right beneath the before/after cards
PILL_Y = 91.5
pill(205, PILL_Y, "always +3 energy", fc="white", ec=BLUE, tc=BLUE, size=6.8, padx=3.6, h=12.6, z=18)
arrow((248.5, PILL_Y), (259.5, PILL_Y), color=TERRA, lw=1.3, head=5.8)
pill(293, PILL_Y, "after Exhaust", fc=TERRA_PILL, ec=TERRA, tc=TERRA, size=6.8, padx=3.8, h=12.6, z=18)

txt(330, FR_Y - 4, "+", size=12, color=VIOLET)
put("char_fool.png", 359, 84, h=80, anchor="b", z=6)
txt(270, 31, "57 cards · 69 relics · 35 monsters changed", size=7.0, color=TERRA)

# v0.114: real v0.114-modified Fool cards from tmp/game_assets/fig3_restyle/
put("card_wah_ki.png", 414, 134, h=67, rot=11, z=5)
put("card_wah_cho_ki.png", 462, 134, h=67, rot=-11, z=5)
put("card_motif.png", 438, 138, h=69, rot=0, z=6)
txt(438, 35, "40 Fool cards", size=7.0, color=TERRA)
txt(438, 24, "rebalanced", size=7.0, color=TERRA)

# the update reaches the game
arrow((288, 187), (288, 291), color=TERRA, lw=2.1, head=9.5)
txt(295, 266, "update", size=7.0, color=TERRA, ha="left")

# ================================================================= c: an update becomes an experiment
EX0 = 498.0
mini_net(EX0, 126, 44, 52)
txt(EX0 + 22, 113, "same network", size=7.0, color=BLUE)
arrow((680, 258), (EX0 + 36, 180), color=BLUE, lw=1.1, head=7, ls=(0, (3, 2)), rad=0.12)

ROWS = [("v0.111", 180.0, INK), ("v0.113", 126.0, TERRA)]
SEEDS = ["8KQ2", "J4N7", "T0BX", "5RWA"]
for v, ry, col in ROWS:
    arrow((EX0 + 46, 152), (570, ry), color=BLUE, lw=1.1, head=6)
    pill(592, ry, v, fc="white", ec=col, tc=col, size=7.5, h=13.5)
    put("combat_scene.png", 656, ry, h=43, z=6)
    for i, s in enumerate(SEEDS):
        sx = 718 + i * 31
        rbox(sx - 14, ry - 8, 28, 16, fc="white", ec=GREY, lw=0.7, r=3, z=6)
        txt(sx, ry, s, size=6.6, color=INK2)
    txt(835, ry, "…", size=8, color=INK2)
txt(764, 204, "same seeds", size=7.0, color=INK)

# each change of the update can be switched back; one is flipped and the games are replayed
SW = [("card_forgotten_ritual_after.png", 46, True), ("relic_akabeko.png", 25, False),
      ("potion_fire.png", 25, False), ("map_monster.png", 23, False), ("map_unknown.png", 23, False)]
for i, (name, hh, flipped) in enumerate(SW):
    sx = 544 + i * 47
    put(name, sx, 73, h=hh, z=6)
    col = BLUE if flipped else TERRA
    rbox(sx - 14, 33, 28, 14, fc=col, ec="none", r=7, z=6)
    ax.add_patch(Circle((sx - 7 if flipped else sx + 7, 40), 5.4, facecolor="white", edgecolor=col, linewidth=0.9,
                        zorder=7))
txt(638, 19, "switch one change back", size=7.0, color=BLUE)
arrow((768, 53), (714, 102), color=BLUE, lw=1.5, head=8, rad=0.38)
txt(782, 53, "replay", size=7.0, color=BLUE, ha="left")

ax.set_xlim(0, W_IN * U)
ax.set_ylim(0, H_IN * U)
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(R, "paper", "figs", f"fig1_world.{ext}"), dpi=350, facecolor="white")
print("written paper/figs/fig1_world.{png,pdf,svg}")
