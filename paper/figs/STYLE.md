# Figure style rules (2026-11-25)

These rules apply to every main figure.

## Rules

1. Let the picture speak. Far fewer words on the figure, above all no small grey notes. A panel gets its axis labels,
   tick labels, a compact legend and at most two short labels (six words or fewer each) that name the thing a reader
   must see. Numbers that are already readable from the plot are not repeated in a box. No explanatory sentences, no
   "callout cards" full of text, no method details (file names, tensor indices, checkpoint ids such as `u45200`).
2. Bigger lettering. At the final size nothing is smaller than 6.5 pt; axis labels 7.5-8 pt; tick labels 6.5-7 pt;
   the few in-plot labels 7-8 pt, bold and in the colour of what they label; panel letters 11 pt bold. Keep the figure
   size of the existing scripts (about 8.5 x 6 in at 350 dpi), so this means larger fonts than the current 4-5.5 pt and
   fewer objects, not a larger canvas.
3. Dense and rectangular. No large white areas inside the outer frame; panels share edges and gridlines where they
   can; a panel that would be mostly empty is merged with a neighbour or given a second view of the same data.
4. Variety of forms. Not every panel is a line plot. Prefer heat maps, dumbbells, ridge/density plots, aligned
   value columns next to a plot, matrices, slope charts, small multiples; a line plot only where the trajectory itself
   is the message.
5. Much more game art. Characters are shown by their portraits or badges (never by a coloured dot alone), cards by
   card renders, bosses, relics, potions and map nodes by their icons, and large enough to be recognised (a character
   badge at least 16 pt high, a card thumbnail at least 30 pt high).
6. One look across panels and figures: the same palette, the same character colours, the same badge art, the same
   rounded outer frame, no per-panel titles (panel letters only).

## Palette and conventions (keep the constants used in the existing scripts)

- Blue `#2166ac` (ours / before / positive), terracotta `#c85228` (loss / after an update / attention), teal `#1b8a6b`
  (preserved / healthy), amber `#c98a1b` (energy, exposure), violet `#5e4cc4` (the Fool; the independent agent
  AlphaDefect), ink `#1f2937`, secondary ink `#475569` (use sparingly; not for small notes).
- Character colours and badges: the same `CHAR_COLORS` and badge art in every figure script.
- Font Helvetica (the scripts set it), no italics except card names where the text needs them, minus signs as "−".
- Outer rounded frame `#e2e8f0` as in the existing figures. White background.
- Every panel is drawn from its source data (`paper/figdata/`); numbers are never typed in when a data file
  holds them. A figure script writes `.png` (350 dpi), `.pdf` and `.svg` next to itself.

## Game art

- Game art (character portraits, card renders, boss, relic, potion and map icons, event paintings) is decoded from the
  game package of one's own copy of the game into `tmp/game_assets/<folder>/`, which the scripts read. It is not
  distributed with this repository; the rendered figures in `paper/figs/` contain it.
