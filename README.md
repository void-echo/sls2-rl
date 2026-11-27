# SpireZero

SpireZero is a small reinforcement learning agent that learns the card game Slay the Spire 2 from its own play,
trained with proximal policy optimization (PPO) on the game's own released code, and followed through real game
updates. It uses no human games, no language model and no hand-written card values or strategy. It wins 73% of games at
the highest difficulty, matching an independent, more heavily engineered agent. Because every game update is a
documented list of rule edits that can be reverted one by one, the agent shows what a policy relearns, retains and
overlooks when its world changes. This repository holds the code and the figure data for the manuscript
(`paper/SpireZero_manuscript.pdf`).

## Contents

| folder | what it holds |
|---|---|
| `engine/` | Headless host for the game: runs the game's own `sts2.dll` in a plain .NET 9 process, with Godot replaced by a generated stub, and offers every decision the real client offers as JSON lines (`engine/RL_PROTOCOL.md`, `engine/README.md`). `src/Sts2Headless/Revert/` restores single rules of an earlier version on request (`SLS2_REVERT`), for the patch experiments. Started as a fork of [sts2-cli](https://github.com/wuhao21/sts2-cli) (`engine/LICENSE`). |
| `rl/` | Training stack: observation encoder, set transformer policy and value network, actors, PPO learner, evaluation, tracing, a small built-in stand-in game (`fake_engine.py`) and unit tests (`rl/README.md`, in Chinese). |
| `tools/` | Engine checks (smoke, determinism, scenarios, save/load, navigation cycles, rule reverts), the analysis and evaluation scripts behind the paper, `update_game.sh` for re-checking the engine after a game update, and background launch/stop wrappers in `tools/b100/`. |
| `parity/` | In-client verification: a mod for the real game client that replays runs recorded by the engine and compares them decision by decision (`parity/README.md`). |
| `paper/` | The manuscript; `figs/` holds the scripts and rendered outputs of the six main and two Extended Data figures, `figdata/` the data behind the panels. |

## Requirements

- Slay the Spire 2, your own copy from Steam. No game files are included in this repository: `engine/setup.sh` copies
  the game's DLLs (unmodified) and English text tables from your installation. The paper's experiments used several
  game versions (v0.107.1 to v0.114); the engine runs the version you have installed.
- .NET 9 SDK, installed under `~/.dotnet` (the Python side starts `~/.dotnet/dotnet`).
- Python 3.10 or newer with PyTorch and NumPy (pytest for the tests); the figure scripts also use matplotlib, Pillow,
  SciPy and fontTools. Full training runs use a GPU and many CPU cores (each engine process takes about 140 MB).

## Build the environment and run a smoke test

```bash
cd engine && ./setup.sh && cd ..                  # or ./setup.sh /path/to/the/folder/with/sts2.dll
python3 tools/rl_smoke.py engine --runs 5          # random legal play: schema, genuine endings, determinism
python -m pytest -q rl/tests                       # training stack on the built-in stand-in game (no game needed)
python -m rl.train --preset tiny --name smoke      # tiny model on the stand-in game, CPU
python -m rl.train --name ironclad_a0 --engine real --engine_dir engine --n_workers 32 --envs_per_worker 8
python -m rl.evaluate --ckpt runs/ironclad_a0/ckpt/latest.pt --n_seeds 200 --greedy
```

`engine/README.md` lists the full engine test suite; every training option is a command-line flag (`rl/config.py`).

## Figures and trained networks

The scripts in `paper/figs/` redraw the figures. To redraw them exactly they need game art (character portraits, card
renders, icons) extracted from your own copy of the game, which is not distributed; the rendered figures are included as
PNG, PDF and SVG, and the data shown in the panels is in `paper/figdata/`.

## What is not here yet

- The rule-reversion switches used for Figure 3 need the earlier versions of the changed cards, relics and rules.
  Those earlier rule bodies come from the game's own code and are not distributed; `engine/src/Sts2Headless/Revert/`
  holds the switch mechanism only, so reverts are inactive in this copy.
- `paper/figdata/` holds the data shown in the figures. A few figure scripts still read intermediate result files that
  are not included yet; these and the trained networks will be added on publication.
- A few unit tests in `rl/tests` compare against earlier commits of the development history and fail outside it.

## Citation

```bibtex
@article{wang2026spirezero,
  title   = {What an RL agent relearns, retains and overlooks when its world is patched},
  author  = {Wang, Ding and Guo, Xingyu and Jin, Junru and Qiao, Jianbo and Wu, Shu and Liu, Qiang},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

The arXiv identifier will be added once it is assigned. Slay the Spire 2 is developed by Mega Crit; this project is
not affiliated with or endorsed by Mega Crit.
