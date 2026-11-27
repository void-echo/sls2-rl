# parity/ — real-client parity oracle

Replays headless-engine runs in the real Godot StS2 client and diffs the state step by step. Results and
divergence classification: `docs/survey/parity_report.md` (v0 first round, then the v1 re-validation).

## Validate the current engine in one command

The batch for the current engine is `v2`, recorded in `tmp/parity/runs/v2/` (the two orphaned-hook-action cases in
`tmp/parity/runs/v2/orphan_hook/`) with the private copy `tmp/parity_engine_v2/` (engine `b7c9b22`: the observation
shared with the mod, `engine/src/Shared/`). With the game closed:

```bash
python3 parity/run_validation.py check   # no launch: builds the mod, validates every run dir and job file, compare.py self-test, plan
python3 parity/run_validation.py run     # LAUNCHES THE GAME: all sessions unattended, mod removed at the end, summary
```

`run` first repeats `check` and launches nothing if it fails. It then installs the mod once, plays the runs in game
sessions of 8 through `oracle.py` (each session: user data backed up, settings.save temporarily edited: other mods
off, windowed 1280x720, muted; launched via Steam; 60 s no-progress watchdog; godot.log flood guard; quit; settings
restored and every user file verified byte-identical, anything touched restored), removes the mod in a `finally`
(also on Ctrl-C or a failed session), counts alarm lines in each session's godot.log (`didn't find it in any queue`,
the orphaned-hook-action error, and `completed with exception`), compares every run and prints the summary, also written to
`tmp/parity/runs/v2/summary.txt` / `summary.json`; the full console output goes to `validation.log` there.
Expected duration: about an hour (103 runs, 9497 steps, 13 sessions; the estimate from ~0.21 s per decision plus
launch/quit overhead is 61 min, the hard caps add up to 3.3 h). Useful options: `--resume` (skip runs that already have a finished
`client.jsonl`, e.g. after an interruption), `--only drift,smith` (groups), `--keep-mod`, `--max-sessions N`.
`run_validation.py record` re-records the batch on the private engine copy (no game), `--force` overwrites; `--batch`
selects another batch directory (the previous one, `v1b`, engine `3405f5c`, was played in full: 83 of 85 runs
identical to game over, 2 truncated, no divergence; it is kept as recorded).

The batch (`GROUPS` in `run_validation.py`; `coverage.json` next to the manifest counts what it exercises): 58 regular
runs (24 random, 12 greedy, 6 per-turn search, 10 `nav`, 6 room-preferring greedy; Ironclad except one run per other
character; A0 and A10; same seeds and drivers as `v1b`) and 45 scenario runs. The navigation rules: Driftwood reroll
then skip, potions used on card reward and removal screens, Forbidden Grimoire's removal reward (decline and back-out),
Smith, Cook (Meat Cleaver), merchant removal, Sozu at a shop (with White Beast Statue for potion rewards); the custom
screens: Crystal Sphere, Trial, Fake Merchant (once with Foul Potion); direct entries into acts 2 and 3; the two
orphaned-hook-action cases; and, for the observation fields added since `3405f5c`, Headbutt putting cards on top of
the draw pile, Decimillipede (segments dead but listed, with Bludgeon), Whistle (stunned moves), a shop with a full
potion belt, a full belt on a rewards screen (`ctx.blocked`), Ambergris (extra turn), Defect with orb cards, six
counter relics (Kunai, Shuriken, Pen Nib, Letter Opener, Ornamental Fan, Nunchaku) and Orobas (an option relic with
own state). Recorded: 104 steps with a known draw-pile top, 96 dead-but-listed enemies, 109 stunned moves, 52 steps of
an extra turn, 87 with two or more orbs, 78 shop shelves with unaffordable items, 29 blocked potion rewards, 6
`relic_d`, 1308 relic counters, 18699 targeted and 307 all-enemy play previews; and for the navigation rules 28 card
reward skips that decline, 7 back-outs with the reward held back and 8 releases, 3 removal-reward declines and 1
back-out, 17 Smith / 7 Cook / 18 merchant-removal selection steps (none offers a cancel), 6 must-take-all screens, 4
shop visits with Sozu (no `buy_potion`), 340 act-2 and 54 act-3 steps.

The oracle speaks the v1 RL protocol (`engine/RL_PROTOCOL.md`, protocol 1) and records against a private engine copy
(`tmp/parity_engine_v2/`: `rsync -a engine/ tmp/parity_engine_v2/`, rebuilt there, source commit in its
`ENGINE_COMMIT`), never against `engine/` itself. `oracle.py check` rejects recordings made with older copies
(`tmp/parity_engine_v1b/`, `tmp/parity_engine_v1/`): they lack the current observation fields.

## Pieces

- `mod/` — the `ParityOracle` client mod (C#, net9, references the game's `sts2.dll`/`GodotSharp.dll`/`0Harmony.dll`).
  Inert unless `ParityOracle.config.json` (next to the DLL) points at a control dir that contains `job.json`.
  When a job runs it patches every save/progress/settings/history/replay write to a no-op (SaveManager methods and the
  GodotFileIo / CloudSaveStore / SteamRemoteSaveStore write/delete/rename calls; `SaveManager.SaveRun` writes nothing
  but completes on the next process frame, the timing of a real save, which the orphaned-hook-action case depends on,
  see `Entry.SaveRunLater` and `engine/README.md` "Run save timing"), skips `PunchOff.PunchEachOther`
  (an animation loop that busy-loops the main thread under FastMode=Instant, a client bug), sets FastMode=Instant in
  memory, and replays each run:
  - `Driver.cs` builds the run exactly like v1 `Harness.StartRun`: `Rng.Chaotic` seeded from the run seed, a fresh
    in-memory `ProgressState` (UniqueId `SLS2RL1`, FTUEs off; the user's profile object is put back before the main
    menu), `Player.CreateForNewRun` with `UnlockState.all`, acts from the `act_selection` RNG with `GetRandomList`'s
    TestMode semantics, `RunState.CreateForNewRun`, `SetUpNewSingleplayer`, `CardRarityOdds.regularCommonOdds` reset,
    then the client's own `NGame.StartRun`. Optional scenario `setup` commands (`Scenario.cs`: `obtain_relics`,
    `dbg_procure_potion`, `dbg_add_card`, `rl_enter_room`, `dbg_enter_act`) make the same game calls as the engine's
    scenario commands at the same point, and the replay waits for each command's game task like the engine does.
  - `Client.cs` decides which v1 decision the client shows (only once the screen is up, the executor idle, no relic in
    its activation flash, and the digest stable for 3 samples), builds the candidates (from enabled UI elements where the client has them: map
    points, reward buttons, proceed buttons, option buttons, relic holders; from the engine's legality code where the
    UI computes it from model state: card play targets, the potion popup rules) and performs each one the UI way:
    `TryManualPlay`, `EnqueueManualUse`, `DiscardPotionGameAction`, `NEndTurnButton.CallReleaseLogic`, `ForceClick` on
    map points / reward buttons / proceed buttons / event and rest options / the chest / relic holders / card-reward
    alternative buttons, the card holder's `Pressed` signal, `OnTryPurchaseWrapper` (what the merchant slots call),
    `SetTool` + `CellClicked` for the Crystal Sphere, and the abandon popup's yes for Trial's Double Down. A queued
    potion discard greys the potion out until it leaves the belt (`NPotionHolder.DisableUntilPotionRemoved`, the
    engine's `_pendingDiscards`). Steps the engine has no decision for (a finished event's synthetic PROCEED, a rest
    site or opened chest with nothing left, ancient dialogue, closing a merchant inventory) are automatic; the map is
    read even while a terminal rewards screen or a finished Crystal Sphere screen is still on the overlay stack.
  - `Nav.cs` is the engine's "no navigation cycles" bookkeeping (`engine/RL_PROTOCOL.md`), kept per client screen:
    which rewards a rewards screen has declined (card reward Skip / removal cancel before anything changed on the
    sub-screen) or holds back (the same back-out after a reroll or a potion, until a reward is taken or a potion is
    used or discarded on the rewards screen), how many inputs other than a pick were given on a card reward screen or
    a selection, and where a screen was opened from. The origin is set only around the synchronous part of the click
    that opens the screen (reward button, Smith / Cook button, the removal entry's `OnTryPurchaseWrapper`), like the
    engine's `_openingReward` / `_openingService`. With it `Client.cs` offers the engine's candidates: card reward
    candidates with `cards` and `alts`, declined and held-back rewards left out, no cancel on Smith / Cook / merchant
    removal, `buy_potion` only when the potion can be procured (not with Sozu, not with a full belt).
  - `Selection.cs` is the card-selection capture (the engine's `SelectionPrefix` fields, kept in an `AsyncLocal` so it
    flows with the `From*` entry's continuation: hook-initiated selections reach the selector after the entry returned)
    and the local-only selector (`CardSelectCmd.UseSelector(localOnly: true)`, the engine's seam), so
    `ctx.sel/cancelable/source/pile/prompt` and the skip/confirm rules are what the real screens would allow. The real
    selection screens themselves are not exercised.
  - The observation is not a port: `ParityOracle.csproj` compiles the engine's own `engine/src/Shared/ObsShared.cs`
    (observation, ctx builders, the data parts of the candidate descriptors: card previews, rest heal, option relics,
    reward contents, shop shelf) and `KnownTop.cs` (the known draw-pile top), so the client builds every field exactly
    as the engine does. `Obs.cs` only holds the client's observation state, and `Entry.cs` installs KnownTop's
    recording prefixes (`CardPileCmd.Add`, `CardPileCmd.Shuffle`, `CardPile.RandomizeOrderInternal`); `Obs.ResetRun`
    resets them when a run starts. Two client-only rules, both about animation windows that take no time headless:
    the client reads a screen many times while it settles, so observations only peek at the known top
    (`KnownTop.Peek`) and the reading is recorded once, when the decision is reported (`Obs.CommitKnownTop`, the
    engine's once-per-decision `KnownTop.Current`); and a decision is not read while a counter relic (Kunai, Pen Nib,
    ... 21 of them) is in its one-second activation flash (`_isActivating`, during which `DisplayAmount` shows the
    threshold and `Status` is reset; in Instant mode the flash takes no time either, it matters in watch mode).
- `headless.py` — client for the RL protocol, runs the private engine copy.
- `record.py` — runs on the headless engine → `headless.jsonl`, `actions.jsonl`, `run.json` per run. Test-driver
  policies: `random` (biased random), `greedy` (survival heuristic), `search` (greedy outside combat, each player turn
  the best of K parallel rollouts replayed from the action prefix; reaches act 2), `nav` (random, but opens card and
  removal rewards, rerolls, uses potions on those screens and then backs out or takes, prefers Cook / Smith / merchant
  removal: exercises every navigation rule). Knobs: `--prefer-rooms`, `--hoard-potions`, `--explore`,
  `--setup '<json list of scenario commands>'`, `--play-first CARD.X` (or `POTION.X`: use it in combat), `--ascension 0,10`.
- `oracle.py` — `check <run_dirs…>` (no launch: each run's action log against its headless digest, setup commands the
  mod supports, the current schema; writes the per-session job files to `tmp/parity/control_check/`; checks the game
  and settings paths; prints the plan) and `install` / `uninstall` / `run <run_dirs…> [--chunk N]`: one game session per chunk (backs up user
  data, temporarily edits settings.save: other mods disabled, windowed, muted; launches via Steam; 60 s no-progress
  watchdog kills the game; godot.log growing by more than 200 MB (a runaway loop) kills it at once and the flooded
  log is deleted; a hard session timeout scaled to the step count; quits the game at the end; restores settings and
  verifies user files are byte-identical, restoring anything touched). Set `STS2_STEAM_ID64` to your SteamID64 (the
  folder name under `~/Library/Application Support/SlayTheSpire2/steam/`) before `run`.
- `compare.py` — first divergence per run (step, decision, field path, both values) + summary. Known differences
  are counted separately so later divergences are still found: `localized_string_var` / `localized_option_key`
  (display text: loc key headless, UI-language text in the client), `trial_entrant_number` (a cosmetic
  `Rng.Chaotic` number in the Trial text; no longer exported, only in older recordings), and
  `engine_bug:hook_selection_ctx` (the lost selection context of hook-initiated selections in recordings made before
  the engine fix, see the report; cannot occur in new ones). The fields added since `3405f5c` got no rule: none of
  them is display text, and the two client-only timing issues (relic flash, known draw-pile top) are handled in the
  mod (above), not excused in the comparison.
  `compare.py --selftest` classifies synthetic runs (identical, a diverging field, an artifact then a divergence, a
  stuck client, no client data) without any game data.
- `run_validation.py` — the batch (groups, record options, order of play), `record` / `check` / `run`, see the top.

**Engine version the oracle mirrors.** The observation and the descriptors' data are shared code (above), so they
follow the engine automatically; rebuild the mod after changing `engine/src/Shared/`. What is still mirrored by hand
is the decision logic: `Client.cs` (which decision, which candidates, how each is performed) and `Nav.cs` follow
`Harness.cs` / `RunSimulator.Rl.cs` / `Prompts.cs` as of the engine commit in `tmp/parity_engine_v2/ENGINE_COMMIT`
(navigation-cycle rules; enemies that stay dead to revive keep their target index; the run-save timing of `2526cf3`
is in `Entry.SaveRunLater`). `oracle.py check` rejects recordings without the current observation fields (`g.odds`,
`g.act_rooms`, `combat.draw_top`, ...): batches recorded with older copies (`v1b` and before) can be compared again
but not replayed by this mod. After an engine change to decisions or candidates, port it to `Client.cs` / `Nav.cs`,
`rsync` a new private copy and re-record (`run_validation.py record --force`).

## Run

Ad-hoc runs (the batch above does all of this):

```bash
rsync -a engine/ tmp/parity_engine_v2/ && (cd tmp/parity_engine_v2 && dotnet build src/Sts2Headless/Sts2Headless.csproj)   # after engine changes
python3 parity/record.py --out tmp/parity/runs/x --runs 6 --characters Ironclad --ascension 0,10 --seed-prefix x_ --fresh
python3 parity/record.py --out tmp/parity/runs/x --runs 2 --policy greedy --explore 0.3 --seed-prefix sc_ --fresh \
        --setup '[{"cmd":"rl_enter_room","type":"Event","event":"CRYSTAL_SPHERE"}]'
python3 parity/oracle.py check tmp/parity/runs/x/*/ --chunk 6    # no launch
python3 parity/oracle.py run tmp/parity/runs/x/*/ --chunk 6      # the game must not be running already
python3 parity/compare.py tmp/parity/runs/x/*/
```

Scratch (runs, backups, control dir with `oracle.log`, built DLL) lives in `tmp/parity/`. The client needs about
0.2 s per decision.

## Watch mode

```bash
python3 parity/oracle.py run <run_dir> --watch Fast --step-delay 0.6 --timeout 3600 --watchdog 120 --out-name watch_client.jsonl
```

With `--watch` and a positive `--step-delay`, merchant and Fake Merchant inventories stay visible during purchases;
card removal opens the selection screen, and leaving closes the inventory. The corner overlay shows the action
index, floor, room, pause state and keys:

- **Space** pauses/resumes replay inputs; current animations finish, and paused time does not count toward
  `--watchdog` or `--timeout`.
- **R** reloads this node's room-entry snapshot and replays from its recorded action index.
- **B** reloads the previous node's room-entry snapshot.

The last eight room-entry snapshots stay in memory; no game saves are written. A node without a valid entry
snapshot (including a room changed by scenario setup) shows an unsupported message. Each successful rewind appends
`{"rewind": <action index>}` to the output JSONL; that session's output is for watching and need not compare with
the headless recording. Without a positive watch delay, batch replay behavior is unchanged.

## Remove / disable the mod

- Remove: `python3 parity/oracle.py uninstall` (deletes
  `~/Library/Application Support/Steam/steamapps/common/Slay the Spire 2/SlayTheSpire2.app/Contents/MacOS/mods/ParityOracle`).
- Disable without removing: toggle "Parity Oracle" off in the game's Mods screen, or just make sure
  `tmp/parity/control/job.json` does not exist (without it the mod does nothing).
- Steam "verify integrity of game files" also removes it.
