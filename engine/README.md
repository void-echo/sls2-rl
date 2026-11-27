# Headless Slay the Spire 2 engine (harness v1)

This directory runs the real game logic (`sts2.dll`, game version v0.111.0) in a plain .NET 9 process, with Godot
replaced by a headless stub of `GodotSharp.dll`, and exposes it to Python through the JSON-lines RL protocol in
`RL_PROTOCOL.md`. It started as a fork of [sts2-cli](https://github.com/wuhao21/sts2-cli) (`UPSTREAM_COMMIT`); harness v1
replaced its simulator completely (the upstream text protocol and its Python tools are gone).

The design goal is fidelity: the agent is offered exactly the decisions the real client offers, every input is the
call the client's UI makes, RNG is consumed exactly as in the client, and a run can only end by the game's own rules.
The audit that motivated v1 is `docs/survey/harness_audit.md` (its last section records the v1 implementation), the
per-item review is `docs/survey/content_review.md`.

## Setup and build

```bash
./setup.sh                 # copies the game DLLs (unmodified) into lib/, regenerates the Godot stubs, builds
./setup.sh --x64           # x86_64 DLLs into lib_x64/ for Linux deployment (copy to lib/ there, or set STS2_LIB)
DOTNET_ROOT=~/.dotnet PATH=~/.dotnet:$PATH dotnet build src/Sts2Headless/Sts2Headless.csproj
dotnet src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll        # JSON lines on stdin/stdout
```

The game DLLs are not modified (v0 IL-patched `sts2.dll`; v1 does everything with Harmony at runtime).
`lib/GodotSharp.api.dll` is the game's real GodotSharp, read only as metadata by the stub generator. The macOS
arm64 and x86_64 `sts2.dll` differ only in the PE machine field (same IL, same Godot references), so the engine built
on the Mac runs on Linux x86_64 with `lib_x64/`.

## Architecture

```
Program.cs     JSON-lines front end (rl_start / rl_act / rl_save_get / rl_load / quit + scenario commands)
GameLoop.cs    the single game thread's SynchronizationContext and TaskScheduler
Harness.cs     run lifecycle, stable-state detection (Decide), UI stand-ins for flows, legality helpers, player inputs
SaveLoad.cs    in-memory client saves, process-state snapshots, and the client's Continue path
Prompts.cs     the screens the game waits on, each holding the TaskCompletionSource the game awaits
RunSimulator.Rl.cs   candidate lists (each candidate = descriptor + the UI call it stands for), ctx per decision
../Shared      the observation and the candidates' descriptor data (ObsShared.cs) and the known draw-pile top
               (KnownTop.cs): one source compiled into both the engine and the real-client parity mod (parity/mod)
Patches.cs     every Harmony patch, grouped by purpose; ItemFixes = per-item fixes table
Diag.cs        error context: ring buffer of game events (actions, choices, combat/room transitions, inputs)
Debug.cs       scenario construction for tests (set_player, rl_enter_room, dbg_*)
Census.cs      dbg_census: lists every field of the run and combat object graphs (the state census, tools/state_census.py)
Localization.cs  LocManager without Godot (English tables, lookups never throw)
../GodotStubs  headless GodotSharp.dll (Semantic/ hand-written, Generated/ generated)
```

**One thread, queued continuations.** All game code runs on the process main thread. `GameLoop` is installed as the
`SynchronizationContext`: `Post` only enqueues, the harness drains the queue, so `await Task.Yield()` and every
deferred continuation runs after the current synchronous work, the "next frame" order of the client. A second,
lower-priority queue (`PostWhenIdle`) holds what the client does only after an animation or timed wait (rewards after
the death animation, the chest after its button), so it runs once everything else has settled. `Task.ContinueWith`
without a scheduler (which runs on the thread pool from an await continuation, in the client too) is rewritten to the
game loop where the game uses it. A post from any other thread is an integrity error.

**Prompts instead of polling.** Wherever the client shows a screen and awaits the player, the harness provides a
stand-in that hands the game a `TaskCompletionSource` (`Prompts.cs`): the card selector is registered with
`CardSelectCmd.UseSelector(localOnly: true)`, i.e. where the screens are, so the real single-player path around it
still runs (`ReserveChoiceId`, `SignalPlayerChoiceBegun/Ended`, a choice inside a hook becomes a queued hook action);
the card reward, bundle and Crystal Sphere screens are replaced by Harmony prefixes on their `ShowScreen` /
`OptionSelected` / `CardsSelected`; `RewardsSet.testSelector` (TestMode's hook in `RewardsSet.Offer`) opens the rewards
prompt. After every input the loop is drained and `Decide` reads the stable state: the newest waiting prompt, else
combat play phase with an idle queue, else the room's own decision.

**No navigation cycles.** A candidate that would only lead back to an identical earlier decision, with no
game-state change, is not offered, as long as everything its sub-screen shows is visible at the parent decision
(`RL_PROTOCOL.md`, "No navigation cycles"). Card reward candidates on the rewards screen carry their cards and
alternatives; Skip on a card reward (or cancelling a removal reward) opened from the rewards screen declines the reward,
which the game then leaves behind at proceed (`SkipLocalRewardsSet` -> `Reward.OnSkipped`), exactly as if it had never
been opened; after a reroll or a potion it is a plain back-out and the reward is held back until the game changes again.
Smith, Cook and the merchant's removal have no cancel, and a merchant potion is offered only when it can be procured.
The origin of a screen is known because the inputs that open it (`SelectLocalReward`, `ChooseLocalOption`, the removal
entry's `OnTryPurchaseWrapper`) reach `ShowScreen` / the selector before their first await (`RunSimulator.TagOrigin` in `Harness.cs`).
`tools/cycle_check.py` and the content fuzz suites check that no (decision, obs, cands) repeats within a room or turn.

**Run save timing.** The client always saves a single-player run, and the write runs on the thread pool, so every
awaited `SaveManager.SaveRun` yields and its caller resumes in a later frame. New runs use `shouldSave: false`,
where the save returns a completed task; that changed game behaviour (`CombatManager.EndCombatInternal` awaits the save
before `SetCombatState(NotInCombat)`, and a combat ending inside a hook action's inline completion cancelled that action
before `PopAction`: "Tried to pop action GenericHookGameAction ... didn't find it in any queue"). `RunSaveManager.SaveRun`
is therefore replaced by a stand-in that writes nothing and completes through `GameLoop.PostWhenIdle`, i.e. like the
write, once the work queued now has settled (patch category (e), `Patches.PatchSaving`).

Every such save is also captured with the game's `ToSave` and JSON serializer, plus the profile, `Rng.Chaotic`,
`CardRarityOdds.regularCommonOdds` and Yummy Cookie's cached character (its first icon lookup consumes Chaotic RNG).
`info.save` identifies the latest snapshot; `rl_save_get` exports it and `rl_load` restores a room-entry snapshot in
the same or another process. Loading first tears down the old run like `rl_start`, restores process state and attaches
the selector and treasure synchronizer, then follows `RunState.FromSerializable`, `SetUpSavedSingleplayer`, and
`NGame.LoadRun`'s `Launch`, `GenerateMap`, `LoadIntoLatestMapCoord` calls; saving still writes no files after load.
These are training/evaluation tools, outside the agent's candidate actions. Saves start at id 1 for each `rl_start`,
and subsequent saves keep increasing across loads.

Only `kind:room` can be loaded; replay the room's inputs to reach its rewards or other decisions. The client re-enters
pre-finished rooms (`RunManager.LoadIntoLatestMapCoord`, `EventRoom.EnterInternal`), repeating entry hooks such as Maw
Bank's 12 gold, so `finished` and `event` snapshots are exported but rejected by `rl_load`.
For matching relic ids, differences in `obs.relics[i].d` or `cards` backed by unsaved instance members (for example
Ornamental Fan's turn counter and Bing Bong's clone bookkeeping) are recorded as `known` using a reflection table
of the game DLL; saved members and every other observation/candidate difference still fail, without changing the
observations or compensating the game. A relic's display `status` is not saved either: Lava Rock shows `Disabled`
(greyed out) before a reload and normal after it, as in the client. `tools/saveload_check.py` (2026-10-28): 170 runs,
620 room saves replayed in the same and in a fresh process, every run's outcome and step count unchanged; the only
differences were these three unsaved relic states.
Continue increments `SerializableRun.NumReloads` (`SaveManager.IncrementNumReloads`); this feeds run history/metrics,
not gameplay. Save timestamps and elapsed run time also retain the client's wall-clock semantics; none is in
`decision`, `obs` or `cands`, which the replay check compares without excluding fields.

**The observation.** It is built by `../Shared/ObsShared.cs` from the game objects at the decision, under the rule of
`docs/02_观测与编码方法论.md` (what a player who knows the game and remembers the run knows, nothing unrevealed). The
parity mod compiles the same file, so the real client is compared against the same code. Enemies come from
`CombatEnemies()`: the dead that stay in the fight to revive (the Decimillipede segments, Test Subject, the illusions Parafright and Eye With Teeth) are listed
with `alive:false` and their revive intent, and candidate targets index the same list. The known top of the draw
pile (`../Shared/KnownTop.cs`) is kept by record-only patches (category (f)) on `CardPileCmd.Add` to the top or at
random, `Shuffle` and `RandomizeOrderInternal`, and is checked against the real pile at every observation, so it can
only under-report. The field census (`dbg_census`, `tools/state_census.py`, `docs/survey/state_census.md`) classifies
every field of the game state against the observation; after a game update its new fields show up there.

**TestMode.** `TestMode.IsOn` stays on (switching it off wakes UI code everywhere). The three TestMode branches that
change gameplay are put back to the shipping branch: merchant potion prices (`MerchantPotionEntry.CalcCost` draws
from `PlayerRng.Shops`), Calling Bell and Cauldron rewards; Rolling Boulder hits each creature separately like its VFX
does; the choose-a-bundle screen runs its shipping branch against the stand-in.

### Decisions and the game calls behind them

| decision / input | game call (as the client UI / AutoSlay makes it) |
|---|---|
| play card | `CardModel.TryManualPlay(target)`; targets = alive, hittable enemies allowed by `Hook.ShouldAllowTargeting` (`NTargetManager`) |
| end turn | `RequestEnqueue(new EndPlayerTurnAction(me, turnNumber))` (`NEndTurnButton`) |
| use / discard potion | `PotionModel.EnqueueManualUse(target)` / `RequestEnqueue(DiscardPotionGameAction)`; enable rules of `NPotionPopup` |
| card selection (all `CardSelectCmd.From*`) | the selector's task; min/max/cancelable/can-skip captured from the entry's `CardSelectorPrefs` by a prefix |
| card reward | `NCardRewardSelectionScreen` stand-in under the real `CardReward.OnSelect` loop (cards, Skip, REROLL, SACRIFICE); Skip from the rewards screen declines the reward (see above) |
| rewards screen | `RewardsSetSynchronizer.SelectLocalReward(reward)` (card rewards show their cards and alternatives inline); proceed = `SkipLocalRewardsSet` + `NRewardsScreen.OnProceedButtonPressed` (`ProceedFromTerminalRewardsScreen`, second boss, or `ActChangeSynchronizer.SetLocalPlayerReady`) |
| combat won | `NCombatUi.OnCombatWon`: `CombatRoom.OfferRoomEndRewards()` if `Encounter.ShouldGiveRewards`, else `ProceedFromTerminalRewardsScreen` |
| event option | `EventSynchronizer.ChooseLocalOption(i)` (proceed-type options: `option.Chosen()`), `NEventRoom.OptionButtonClicked` |
| rest site | `RestSiteSynchronizer.ChooseLocalOption(i)`; leave once an option succeeded (`NRestSiteRoom`); Smith / Cook selections without cancel |
| shop / Fake Merchant | `MerchantEntry.OnTryPurchaseWrapper(inventory)`; removal `Used` set like `NMerchantCardRemoval`, removal without cancel; potions only when `PotionCmd.TryToProcure` would succeed |
| chest | `NTreasureRoom.OpenChest`: `DoNormalRewards`, `DoExtraRewardsIfNeeded`, then `PickRelicLocally(i)` / `SkipRelicLocally()`; relics granted like `NTreasureRoomRelicCollection` |
| Crystal Sphere | `CrystalSphereMinigame.SetTool` + `CellClicked(cell)` |
| map | `RequestEnqueue(VoteForMapCoordAction)` (`NMapScreen.OnMapPointSelectedLocally`); travelable set = `RecalculateTravelability` + `MapTravel` |
| Foul Potion at a merchant | usability of `FoulPotion` with the merchant button present; Fake Merchant branch of `OnUse` without its node check |
| Trial's Double Down | `RunManager.Abandon()` (the confirmation popup's yes) |
| new run | `StartRunLobby.BeginRunLocally` + `NGame.StartNewSingleplayerRun`: `ActModel.GetRandomList` ("act_selection" RNG), `RunState.CreateForNewRun`, `SetUpNewSingleplayer`, `FinalizeStartingRelics`, `Launch`, `EnterAct(0)`. The profile is all-unlocked with every act discovered, so act 1 is Overgrowth or Underdocks by seed (a brand-new real profile would always get Underdocks first); verified against the real client in `docs/survey/parity_report.md`. |
| `rl_save_get` (training/evaluation tool) | export a snapshot recorded at the game's `RunSaveManager.SaveRun(AbstractRoom?)`; never a candidate action |
| `rl_load` (training/evaluation tool) | room-entry saves only: the client's Continue path (`RunState.FromSerializable`, `SetUpSavedSingleplayer`, `Launch`, `GenerateMap`, `LoadIntoLatestMapCoord`); never a candidate action |

**Terminal integrity.** `game_over` is reported only for a death by the game's rules (all players at 0 HP), an abandon
(`RunManager.IsAbandoned`) or `WinRun` in the last act; everything else that looks terminal, a decision without any
legal action, a stall, a game exception (`TaskHelper.UnobservedFault`, loop callbacks, unobserved tasks) or a foreign
thread is returned as `type:error` with a state description.

**Run reset.** `rl_start` first cancels every pending prompt (so the old run's tasks, including a suspended combat
turn loop, unwind now), calls `RunManager.CleanUp`, drains and clears the loop, resets the selector, the profile
(`ProgressState.CreateDefault`, fixed UniqueId), `Rng.Chaotic` (seeded from the run seed) and
`CardRarityOdds.regularCommonOdds`, and asserts that no combat or turn loop survived. A run in a long-lived process is
bit-identical to the same seed and actions in a fresh process (`tools/determinism.py`).

**Godot singletons.** `NAudioManager.Instance` / `NDebugAudioManager.Instance` return inert objects (gameplay calls
audio without null checks). `NGame.Instance` stays null on purpose. Gameplay methods that dereference other UI
singletons are listed in `Patches.ItemFixes` (one row each: DenseVegetation.Rest, PunchOff.Nab, Amalgamator, Trial,
SoulNexus.AfterDeath, ActionQueueSynchronizer.GetHookActionForId); a new finding is one more row.

## Godot stubs

`sts2.dll` references about 300 Godot types and 1,140 members. The CLR binds a method's references when it compiles
the method, so one missing member makes the whole method (an async method: its state machine) throw, even on a line
that never runs: this is what made Kaiser Crab and Rolling Boulder unplayable in v0. The stub is therefore complete by
construction:

- `src/GodotStubs/Semantic/`: types game logic computes with, with Godot's semantics. The value types (`Vector2`,
  `Vector2I`, `Vector3`, `Rect2`, `Color`, `Transform2D`, `Basis`, `Quaternion`, `Mathf`, ...) are Godot's own C# sources
  (MIT, see `Semantic/README.md`); `StringName`, `NodePath`, `Callable` (delegate calls, `CallDeferred` on the game
  loop: `GameAction.Cancel` depends on it), `Variant`, the collections, `SignalAwaiter` (completes immediately: no
  frames or signals exist headless) and `GD` are written by hand.
- `src/GodotStubs/Generated/Generated.cs`: every other referenced type and member, declared exactly like the real
  GodotSharp (kind, base type, accessibility, virtuality, static-ness, signature) with inert bodies, produced by
  `tools/GodotStubGen` from `lib/GodotSharp.api.dll` and the references of `lib/sts2.dll`.

`tools/gen_stubs.sh` regenerates and then checks that every TypeRef/MemberRef of `sts2.dll` resolves with an exact
signature (custom modifiers and valuetype/class markers included) and that no override can reduce access
(`GodotStubGen check`, currently 0 problems; the older name/arity scanner in `tmp/content_review/StubCheck` also reports 0).
A member missing from a semantic type is an error the generator refuses to paper over.

## After a game update

1. `./setup.sh` (copies the DLLs, regenerates and checks the stubs, builds). A `SEMANTIC ... missing` error means a
   new member of a semantic type: implement it in `Semantic/` from Godot's source.
2. Startup prints `[WARN] item fix ... changed nothing` / `target missing` when an `ItemFixes` row no longer matches.
3. `dotnet tools/GodotStubGen/bin/Release/net9.0/GodotStubGen.dll threads lib/sts2.dll` lists gameplay call sites that
   can leave the game thread or start timers (Task.ContinueWith, Task.Run, Task.Delay, WaitAsync, ...); new ones in
   gameplay code need a look (`ItemFixes` has the `ContinueOnGameLoop` kind).
4. Run the tests below; diff `ModelDb` content (new events/relics) against `docs/survey/content_review.md`.

`tools/update_game.sh <game_dir> <out_dir> [--quick] [--baseline engine/content_ids_v0.111.0.json]` does all of this in `engine/` in place and writes `<out_dir>/report.md`. On v0.111.0 itself: `--quick` 7/7, full 18/19; the one failure is `RELIC.LAVA_ROCK`, whose `status: Disabled` is not restored after a save/load although its saved `HasTriggered` is (same candidates on both sides), so it is a save-load observation difference and not a new regression.

Servers are x86_64 Linux: the game ships `sts2.dll` per architecture (`data_sts2_macos_arm64/` and `data_sts2_macos_x86_64/`, same source, different ReadyToRun code; every other DLL is identical). `setup.sh` on an Apple Silicon Mac copies the arm64 one into `lib/`; a server copy of the engine needs `lib/sts2.dll` from `data_sts2_macos_x86_64/` (an arm64 `sts2.dll` fails with "assembly architecture is not compatible").

## Tests (from the repo root)

| tool | what it checks |
|---|---|
| `python3 tools/rl_smoke.py engine --runs 30 --character C` | schema (`rl/protocol.py`), genuine terminals, same-process replay determinism, throughput, per-decision latency |
| `python3 tools/determinism.py --runs 40 --parallel 32` | each run identical in a fresh process, in one long-lived process, and in 32 parallel processes |
| `python3 tools/fullrun.py --runs 500 --asc 0,10` / `--god` | many random full runs (9999 HP with `--god`, which reaches the act 3 bosses and The Architect) |
| `python3 tools/cycle_check.py --runs 600 --asc 0,10` / `--god` | no navigation cycles: no exact repeat of (decision, obs, cands) within a room or turn, under random, state-deterministic and loop-seeking policies |
| `python3 tools/scenario_tests.py` | act 2/3 bosses, A10 double boss and victory, Lord's Parasol, Fake Merchant, chest, act variants, abandon, potion slots, the orphaned hook action, and one test per navigation rule (card reward skip, reroll / potion then skip, removal reward, Smith, Cook, shop removal, Sozu at the shop, repeated autoplay selections) |
| `python3 tools/saveload_check.py --runs 200 --workers 4` | exact room-entry replay in the same and fresh processes, common random numbers for encounters/reward branches, pre-finished rejection, load latency and export sizes |
| `tmp/content_review/probe_specific.py`, `fuzz_content.py --suite ...` | the content review's per-item regression checks and per-item fuzzing (with the navigation-cycle check) |
