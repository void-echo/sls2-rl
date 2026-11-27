# RL protocol (engine ⇄ Python), protocol 1 (harness v1)

The engine (`engine/src/Sts2Headless`, see `README.md`) runs the real `sts2.dll` and reads one JSON object per line
on stdin, writing one JSON object per line on stdout. Lines on stdout that do not start with `{` are logs and must be
ignored. The first line is `{"type":"ready","version":"1.0.0","protocol":1}`.

The engine owns legality: every response lists the legal actions as `cands`, and Python only ever answers with an
index into that list. Each candidate is exactly one input the real client's UI accepts in that state (see the
decision table in `README.md`), so the agent is offered every choice the game offers and nothing else.

Protocol 1 is backward compatible with v0: the same commands, the same `obs` field names, the same candidate kinds
for the same situations. It adds decision kinds, candidate kinds and fields (marked **new** below); consumers that
embed unknown kinds and fields generically (as `rl/encode.py` does) keep working.

## No navigation cycles

An action that only navigates back to an identical earlier decision state, with no game-state change, is not offered,
provided everything the sub-screen showed is visible at the parent decision. Such an action adds nothing the agent
could not do otherwise, but under a deterministic policy (same state, same action) it loops forever. The reachable game
outcomes and the RNG use stay exactly the real game's, and the order of taking rewards stays free. Concretely:

- `rewards`: a card reward candidate carries the offered cards (`cards`) and the alternatives its screen will show
  (`alts`), so opening it is never needed just to look. `skip` on a card reward opened from the rewards screen, before
  anything else was done on that screen, **declines** the reward: it is no longer offered and is left behind at
  `proceed` exactly like a reward that was never opened (`SkipLocalRewardsSet` → `CardReward.OnSkipped`). Cancelling the
  selection of a card removal reward (Forbidden Grimoire) declines it the same way. After something changed on the
  screen (Driftwood's `reroll`, a potion), `skip` is the client's plain back-out: the reward stays (with the rerolled
  cards), but it is held back until the game changes again on the rewards screen (a reward taken, a potion used or
  discarded), because opening it at once would only show the screen just left. On a must-take-all screen
  (`ctx.must_take_all`) these screens have no `skip`.
- `card_select` opened by Smith, Cook (Meat Cleaver) or the merchant's card removal has no `skip`: cancelling would
  return to the identical rest site or shop, where the option is still offered. Nothing is lost, because the only other
  inputs on those screens are potions, which do the same at the rest site or the shop.
- `shop`: `buy_potion` is offered only when the purchase succeeds (a free slot, and procuring allowed, i.e. not with
  Sozu); otherwise the client's click only shows a failure message.
- Cancels and skips that change game state stay: choose-a-card skips, selections in combat, the chest's `skip`.

`tools/cycle_check.py` checks this property: it hashes (decision, obs, cands) at every step and flags any exact repeat
within one room / combat turn, under a random, a state-deterministic and a loop-seeking policy; the content fuzz suites
(`tmp/content_review/fuzz_content.py`) run the same check.

## Additions of 2026-10-28 (the observation audit)

Rule for every field (`docs/02_观测与编码方法论.md`): what a player who knows the game and remembers the run knows
at this decision (what the screen shows, what he has seen, numbers the UI computes for him), and nothing that is
not revealed yet (RNG state, draw order, unentered rooms, unopened chests, which of two look-alike moves was rolled).
The observation is built by `engine/src/Shared/ObsShared.cs`, which the real-client parity mod compiles too.
The field census (`docs/survey/state_census.md`, `state_census_classified.tsv`) classifies every field of the game's
state against this list; `dbg_census` (debug) lists them.

| where | field | meaning |
|---|---|---|
| `g` | `removals_used` | merchant card removals used this run (the next removal price) |
| `g` | `odds` | `{q_monster, q_shop, q_treasure, rare, potion}`: unknown-room odds, card rarity offset, potion drop chance; public functions of revealed results (`rare` can know a little more only when a rare roll is downgraded because the rare pool is empty) |
| `g` | `act_rooms` | this act's rooms so far, in order: `[{t: room type, id: encounter / event}]` |
| `relics[]` | `cnt` | the number on the relic bar (`DisplayAmount` when `ShowCounter`) |
| `relics[]` | `status` | `Active` (glowing) / `Disabled` (greyed out); absent when normal |
| `relics[]`, `cards`, `pw[]`, enemies | `d` | the object's own state: scalar mutable fields its class declares (counters, flags, modes, model references as ids, numeric dictionaries as `<name>.sum`, non-numeric `[SavedProperty]` values); cosmetic and TestMode-only fields excluded |
| `relics[]`, `pw[]` | `cards` | cards it holds: `id[+upgrades][@enchantment]` (Swipe's stolen card, Nightmare, Pael's Tooth) |
| `potions[]` | `v` | its tooltip numbers |
| card | `v` | also `saved.<Name>` numeric `[SavedProperty]` counters |
| card | `ench_d`, `ench_s` | the enchantment's own state; `Disabled` when spent (greyed out) |
| card | `replay` | "Replay X" on the card |
| card | `kw` | also `Retain` / `Sly` granted for this turn |
| card | `played_turn` | played this turn |
| card | `cost_mods`, `star_mods` | expiry of active cost changes: `ThisTurn` / `UntilPlayed` / `EndOfTurn` / `ThisCombat` |
| hand card | `glow` | `gold` / `red` (the card's glow conditions) |
| `pw[]` | `v`, `dn` | tooltip vars; the icon number when it differs from `n` |
| `combat` | `enemies` | every enemy still in the fight, including the dead that stay to revive (`alive: false`); `tgt` indices refer to this list |
| enemy | `move` | id of the move it performs this enemy turn, when its pattern, its performed moves and the shown intents identify it |
| enemy | `then` | the move after it when the pattern fixes it |
| enemy | `prev`, `done` | last three performed moves (newest first); every move performed this combat, counted |
| enemy | `slot` | its position slot |
| `intents[]` | `n` | status card count of a StatusCard intent |
| `combat` | `turn` | this turn: `first` (cards played, replays not counted), `attacks`, `skills`, `powers`, `last_attack`, `status_drawn`, `drawn_extra`, `exhausted`, `discarded`, `hp_lost`, `energy_spent` |
| `combat` | `pturn`, `enc` | the player's turn number (differs from `round` after an extra turn); the encounter id |
| `combat` | `draw_top` | cards on top of the draw pile whose order the player knows, top first (put there in plain sight; a random insertion or a reshuffle clears it); `draw` itself is sorted by the full card |
| `orbs[]` | `slot` | queue position (0 = evoked next) |
| `ctx` (shop, Fake Merchant) | `unavailable` | stocked items that are not candidates: `[{k: card/relic/potion, card / id, price}]` |
| `ctx` (rewards) | `blocked` | potion rewards the full belt blocks (takeable after a discard) |
| `play` cand | `pv` | the card's values aimed at that target; for all/random-enemy cards the multi-target preview |
| `rest` cand | `heal` | the heal amount on the button (before the max-HP cap) |
| `option` cand | `relic_d` | the offered relic's own state (e.g. the character a Sea Glass offers) |

Removed: Trial's `EntrantNumber` from option `v` (a flavour number from `Rng.Chaotic`).

## Commands

- `{"cmd":"rl_start","character":"Ironclad","ascension":0,"seed":"abc"}` starts a new run and returns a response `R`.
  Characters: `Ironclad`, `Silent`, `Defect`, `Regent`, `Necrobinder`; ascension 0–10. Calling `rl_start` again in the
  same process starts a new run; the previous run is torn down first and the new run is bit-identical to the same
  seed in a fresh process (tested by `tools/determinism.py`), so there is no need to restart processes between runs.
- `{"cmd":"rl_act","i":3}` executes `cands[3]` of the previous response and returns the next `R`.
- `{"cmd":"rl_save_get","id":k}` returns `{"type":"save","id":k,"kind":"room"|"finished"|"event","floor":n,"room":T,"data":"..."}`.
  `data` is an opaque JSON string containing the game's serialized run and a process-state snapshot, portable to
  another engine process of the same version. `floor` is the act floor; `room` is the map point type for `room`
  saves (including `Unknown` before its room is rolled), or the completed room type otherwise.
- `{"cmd":"rl_load","id":k}` or `{"cmd":"rl_load","data":"..."}` restores a **room-entry** snapshot and returns a normal `R`.
  A load by id uses this process's saved snapshots; a load by data imports the snapshot and its id (assigning a fresh
  id if that id already names a different snapshot). Later saves
  continue above the highest id recorded/imported, including after branching; `rl_start` clears them and restarts at 1.
  `finished` and `event` saves return `type:error` without changing the current decision: the client re-enters
  pre-finished rooms and can repeat entry effects (e.g. Maw Bank grants gold again). To reach a rewards screen,
  shop, rest site, event or map decision, load its room-entry snapshot and replay the inputs since entry.
  Client serialization can also omit transient fields visible in raw observations (Ornamental Fan's old turn
  counter outside combat; see `README.md`); the strict replay checker reports these differences.
  Both commands are training/evaluation tools, **never candidate actions**.
- `{"cmd":"quit"}` exits.

Scenario / debug commands (tests only, never used in training; each returns a normal `R`):

| command | effect |
|---|---|
| `{"cmd":"rl_enter_room","type":T,"encounter":ID,"event":ID,"at":"boss"\|"boss2","floor":n}` (alias `enter_room`) | `RunManager.EnterRoomDebug` (the game's dev path) into a `Monster`/`Elite`/`Boss`/`Shop`/`Rest`/`Event`/`Treasure` room. `at` first marks the act's boss (or second boss) node as visited, so the boss's proceed behaves as on the map. |
| `{"cmd":"set_player","hp":..,"max_hp":..,"gold":..,"relics":[ids],"deck":[ids],"potions":[ids]}` | overwrite player state directly (relics attached without their pickup effects) |
| `{"cmd":"obtain_relics","ids":[...]}` / `{"cmd":"dbg_obtain_relic","id":..}` | `RelicCmd.Obtain` (pickup effects run; their choices surface as decisions) |
| `{"cmd":"dbg_procure_potion","id":..}` | `PotionCmd.TryToProcure` |
| `{"cmd":"dbg_add_card","id":..}` | `CardPileCmd.Add` to the deck |
| `{"cmd":"dbg_enter_act","act":2}` | `RunManager.EnterAct(act-1)` |
| `{"cmd":"dbg_obs"}` | re-observe the current decision |
| `{"cmd":"dbg_info"}` | diagnostics (`{"type":"info",...}`: room, prompts, game-over flags, acts, ...) |

IDs may be given with or without the category prefix (`ORRERY` or `RELIC.ORRERY`).

## Response `R`

```
{
  "type": "rl",
  "done": false,                 // true once the run is over; then cands == []
  "victory": null,               // true/false when done
  "decision": "combat_play",     // see below
  "obs": { ... },
  "cands": [ {...}, ... ],       // legal actions, never empty unless done
  "info": {"act":1, "floor":3, "total_floor":3, "hp":70, "max_hp":80, "gold":120, "steps":57,
           "abandoned":false,     // new: true when the run ended by abandoning (Trial's Double Down)
           "save":{"id":1,"kind":"room","floor":1,"room":"Ancient"}, // latest save, or null
           "t_exec_ms":..., "t_obs_ms":..., "warn":null}
}
```

`info.save.kind` is `room` for the save before room generation/entry, `finished` for a pre-finished combat,
and `event` for a completed ancient event. Every game save is numbered and recorded, even when it cannot be loaded.

**Terminal integrity.** A run ends (`done:true`) only as
- a death by the game's rules: every player creature at 0 HP (`RunState.IsGameOver`), `victory:false`, `info.hp == 0`;
- an abandon: Trial's Double Down (the client's abandon-run confirmation, `RunManager.Abandon`), `victory:false`,
  `info.abandoned:true` (the game also kills the player, so `hp == 0`);
- a victory: `RunManager.WinRun` in the last act (after The Architect), `victory:true`.

Anything else that looks terminal, a decision with no legal action, a stall (the game waits on nothing the harness
knows), game code running on a foreign thread, or an exception anywhere in game code is returned as
`{"type":"error","message":...}` with diagnostics. Treat an error as an aborted episode (no terminal reward); the
next `rl_start` in the same process is safe.

## Decisions

| decision | when | candidate kinds |
|---|---|---|
| `combat_play` | player turn, play phase, queue idle | `play`, `potion`, `discard_potion`, `end_turn` |
| `card_select` | any `CardSelectCmd` selection (hand, piles, deck, grid, choose-a-card), including ones at turn start or during the enemy turn | `pick`, `confirm`, `skip`, potion kinds |
| `bundle_select` | choose-a-bundle (Scroll Boxes) | `bundle`, potion kinds |
| `card_reward` | the card reward screen of a card reward | `reward_card`, `skip`, **`reroll`**, **`sacrifice`**, **`alt`**, potion kinds |
| **`rewards`** | a rewards screen (after combat, from events, relics, the chest, the Crystal Sphere) | **`reward`**, `proceed`, potion kinds |
| `event_choice` | event / ancient options | `option`, potion kinds |
| **`fake_merchant`** | the Fake Merchant event's shop | `buy_relic`, `leave`, potion kinds (Foul Potion is thrown with `potion`) |
| `rest_site` | rest site | `rest`, `leave` (once an option succeeded), potion kinds |
| `shop` | merchant | `buy_card`, `buy_relic`, `buy_potion`, `remove`, `leave`, potion kinds |
| `treasure` | chest room: before opening (`ctx.opened:false`), then the relic choice (`ctx.opened:true`) | **`open`**; then `treasure`, `skip`; potion kinds |
| `crystal_sphere` | Crystal Sphere minigame | `sphere`, potion kinds |
| `map_select` | the map is open with travel enabled | `map`, potion kinds |
| `game_over` | `done:true` | none |

The v1 engine never emits `unknown`. A finished event (the client's synthetic PROCEED, which only opens the map), a
proceeded rewards screen and a left room are all `map_select`.

## Observation `obs`

All categorical values are strings (model IDs such as `CARD.STRIKE_IRONCLAD`, enum names such as `Attack`). Numeric
dictionaries (`v`) map a variable name to a number.

```
obs = {
  "g": {"act":1, "floor":3, "total_floor":3, "asc":0, "gold":120, "hp":70, "max_hp":80, "char":"CHARACTER.IRONCLAD",
        "potion_slots":3, "room":"Monster", "boss":"ENCOUNTER....",
        "boss2": null | "ENCOUNTER...",       // new: second boss of the act (A10, last act)
        "act_id": "ACT.UNDERDOCKS"},           // new: the act variant
  "deck":    [card, ...],
  "relics":  [{"id":..., "v":{...}}, ...],     // v: numeric dynamic vars + numeric [SavedProperty] as "saved.<Name>"
  "potions": [{"id":..., "slot":1}, ...],      // filled slots only; slot = the real belt index (gaps possible)
  "combat":  null | {
      "round":1, "played":0,         // played (new): cards the player started playing this turn, autoplays included
      "energy":3, "max_energy":3, "stars":0,
      "hand": [card + {"playable":bool, "pv":{...}}], "draw": [card...], "discard": [...], "exhaust": [...],
      "player": creature, "allies": [creature...],
      "enemies": [creature + {"intents":[{"t":"Attack","dmg":6,"hits":2}, ...],
                              "untargetable": true}],          // new, only when the enemy cannot be targeted now
      "orbs": [{"id":..., "passive":3, "evoke":8}], "orb_slots":3
  },
  "map": {"nodes":[{"c":3,"r":1,"t":"Monster", "q":[ids]}, ...],   // q (new): quest markers (Fur Coat, Spoils Map)
          "edges":[[0,5], ...], "cur":-1},                          // the second boss node is included and linked
  "ctx": {...}                                                      // per decision, below
}
card     = {"id":..., "up":0, "cost":1, "x":false, "star":0, "type":"Attack", "rar":"Basic", "tgt":"AnyEnemy",
            "v":{"Damage":6}, "kw":["Exhaust"], "ench":null, "ench_n":0, "aff":null, "aff_n":0}
creature = {"id":..., "hp":13, "max_hp":13, "block":0, "alive":true, "pw":[{"id":..., "n":2}, ...]}
```

`enemies` lists every enemy still in the fight in combat order, including dead enemies waiting to revive (`alive: false`); `tgt` indices refer to this list. The draw pile is sorted (its order is hidden).

`ctx` by decision:

| decision | ctx |
|---|---|
| `card_select` | `min`, `max`, `picked` (cards), new: `sel` (entry kind: `hand`, `hand_discard`, `hand_upgrade`, `combat_pile`, `deck_upgrade`, `deck_transform`, `deck_enchant`, `deck_remove`, `deck`, `grid`, `grid_rewards`, `choose_a_card`), `cancelable`, `source` (model id), `pile`, `prompt` (loc key) |
| `card_reward` | new: `alts` (alternative option ids, e.g. `Skip`, `REROLL`, `SACRIFICE`) |
| `rewards` | new: `terminal` (combat room screen), `must_take_all` (skipping disallowed) |
| `event_choice` | `event`; new: `sv` (string vars of the event, e.g. Slippery Bridge's card name). `sv` values, and the option `key`s of events whose options are generated from text (Doll Room), are display text: localization keys here, translated text in the real client. They are not game state. |
| `fake_merchant` | `event` |
| `shop` | `removal_cost` |
| `rest_site` | new: `done` (an option already succeeded) |
| `treasure` | new: `opened` |
| `crystal_sphere` | `divinations`, `w`, `h`, `cells` [{x, y, hidden, item}] (only what the screen shows) |

## Candidates `cands`

| k | fields | meaning (the UI input) |
|---|---|---|
| `play` | `hand`, `tgt` (enemy index or -1) | `CardModel.TryManualPlay(target)`; one candidate per targetable enemy for single-target cards (`Hook.ShouldAllowTargeting`, hittable) |
| `potion` | `slot`, `tgt` | `PotionModel.EnqueueManualUse(target)`; offered by `NPotionPopup`'s rules: AnyTime potions in every decision, CombatOnly ones on the player's side of combat outside selection screens; not while the belt is locked (`CanUseOrRemovePotions`) or the potion is already queued. Foul Potion is usable at the shop and the Fake Merchant |
| `discard_potion` | `slot` | `DiscardPotionGameAction` (any decision, belt not locked) |
| `end_turn` | | `EndPlayerTurnAction` |
| `map` | `node` | `VoteForMapCoordAction`; nodes per `NMapScreen.RecalculateTravelability` + `MapTravel` (only the start node at act start, the boss from the last row, the second boss after the first, free travel with Winged Boots / Flight) |
| `pick` | `card`, `src` | add a card to the selection (any order, no repeats); reaching `max` submits |
| `confirm` | | submit the picked cards (offered once `min` is reached) |
| `skip` | | card_select: cancel (only `cancelable` selections, and not for Smith / Cook / the merchant's removal; for a removal reward it declines the reward) or skip a choose-a-card screen that allows it; card_reward: the Skip alternative (opened from a rewards screen: declines the reward, or after a reroll / potion backs out with the reward held back, see "No navigation cycles"); treasure: leave the relic |
| `bundle` | `cards` | pick a bundle |
| `reward_card` | `card` | take that card from the card reward |
| **`reroll`** / **`sacrifice`** / **`alt`** (`key`) | | card reward alternatives (Driftwood's REROLL, Pael's Wing's SACRIFICE, any other id) |
| **`reward`** | `idx`, `rt` (`Gold`, `Potion`, `Relic`, `Card`, `SpecialCard`, `RemoveCard`, `Linked`), `id` (relic/potion), `n` (gold, number of cards), `rar`, `card` (special card), **`cards`** (rt `Card`: the offered cards), **`alts`** (rt `Card`: the alternatives its screen shows, e.g. `["Skip","REROLL"]`) | `RewardsSetSynchronizer.SelectLocalReward` (a card reward opens `card_reward`, a removal reward opens `card_select`); a potion reward is not offered while the belt is full; declined and held-back rewards are not offered |
| `proceed` | | rewards screen: leave (remaining rewards are skipped; for a boss: next act / second boss) |
| `option` | `key`, `v`, new: `relic`, `tips` (hover-tip model ids), `proceed` | `EventSynchronizer.ChooseLocalOption` (locked and already-spent options are not offered) |
| `rest` | `key` | `RestSiteSynchronizer.ChooseLocalOption` (enabled options only) |
| `buy_card` (`card`, `price`), `buy_relic` (`id`, `price`), `buy_potion` (`id`, `price`), `remove` (`price`) | | `MerchantEntry.OnTryPurchaseWrapper` (removal: once per shop, not cancelable; a potion only when it can be procured) |
| `treasure` | `id` | take the chest relic (`PickRelicLocally`) |
| **`open`** | | open the chest (gold and extra rewards first, then the relic) |
| `leave` | | leave the room (opens the map) |
| `sphere` | `tool`, `x`, `y` | Crystal Sphere: `SetTool` + `CellClicked` on a hidden cell |

Known choices the client offers that are deliberately not candidates: abandoning the run from the pause menu,
closing the map to look at an already-proceeded rewards screen again (a no-op), deselecting picked cards (any set is
reachable by picking in another order), inspecting/hovering, and the navigation-only actions of "No navigation
cycles" (re-opening a reward screen unchanged, cancelling Smith / Cook / the merchant's removal, a potion purchase
that cannot succeed).
