import copy
import gzip
import json

import numpy as np
import pytest

from rl.behavior_metrics import compute_metrics, load_traces, rate, sensitivity


def step(decision, cands, choice=0, hp=100, floor=1, act=1, room="Event", **extra):
    r = {"type": "step", "seed": "pref_0", "step": floor, "decision": decision, "cands": cands, "choice": choice,
         "info": {"hp": hp, "max_hp": 100, "act": act, "total_floor": floor},
         "state": {"g": {"hp": hp, "max_hp": 100, "act": act, "total_floor": floor, "room": room}}}
    r.update(extra)
    return r


def run(rows, act=1, floor=10, victory=False, deck=None, aborted=False):
    ep = {"type": "episode", "seed": "pref_0", "victory": victory, "info": {"act": act, "total_floor": floor},
          "obs": {"g": {"act": act, "total_floor": floor}, "deck": deck or []}}
    if aborted:
        ep["aborted"] = "error"
    return {"seed": "pref_0", "steps": rows, "episode": ep}


def test_wilson_and_empty_samples():
    assert rate(5, 10) == {"k": 5, "n": 10, "rate": .5, "ci95": pytest.approx([.23658959, .76341041])}
    assert rate(0, 1)["ci95"] == pytest.approx([0, .79345671])
    assert rate(1, 1)["ci95"] == pytest.approx([.20654329, 1])
    assert rate(0, 31)["ci95"][0] == 0
    assert rate(31, 31)["ci95"][1] == 1
    assert rate(0, 0) == {"k": 0, "n": 0, "rate": None, "ci95": [None, None]}
    result = compute_metrics([])
    assert result["overall"]["floor"] == {"n": 0, "total": 0, "mean": None}
    json.dumps(result, allow_nan=False)


def test_rest_bins_entry_hp_and_action_distribution():
    options = [{"k": "rest", "key": "HEAL"}, {"k": "rest", "key": "SMITH"}, {"k": "rest", "key": "DIG"}]
    rows = [step("rest_site", options, choice=i, hp=hp, floor=f)
            for f, (hp, i) in enumerate([(39, 0), (40, 1), (69, 0), (70, 1), (99, 0), (100, 0), (100, 1), (100, 2)], 1)]
    rows += [step("rest_site", [{"k": "potion", "slot": 2}], hp=100, floor=9),
             step("rest_site", options, hp=70, floor=9),
             step("rest_site", [{"k": "leave"}], floor=9)]
    result = compute_metrics([run(rows)])["rest_site"]
    assert result["choices"]["HEAL"] == rate(5, 9)
    assert result["choices"]["SMITH"] == rate(3, 9)
    assert result["choices"]["DIG"] == rate(1, 9)
    for name, heal, smith, n in [("<40%", 1, 0, 1), ("40-70%", 1, 1, 2), ("70-99%", 1, 1, 2), ("full", 2, 1, 4)]:
        assert result["hp_bins"][name] == {"heal": rate(heal, n), "smith": rate(smith, n)}
    assert result["full_hp_heal"] == rate(2, 4)


def test_elite_entries_map_choices_and_sphere():
    nodes = [{"t": "Elite"}, {"t": "Monster"}]
    rows = [step("map_select", [{"k": "map", "node": 0}, {"k": "map", "node": 1}], choice=i, floor=i)
            for i in (0, 1)]
    rows.append(step("map_select", [{"k": "map", "node": 0}], floor=2))
    for r in rows:
        r["state"]["map_nodes"] = nodes
    rows += [step("combat_play", [{"k": "end_turn"}], floor=f, room="Elite") for f in (3, 3, 5)]
    rows += [step("crystal_sphere", [{"k": "sphere", "tool": t}]) for t in ("small", "big", "small")]
    rows.append(step("crystal_sphere", [{"k": "discard_potion", "slot": 0}]))
    result = compute_metrics([run(rows), run([]), run(rows, aborted=True)])
    assert result["elite"]["entries_per_run"] == {"n": 2, "total": 2, "mean": 1}
    assert result["elite"]["choice_when_available"] == rate(4, 6)
    assert result["crystal_sphere"]["big_tool"] == rate(2, 6)


def test_potion_ids_slots_full_hp_and_combat():
    rows = []
    for pid, hp, combat in [("POTION.BLOOD_POTION", 100, True), ("REGEN_POTION", 99, True),
                            ("POTION.FRUIT_JUICE", 100, False), ("POTION.FIRE_POTION", 100, False)]:
        r = step("combat_play" if combat else "event_choice", [{"k": "potion", "slot": 4}], hp=hp)
        r["potion"] = {"id": pid, "slot": 4}
        r["state"]["combat"] = {"round": 1} if combat else None
        rows.append(r)
    # A compact legacy list cannot identify a gapped slot; full obs can.
    rows[2].pop("potion")
    rows[2]["obs"] = copy.deepcopy(rows[2]["state"])
    rows[2]["obs"]["potions"] = [{"id": "POTION.FRUIT_JUICE", "slot": 4}]
    r = step("event_choice", [{"k": "potion", "slot": 4}])
    r["state"]["potions"] = ["POTION.BLOOD_POTION"]
    rows.append(r)
    rows.append(step("event_choice", [{"k": "discard_potion", "slot": 4}]))
    result = compute_metrics([run(rows)])["potions"]
    assert result["healing_at_full_hp"] == rate(2, 3)
    assert result["outside_combat"] == rate(3, 5)
    assert result["identified"] == rate(4, 5)


def test_rewards_decks_removal_submission_and_overall():
    cards = [{"id": "CARD.A", "type": "Attack", "v": {"Block": 2}},
             {"id": "CARD.S", "type": "Skill", "v": {"Block": 0}},
             {"id": "CARD.P", "type": "Power", "v": {}},
             {"id": "CARD.S2", "type": "Skill", "v": {"Damage": 3}}]
    options = [{"k": "reward_card", "card": c} for c in cards[:3]] + [{"k": "skip"}, {"k": "reroll"}]
    rows = [step("card_reward", options, choice=i) for i in range(5)]
    for action, picked, maximum in [("pick", [], 1), ("pick", [], 2), ("pick", [cards[0]], 2),
                                     ("confirm", [cards[0]], 2), ("skip", [cards[0]], 2)]:
        r = step("card_select", [{"k": action}])
        r["state"]["ctx"] = {"sel": "deck_remove", "max": maximum, "picked": picked}
        rows.append(r)
    rows.append(step("shop", [{"k": "remove"}]))
    upgrade = step("card_select", [{"k": "pick"}])
    upgrade["state"]["ctx"] = {"sel": "deck_upgrade", "max": 1}
    rows.append(upgrade)
    runs = [run(rows, act=3, floor=50, victory=True, deck=cards),
            run([], act=2, floor=22, deck=[cards[1], cards[1]]),
            run([], act=1, floor=5, deck=[cards[0]]), run(rows, act=3, floor=99, aborted=True)]
    result = compute_metrics(runs)
    assert result["card_reward"]["skip"] == rate(2, 8)
    for t in ("Attack", "Skill", "Power"):
        assert result["card_reward"]["picked_types"][t] == rate(2, 6)
    assert result["deck"]["size"] == {"n": 3, "total": 7, "mean": 7 / 3}
    assert result["deck"]["defensive_skills"] == {"n": 3, "total": 3, "mean": 1}
    assert result["deck"]["removals_per_run"] == {"n": 3, "total": 4, "mean": 4 / 3}
    assert result["overall"]["win"] == rate(1, 3)
    assert result["overall"]["floor"] == {"n": 3, "total": 77, "mean": 77 / 3}
    assert result["overall"]["passed_act1_boss"] == rate(2, 3)
    assert result["overall"]["passed_act2_boss"] == rate(1, 3)
    assert result["samples"]["aborted"] == 1


def test_plain_gzip_duplicate_seeds_and_legacy_deck(tmp_path):
    r = step("card_reward", [{"k": "skip"}])
    r["state"]["deck"] = [{"id": "CARD.DEFEND", "type": "Skill"}]
    ep = {"type": "episode", "seed": "pref_0", "victory": False, "info": {"act": 1, "total_floor": 9},
          "final": {"combat": {"round": 1}}}
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl.gz"]
    for path, opener in zip(paths, (open, gzip.open)):
        with opener(path, "wt") as f:
            for row in ({"type": "meta", "update": 42}, r, ep):
                f.write(json.dumps(row) + "\n")
    meta, runs = load_traces(paths)
    result = compute_metrics(runs, meta)
    assert len(meta) == len(runs) == 2
    assert result["card_reward"]["skip"] == rate(2, 2)
    assert result["deck"]["size"]["mean"] == 1
    assert result["deck"]["sources"] == {"last_observed": 2}
    assert result["deck"]["defensive_skills"]["n"] == 0


def test_boss_sensitivity_variant_grouping_tv_flips_and_hp_control():
    rows = []
    for d, boss, variant in [("card_reward", "A", "X"), ("card_reward", "B", "X"), ("card_reward", "C", "X"),
                             ("rest_site", "D", "Y"), ("map_select", "A", "X"), ("shop", "A", "X")]:
        r = step(d, [{"k": "a"}, {"k": "b"}])
        r["obs"] = copy.deepcopy(r["state"])
        r["obs"]["g"].update(boss=boss, act_id=variant)
        rows.append(r)
    rows.append(step("map_select", [{"k": "a"}, {"k": "b"}]))
    original = copy.deepcopy(rows)

    def predict(records):
        out = []
        for r in records:
            g = r["obs"]["g"]
            assert g["boss"] in ({"A", "B", "C"} if g["act_id"] == "X" else {"D"})
            p = {"A": .8, "B": .2, "C": .5, "D": .8}[g["boss"]]
            if g["hp"] < 100:
                assert g["hp"] == 70
                p -= .1
            out.append(np.array([p, 1 - p]))
        return out

    result = sensitivity([run(rows)], predict, batch_size=2)
    card = result["by_decision"]["card_reward"]
    assert card["boss_states"] == 3
    assert card["boss"]["tv"]["n"] == 6
    assert card["boss"]["tv"]["mean"] == pytest.approx(.4)
    assert card["boss"]["argmax_flip"] == rate(4, 6)
    assert card["hp_minus_30pct"]["tv"]["mean"] == pytest.approx(.1)
    assert card["hp_minus_30pct"]["argmax_flip"] == rate(1, 3)
    assert result["by_decision"]["map_select"]["missing_obs"] == 1
    rest = result["by_decision"]["rest_site"]
    assert rest["no_alternative_boss"] == 1
    assert rest["boss"]["tv"]["mean"] is None
    assert rest["hp_minus_30pct"]["tv"]["n"] == 1
    assert result["all"]["boss"]["tv"]["n"] == 10
    assert result["all"]["hp_minus_30pct"]["tv"]["n"] == 6
    assert rows == original
    json.dumps(result, allow_nan=False)
