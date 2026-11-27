def parse_force_list(value):
    return {x.strip() for x in value.split(",") if x.strip()}


def _listed(card, ids, variables):
    return bool(card) and (card.get("id") in ids or bool(variables.intersection(card.get("v") or {})))


def force_matches(response, ids, variables):
    """Per candidate: taking it is forced. Card rewards and shops: the listed cards themselves. The rewards screen: a card
    reward item that shows a listed card must be opened (the item lists its cards, so without this a policy trained with
    forced picks learns to leave such items unopened: 11.8 card rewards opened per game instead of 19.5, 11-17)."""
    decision, cands = response.get("decision"), response.get("cands") or []
    if decision in ("card_reward", "shop"):
        return [c.get("k") in ("reward_card", "buy_card") and _listed(c.get("card"), ids, variables) for c in cands]
    if decision == "rewards":
        return [c.get("k") == "reward" and c.get("rt") == "Card" and any(_listed(x, ids, variables) for x in c.get("cards") or [])
                for c in cands]
    return [False] * len(cands)


def force_choice(matches, probabilities):
    hits = [i for i, match in enumerate(matches) if match]
    return max(hits, key=lambda i: float(probabilities[i])) if hits else None
