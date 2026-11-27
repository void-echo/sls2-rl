"""Archetype labels for the cards of Slay the Spire 2 (card pools of version 0.113.1).

Each archetype has enablers (cards that produce a resource or condition) and payoffs (cards whose value depends on it).
The lists were read off the rules text of the cards and follow the archetype names used in community guides.
Figure 5 marks as "combination" cards the payoffs of the archetypes that need their enablers to work at all.
"""


def _ids(s):
    return {"CARD." + x.strip() for x in s.split() if x.strip()}


def has_v(key):
    return lambda desc: key in ((desc or {}).get("v") or {})


def has_tag(tag):
    return lambda desc: tag in ((desc or {}).get("tags") or [])


ARCH = {
    "Ironclad": {
        "力量多段": {"enablers": _ids("INFLAME DEMON_FORM FIGHT_ME RUPTURE BRAND DOMINATE SETUP_STRIKE"),
                  "payoffs": _ids("TWIN_STRIKE SWORD_BOOMERANG WHIRLWIND CONFLAGRATION THRASH DISMANTLE TEAR_ASUNDER SPITE")},
        "消耗": {"enablers": _ids("BURNING_PACT TRUE_GRIT SECOND_WIND FIEND_FIRE STOKE BRAND CINDER THRASH HAVOC DRUM_OF_BATTLE CORRUPTION"),
               "payoffs": _ids("FEEL_NO_PAIN DARK_EMBRACE ASHEN_STRIKE PACTS_END FORGOTTEN_RITUAL EVIL_EYE MIDNIGHT")},
        "格挡": {"enablers_pred": has_v("Block"), "members": _ids("STONE_ARMOR IMPERVIOUS FLAME_BARRIER CRIMSON_MANTLE RAGE"),
               "payoffs": _ids("BODY_SLAM JUGGERNAUT BARRICADE UNMOVABLE")},
        "易伤": {"enablers_pred": has_v("VulnerablePower"),
               "payoffs": _ids("CRUELTY BULLY DISMANTLE MOLTEN_FIST VICIOUS COLOSSUS DOMINATE")},
        "自伤": {"enablers": _ids("OFFERING BLOODLETTING HEMOKINESIS BREAKTHROUGH BLOOD_WALL BRAND CRIMSON_MANTLE INFERNO"),
               "payoffs": _ids("RUPTURE INFERNO SPITE TEAR_ASUNDER")},
        "打击": {"enablers_pred": has_tag("Strike"), "payoffs": _ids("PERFECTED_STRIKE HELLRAISER")},
    },
    "Silent": {
        "毒": {"enablers_pred": has_v("PoisonPower"), "members": _ids("NOXIOUS_FUMES CORROSIVE_WAVE ENVENOM HAZE OUTBREAK"),
              "payoffs": _ids("ACCELERANT BUBBLE_BUBBLE MIRAGE")},
        "小刀": {"enablers": _ids("BLADE_DANCE CLOAK_AND_DAGGER INFINITE_BLADES LEADING_STRIKE HIDDEN_DAGGERS STORM_OF_STEEL UP_MY_SLEEVE BLADE_OF_INK FAN_OF_KNIVES"),
               "payoffs": _ids("ACCURACY PHANTOM_BLADES KNIFE_TRAP FAN_OF_KNIVES FINISHER")},
        "弃牌奇巧": {"enablers": _ids("ACROBATICS SURVIVOR DAGGER_THROW PREPARED CALCULATED_GAMBLE HIDDEN_DAGGERS STORM_OF_STEEL TOOLS_OF_THE_TRADE HAND_TRICK MASTER_PLANNER"),
                 "payoffs_pred": lambda d: "Sly" in ((d or {}).get("kw") or []), "payoffs": _ids("TACTICIAN REFLEX")},
    },
    "Defect": {
        "闪电（电流相生）": {"enablers": _ids("ZAP BALL_LIGHTNING LIGHTNING_ROD STORM TEMPEST RAINBOW VOLTAIC"),
                     "payoffs": _ids("VOLTAIC THUNDER TESLA_COIL")},
        "冰霜": {"enablers": _ids("COLD_SNAP COOLHEADED GLACIER CHILL ICE_LANCE HIBERNATE RAINBOW"),
               "payoffs": _ids("HAILSTORM COOLANT")},
        "集中/激发": {"enablers": _ids("ZAP BALL_LIGHTNING LIGHTNING_ROD STORM TEMPEST RAINBOW VOLTAIC COLD_SNAP COOLHEADED GLACIER CHILL ICE_LANCE HIBERNATE NULL DARKNESS SHADOW_SHIELD CONSUMING_SHADOW FUSION METEOR_STRIKE CHAOS GLASSWORK REFRACT SPINNER"),
                  "payoffs": _ids("DEFRAGMENT BIASED_COGNITION FOCUSED_STRIKE HOTFIX SYNCHRONIZE LOOP CAPACITOR MULTI_CAST BARRAGE DUALCAST QUADCAST SHATTER COMPILE_DRIVER")},
        "爪击/0费": {"enablers_pred": lambda d: (d or {}).get("type") == "Attack" and (d or {}).get("cost") == 0,
                  "payoffs": _ids("CLAW ALL_FOR_ONE FERAL ONE_FOR_ALL SCRAPE")},
        "状态牌": {"enablers": _ids("BOOST_AWAY FIGHT_THROUGH OVERCLOCK GUNK_UP TURBO"),
                "payoffs": _ids("SMOKESTACK TRASH_TO_TREASURE ITERATION COMPACT FLAK_CANNON ROCKET_PUNCH")},
    },
    "Regent": {
        "铸剑": {"enablers_pred": lambda d: any(k in ((d or {}).get("v") or {}) for k in ("Forge", "CalculatedForge")),
               "payoffs": _ids("CONQUEROR SEEKING_EDGE SWORD_SAGE PARRY SUMMON_FORTH")},
        "辉星": {"enablers": _ids("GATHER_LIGHT GLOW HIDDEN_CACHE ROYAL_GAMBLE GENESIS VENERATE SOLAR_STRIKE SHINING_STRIKE KNOCKOUT_BLOW CONVERGENCE BIG_BANG THE_SEALED_THRONE"),
               "payoffs_pred": lambda d: ((d or {}).get("star") or -1) > 0,
               "payoffs": _ids("BLACK_HOLE CHILD_OF_THE_STARS RADIATE CRESCENT_SPEAR")},
        "生成牌": {"enablers": _ids("MANIFEST_AUTHORITY QUASAR BUNDLE_OF_JOY SPECTRUM_SHIFT COLLISION_COURSE CRASH_LANDING HEIRLOOM_HAMMER"),
                "payoffs": _ids("ARSENAL PILLAR_OF_CREATION SUPERMASSIVE")},
    },
    "Necrobinder": {
        "灾厄": {"enablers": _ids("BLIGHT_STRIKE COUNTDOWN DEATHBRINGER NEGATIVE_PULSE NO_ESCAPE SCOURGE OBLIVION REAPER_FORM"),
               "payoffs": _ids("END_OF_DAYS TIMES_UP DEATHS_DOOR SHROUD NO_ESCAPE")},
        "奥斯提": {"enablers": _ids("POKE FETCH FLATTEN SNAP SIC_EM RATTLE RIGHT_HAND_HAND HIGH_FIVE SPUR NECRO_MASTERY PULL_AGGRO AFTERLIFE REANIMATE INVOKE BODYGUARD"),
                "payoffs": _ids("CALCIFY SQUEEZE RATTLE PROTECTOR UNLEASH NECRO_MASTERY BONE_SHARDS SACRIFICE")},
        "灵魂": {"enablers": _ids("CAPTURE_SPIRIT DIRGE GLIMPSE_BEYOND GRAVE_WARDEN REAVE SEANCE SEVERANCE"),
               "payoffs": _ids("DEVOUR_LIFE HAUNT SOUL_STORM")},
        "虚无": {"enablers": _ids("CALL_OF_THE_VOID SCULPTING_STRIKE"),
               "payoffs": _ids("BANSHEES_CRY PULL_FROM_BELOW EIDOLON PAGESTORM SPIRIT_OF_ASH VEILPIERCER")},
    },
}


def is_enabler(arch, cid, desc):
    return cid in arch.get("enablers", set()) or ("enablers_pred" in arch and arch["enablers_pred"](desc))


def is_payoff(arch, cid, desc):
    return cid in arch.get("payoffs", set()) or ("payoffs_pred" in arch and arch["payoffs_pred"](desc))


def is_member(arch, cid, desc):
    return is_enabler(arch, cid, desc) or is_payoff(arch, cid, desc) or cid in arch.get("members", set())
