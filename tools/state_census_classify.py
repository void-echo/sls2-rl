"""Classify every census row (tmp/obs_audit/census.tsv) as E/H/D/K/G with a one-line reason.

  python3 tmp/obs_audit/census/classify.py   ->  tmp/obs_audit/census/census_classified.tsv

Keys are "<type without MegaCrit.Sts2.Core.>::<field without k__BackingField>". Every row must have an entry
(the script fails otherwise). Reasons are Chinese; ids and code names stay English.
Cross references: R# = docs/survey/obs_audit_relics_cards_screens.md finding #, P = obs_audit_powers.md,
M-F# = obs_audit_monsters.md finding #, G# = gap ids in state_census_classification.md.
"""
import csv
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "census.tsv")
OUT = os.path.join(HERE, "census_classified.tsv")

C = {}


def add(t, fields, cls, why):
    for f in fields.split():
        key = f"{t}::{f}"
        assert key not in C, key
        C[key] = (cls, why)


# ---------------------------------------------------------------- combat core
add("Combat.CombatId", "<Value>", "K", "战斗内 CombatId 序号，进程级计数器，基础设施")
add("Combat.CombatManager", "<DebugForcedTopCardOnNextShuffle>", "K", "调试钩子，恒 null")
add("Combat.CombatManager", "<History>", "E", "战斗历史容器：出牌数 combat.played、敌人 prev 由它算出；聚合缺口见 G4")
add("Combat.CombatManager", "<IsPaused> <StateTracker> _cardOrPotionEffectDepth _playerActionsDisabled _turnLoopTask _turnLoopWaitingForPreviousSource _turnState",
    "K", "回合循环/暂停/同步的运行时基础设施，不是局面")
add("Combat.CombatState", "<BadgeModels>", "K", "成就徽章模型")
add("Combat.CombatState", "<CurrentSide>", "K", "决策点上恒为 Player")
add("Combat.CombatState", "<Modifiers>", "K", "自定义/每日模式修改器，标准模式恒空；若启用需导出")
add("Combat.CombatState", "<MultiplayerScalingModel>", "K", "多人数值缩放，单人恒定")
add("Combat.CombatState", "<RoundNumber>", "E", "combat.round")
add("Combat.CombatState", "<RunState>", "K", "回指 RunState")
add("Combat.CombatState", "_allCards", "K", "战斗内卡牌注册表；内容经各牌堆导出")
add("Combat.CombatState", "_allies", "E", "combat.allies（Osty）；Byrdpip/PaelsLegion 宠物是 9999 血的装饰，不导出无妨")
add("Combat.CombatState", "_encounter", "G", "G11（低）：遭遇 id 未导出（召唤受空槽位限制、遭遇决定金币奖励），见 M-F15")
add("Combat.CombatState", "_enemies", "G", "G1：只导出存活敌人（Harness.AliveEnemies 按 IsAlive 过滤）；死而未离场、等待复活的敌人整只消失，见 M-F1/P")
add("Combat.CombatState", "_escapedCreatures", "D", "逃跑的怪物（带走金币/牌），屏幕上是公开事件；之后只影响战后金币比例，奖励界面可见")
add("Combat.CombatState", "_nextCreatureId", "K", "生物 id 计数器")
add("Combat.CombatStateTracker", "_combatManager _combatStateChangedDeferredTask _state", "K", "UI 刷新用的状态跟踪器")
add("Combat.CombatTurnState", "<BeginEnemyTurnSignalSource> <EndTurnSignalSource> <EndingPlayerTurnPhaseOne> <EndingPlayerTurnPhaseTwo> <Id> <IsEnemyTurnStarted> <IsInProgress> <IsStarting> <PendingLoss> <PlayersReadyToBeginEnemyTurn> <PlayersReadyToEndTurn> <ReadyLock> <State> _cts",
    "K", "回合状态机/多人同步的信号量与锁，决策点上取值固定")
add("Combat.CombatTurnState", "<PlayersTakingExtraTurn>", "D", "额外回合标记；下一次结束回合是否给额外回合可由 AmbergrisPower 的 pw.n 看出（CombatManager.cs:1829-1838）")

# ---------------------------------------------------------------- combat history
add("Combat.History.CombatHistory", "_entries", "E", "公开历史日志；导出了 combat.played 与敌人 prev，手牌 pv 含历史计算值；未导出的聚合量见 G4")
add("Combat.History.CombatHistoryEntry", "<Actor> <CurrentSide> <History> <RoundNumber> _playerTurnNumbers", "D",
    "历史条目的公共头（谁、哪一轮）；用途见各条目")
add("Combat.History.Entries.BlockGainedEntry", "<Amount> <CardPlay> <Props>", "D",
    "仅 UnmovablePower 读（本回合第几次出牌得格挡），其效果在手牌 pv 里；公开历史")
add("Combat.History.Entries.CardAfflictedEntry", "<Affliction> <Card>", "D",
    "ChainsOfBindingPower 读；其计数已在 pw.d.boundCardPlayed，牌上 aff 已导出")
add("Combat.History.Entries.CardDiscardedEntry", "<Card>", "D", "MementoMori 的本回合弃牌数，手牌 pv 已含")
add("Combat.History.Entries.CardDrawnEntry", "<Card>", "G",
    "G4（低）：IterationPower 需要本回合抽到的状态牌数（IterationPower.cs:22），未导出；Murder/DeathMarch 的抽牌数在手牌 pv 里")
add("Combat.History.Entries.CardDrawnEntry", "<FromHandDraw>", "D", "DeathMarch 区分回合抽牌，手牌 pv 已含")
add("Combat.History.Entries.CardExhaustedEntry", "<Card>", "G",
    "G4：Evil Eye 的'本回合消耗过牌'条件只体现为金光（EvilEye.cs:19,25），未导出；Midnight 的本场消耗数也不在牌堆中的牌上")
add("Combat.History.Entries.CardGeneratedEntry", "<Card> <Creator>", "D", "Supermassive 的本场生成数，手牌 pv 已含")
add("Combat.History.Entries.CardPlayFinishedEntry", "<CardPlay>", "G",
    "G4：按类型的本回合出牌数（Make It So 回手 MakeItSo.cs:40、Pinpoint、RippleBasin、PaleBlueDot）、"
    "本回合打过哪张牌（Bolas/ThrummingHatchet 下回合回手、Fetch）、本回合最后一张攻击（HistoryCourse.cs:21）都未导出；只有总数 played")
add("Combat.History.Entries.CardPlayFinishedEntry", "<WasEthereal>", "D", "BansheesCry/PullFromBelow 的虚无出牌数，手牌 pv 含")
add("Combat.History.Entries.CardPlayStartedEntry", "<CardPlay>", "E",
    "combat.played = 本回合 CardPlaysStarted 计数（含重放）；EchoForm 需要的 IsFirstInSeries 计数见 G4/P")
add("Combat.History.Entries.CreatureAttackedEntry", "<DamageResults>", "G",
    "G4（低）：Flatten 的'Osty 本回合攻击过'只体现为金光（Flatten.cs:27）；Rattle 的段数在 pv")
add("Combat.History.Entries.DamageReceivedEntry", "<CardSource> <Dealer>", "D", "GangUp/BeatIntoShape 的计数在手牌/目标 pv 里")
add("Combat.History.Entries.DamageReceivedEntry", "<Result>", "G",
    "G4：'本回合掉过血'（Spite.cs:48 金光条件）未导出；EmotionChip 看上回合，已在回合开始结算")
add("Combat.History.Entries.EnergySpentEntry", "<Amount>", "D", "HelixDrill 的本回合花费能量，手牌 pv 已含")
add("Combat.History.Entries.MonsterPerformedMoveEntry", "<Monster> <Move>", "E", "敌人 prev（最近 3 招）与 move/then 的推断依据")
add("Combat.History.Entries.MonsterPerformedMoveEntry", "<Targets>", "D", "招式目标，单人恒为玩家")
add("Combat.History.Entries.OrbChanneledEntry", "<Orb>", "D", "Voltaic 的闪电充能数，手牌 pv 已含")
add("Combat.History.Entries.PotionUsedEntry", "<Potion> <Target>", "K", "无玩法读取者（仅统计）")
add("Combat.History.Entries.PowerReceivedEntry", "<Amount> <Applier>", "D", "公开历史；效果体现在 pw.n")
add("Combat.History.Entries.PowerReceivedEntry", "<Power>", "G",
    "G4（低）：Death's Door 的'本回合施加过 Doom'只体现为金光（DeathsDoor.cs:30）")
add("Combat.History.Entries.StarsModifiedEntry", "<Amount>", "D", "Radiate 的本回合获得星数，手牌 pv 已含")
add("Combat.History.Entries.SummonedEntry", "<Amount>", "K", "无玩法读取者")

# ---------------------------------------------------------------- commands (transient)
add("Commands.Builders.AttackCommand",
    "<Attacker> <CardPlay> <DamageProps> <HitSfx> <HitVfx> <IsRandomlyTargeted> <ModelSource> <TargetSide> <TmpHitSfx> "
    "_attackerAnimDelay _attackerAnimName _attackerSfx _attackerVfx _calculatedDamageVar _combatState _customAttackerVfxNodes "
    "_customHitVfxNodes _damagePerHit _doesRandomTargetingAllowDuplicates _hitCount _playOnEveryHit _results _shouldPlayAnimation "
    "_singleTarget _sourceType _spawnVfxOnCreatureCenter _spawnVfxOnEachCreature _tmpAttackerSfx _visualAttacker _waitBeforeHit",
    "K", "Vigor/Gigantification 的 commandToModify 残留引用：一次攻击执行期间的临时对象（含动画音效参数），决策点上无意义")

# ---------------------------------------------------------------- cards
add("Entities.Cards.CardCreationResult", "_modifiedCard _modifyingRelics originalCard", "K",
    "商店/奖励生成卡牌时的中间结果；最终卡牌经 CreationResult.Card 导出")
add("Entities.Cards.CardEnergyCost", "<Canonical>", "K", "卡牌原始费用，按 id 固定")
add("Entities.Cards.CardEnergyCost", "<CostsX>", "E", "card.x")
add("Entities.Cards.CardEnergyCost", "<WasJustUpgraded>", "K", "升级动画标志")
add("Entities.Cards.CardEnergyCost", "_base", "E", "card.cost（GetResolved 含基础费用、局部与全局修正）")
add("Entities.Cards.CardEnergyCost", "_capturedXValue", "K", "X 费牌出牌时临时捕获的能量值；决策点上是上次出牌的陈旧值")
add("Entities.Cards.CardEnergyCost", "_card", "K", "回指卡牌")
add("Entities.Cards.CardEnergyCost", "_localModifiers", "G",
    "G10（低）：当前费用已含修正，但修正何时失效（本回合/打出前/本场，CardEnergyCost.cs Set*/Add*）未导出")
add("Entities.Cards.CardPile", "<Type>", "K", "牌堆类型标签")
add("Entities.Cards.CardPile", "_cards", "E",
    "各牌堆内容：deck/hand/discard/exhaust 按序，draw 按 id 排序（顺序 H，已隐藏）；已知顶牌缺口见 G2")
add("Entities.Cards.CardPlay", "<Card> <IsAutoPlay> <PlayCount> <PlayIndex> <Player> <Resources> <ResultPile> <Target>", "D",
    "历史条目里的出牌记录；按类型聚合的缺口见 G4")
add("Entities.Cards.LocalCostModifier", "<Amount> <Type> <IsReduceOnly>", "D", "已折算进 card.cost")
add("Entities.Cards.LocalCostModifier", "<Expiration>", "G", "G10（低）：费用修正的失效条件（EndOfTurn/WhenPlayed/EndOfCombat）未导出")
add("Entities.Cards.ResourceInfo", "<EnergySpent> <EnergyValue> <StarValue> <StarsSpent>", "D", "出牌记录里的花费，公开历史")
add("Entities.Cards.TemporaryCardCost", "<Cost>", "E", "card.star（CurrentStarCost 取最后一个临时星费）")
add("Entities.Cards.TemporaryCardCost", "<ClearsWhenCardIsPlayed> <ClearsWhenTurnEnds>", "G",
    "G10（低）：临时星费何时失效未导出（CardModel.cs:1620,2011）")

# ---------------------------------------------------------------- creatures
add("Entities.Creatures.Creature", "<CombatId> <CombatState> <Player> <Side> <MonsterMaxHpBeforeModification> _petOwner", "K",
    "引用、阵营与生成时的原始血量，基础设施")
add("Entities.Creatures.Creature", "<HpDisplay>", "D",
    "血条显示模式：WaterfallGiant 濒爆时 hp=999999999 已导出；HardenedShell 的'本回合已封顶'可由 pw.d.damageReceivedThisTurn≥n 推出")
add("Entities.Creatures.Creature", "<Monster>", "E", "怪物 id（cr.id）")
add("Entities.Creatures.Creature", "<SlotName>", "G", "G11（低）：敌人槽位/站位未导出，召唤需空槽、站位即行动顺序，见 M-F15")
add("Entities.Creatures.Creature", "_block", "E", "cr.block")
add("Entities.Creatures.Creature", "_currentHp", "E", "cr.hp / g.hp")
add("Entities.Creatures.Creature", "_maxHp", "E", "cr.max_hp / g.max_hp")
add("Entities.Creatures.Creature", "_powers", "E", "cr.pw[]（id、n、v、d、cards）")
add("Entities.Creatures.DamageResult",
    "<BlockedDamage> <OverkillDamage> <Props> <Receiver> <UnblockedDamage> <WasBlockBroken> <WasFullyBlocked> <WasTargetKilled>", "D",
    "历史里的伤害结算；'本回合掉过血'缺口记在 DamageReceivedEntry.Result（G4）")

# ---------------------------------------------------------------- merchant
add("Entities.Merchant.MerchantCardEntry", "<CreationResult>", "E", "buy_card 候选的 card")
add("Entities.Merchant.MerchantCardEntry", "<IsOnSale>", "D", "打折标签；折后价已在 price")
add("Entities.Merchant.MerchantCardEntry", "_cardPool _cardRarity _cardType _inventory", "K", "货架槽位的生成规格（类型/稀有度/卡池）")
add("Entities.Merchant.MerchantCardRemovalEntry", "<Used>", "D", "已用则无 remove 候选；ctx.removal_cost 仍给价格")
add("Entities.Merchant.MerchantEntry", "_cost", "E", "候选的 price；ctx.removal_cost")
add("Entities.Merchant.MerchantEntry", "_player", "K", "回指玩家")
add("Entities.Merchant.MerchantInventory", "<CardRemovalEntry>", "E", "remove 候选与 ctx.removal_cost")
add("Entities.Merchant.MerchantInventory", "<Player>", "K", "回指玩家")
add("Entities.Merchant.MerchantInventory", "_characterCardEntries _colorlessCardEntries _relicEntries", "G",
    "G6：只导出买得起且有货的条目（RunSimulator.Rl.cs:1009/1015），屏幕上买不起的货架看不到，见 R#3")
add("Entities.Merchant.MerchantInventory", "_potionEntries", "G",
    "G6：药水栏满时整排药水不导出（CanBuyPotion，Rl.cs:1021/1188），无法判断'弃一瓶换货'，见 R#3")
add("Entities.Merchant.MerchantPotionEntry", "<Model>", "E", "buy_potion 候选的 id")
add("Entities.Merchant.MerchantRelicEntry", "<Model>", "E", "buy_relic 候选的 id")

# ---------------------------------------------------------------- orbs / players
add("Entities.Orbs.OrbQueue", "<Capacity>", "E", "combat.orb_slots")
add("Entities.Orbs.OrbQueue", "_orbs", "E", "combat.orbs（id、passive、evoke），按队列顺序；编码器丢了位置，见 R#2（编码）")
add("Entities.Orbs.OrbQueue", "_owner", "K", "回指玩家")
add("Entities.Players.ExtraPlayerFields", "<CardShopRemovalsUsed>", "G",
    "G12（低）：已用商店删牌次数决定下一次删牌价格（MerchantCardRemovalEntry.cs:32），商店外不可见")
add("Entities.Players.ExtraPlayerFields", "<CccomboBadgeUnlocked> <DamageDealt> <DebuffsApplied>", "K", "成就/徽章统计，无玩法读取者")
add("Entities.Players.ExtraPlayerFields", "<WongoPoints>", "K", "Wongo 徽章积分（与存档进度相关），引擎每局重置档案")
add("Entities.Players.Player", "<BaseOrbSlotCount>", "D", "角色常量（Defect=3），战斗中 orb_slots 已导出")
add("Entities.Players.Player", "<Character>", "E", "g.char")
add("Entities.Players.Player", "<Creature>", "E", "玩家生物（hp、block、pw）")
add("Entities.Players.Player", "<Deck>", "E", "obs.deck")
add("Entities.Players.Player", "<DiscoveredCards> <DiscoveredEnemies> <DiscoveredEpochs> <DiscoveredPotions> <DiscoveredRelics>", "K", "图鉴发现进度")
add("Entities.Players.Player", "<ExtraFields>", "K", "统计/徽章容器；唯一有玩法意义的删牌次数见 G12")
add("Entities.Players.Player", "<IsActiveForHooks> <MaxAscensionWhenRunStarted> <NetId>", "K", "多人/档案基础设施")
add("Entities.Players.Player", "<MaxEnergy>", "D", "角色常量（只在构造时写入）；战斗中 max_energy（含钩子修正）已导出，战斗外由角色+遗物可知")
add("Entities.Players.Player", "<PlayerCombatState>", "E", "obs.combat")
add("Entities.Players.Player", "<PlayerOdds>", "G", "G7：卡牌稀有度保底与药水掉率的当前值（见 AbstractOdds.CurrentValue）")
add("Entities.Players.Player", "<PlayerRng>", "H", "玩家随机数集合")
add("Entities.Players.Player", "<RelicGrabBag>", "H", "预洗好的遗物出现顺序")
add("Entities.Players.Player", "<UnlockState>", "K", "档案解锁（引擎固定为全解锁）")
add("Entities.Players.Player", "_canUseOrRemovePotions", "D", "某些事件选择期间禁用药水；体现为药水候选消失")
add("Entities.Players.Player", "_gold", "E", "g.gold")
add("Entities.Players.Player", "_potionSlots", "E", "obs.potions（id、slot）与 g.potion_slots")
add("Entities.Players.Player", "_relics", "E", "obs.relics")
add("Entities.Players.Player", "_runPiles _runState", "K", "内部引用")
add("Entities.Players.PlayerCombatState", "<DiscardPile>", "E", "combat.discard")
add("Entities.Players.PlayerCombatState", "<DrawPile>", "G",
    "G2：组成已导出（combat.draw 按 id 排序，真实顺序为 H），但玩家亲手放到堆顶的已知顶牌丢失（Headbutt 等），见 R#1")
add("Entities.Players.PlayerCombatState", "<ExhaustPile>", "E", "combat.exhaust")
add("Entities.Players.PlayerCombatState", "<Hand>", "E", "combat.hand（含 playable、pv）")
add("Entities.Players.PlayerCombatState", "<OrbQueue>", "E", "combat.orbs / orb_slots")
add("Entities.Players.PlayerCombatState", "<PlayPile>", "D", "出牌中的牌；决策点上为空，出牌中途的选牌界面用 ctx.source 标出来源牌 id")
add("Entities.Players.PlayerCombatState", "<TurnNumber>", "G",
    "G9（低）：玩家回合数，额外回合后与 round 不同，46 个遗物按它触发（如 Horn Cleat 第 2 回合），见 R#8")
add("Entities.Players.PlayerCombatState", "_energy", "E", "combat.energy")
add("Entities.Players.PlayerCombatState", "_pets", "E", "Osty 经 combat.allies 导出；其他宠物为装饰")
add("Entities.Players.PlayerCombatState", "_phase", "K", "回合阶段，决策点上恒为 Play")
add("Entities.Players.PlayerCombatState", "_piles _player", "K", "内部引用")
add("Entities.Players.PlayerCombatState", "_stars", "E", "combat.stars")
add("Entities.RestSite.RestSiteOption", "<Owner>", "K", "回指玩家")
add("Entities.RestSite.SmithRestSiteOption", "<SmithCount>", "D", "锻造张数；由遗物决定（已导出），选牌时 ctx.max 给出")
add("Entities.RestSite.SmithRestSiteOption", "_selection", "K", "锻造选择的临时结果")

# ---------------------------------------------------------------- hover tips / localization
add("HoverTips.CardHoverTip", "<Card> <Id>", "K", "悬浮提示对象；其中的牌已由所属模型的 cards/tips 导出")
add("HoverTips.HoverTip", "<CanonicalModel> <Description> <Icon> <Id> <IsDebuff> <IsInstanced> <IsSmart> <ShouldOverrideTextOverflow> <Title>",
    "K", "悬浮提示的显示数据；事件选项的提示模型 id 经候选 tips 导出")
add("Localization.DynamicVars.BlockVar", "<Props>", "K", "数值属性标志，按卡牌固定")
add("Localization.DynamicVars.CalculatedBlockVar", "<Props>", "K", "数值属性标志")
add("Localization.DynamicVars.CalculatedDamageVar", "<IsFromOsty> <Props>", "K", "数值属性标志")
add("Localization.DynamicVars.DamageVar", "<Props>", "K", "数值属性标志")
add("Localization.DynamicVars.DynamicVar", "<Name>", "E", "v/pv 的键名")
add("Localization.DynamicVars.DynamicVar", "<WasJustUpgraded>", "K", "升级闪烁标志")
add("Localization.DynamicVars.DynamicVar", "_baseValue", "E", "card.v / pw.v / relic.v / 事件选项 v")
add("Localization.DynamicVars.DynamicVar", "_enchantedValue", "D", "附魔加成后的值 = 基础值 + 附魔（ench、ench_n 已导出）；手牌 pv 含之")
add("Localization.DynamicVars.DynamicVar", "_owner", "K", "回指模型")
add("Localization.DynamicVars.DynamicVar", "_previewValue", "E", "手牌 pv、指向候选 pv（读取时 ClearPreview 括起）")
add("Localization.DynamicVars.DynamicVarSet", "_vars", "E", "v/pv 容器")
add("Localization.DynamicVars.EnergyVar", "<ColorPrefix>", "K", "文本颜色前缀")
add("Localization.DynamicVars.ExtraDamageVar", "<IsFromOsty>", "K", "数值属性标志")
add("Localization.DynamicVars.OstyDamageVar", "<Props>", "K", "数值属性标志")
add("Localization.DynamicVars.StringVar", "<StringValue>", "D",
    "文本变量：事件的经 ctx.sv 导出；能力/遗物的（Shrink 施加者名、Nightmare 牌名、DustyTome 牌名）可由其他字段推出；"
    "例外是 Sea Glass 选项的角色名，见 G13")
add("Localization.LocString", "<locEntryKey>P <locTable>P _variables", "K", "本地化键")
add("Logging.Logger", "<Context> _logType", "K", "日志")

# ---------------------------------------------------------------- map
add("Map.ActMap", "startMapPoints", "E", "map.nodes/edges（起点）")
add("Map.MapCoord", "col row", "E", "map.nodes[].c / r")
add("Map.MapPoint", "<CanBeModified>", "K", "地图生成期标志")
add("Map.MapPoint", "<Children> parents", "E", "map.edges")
add("Map.MapPoint", "<PointType>", "E", "map.nodes[].t（问号房在进入时才掷类型，不泄露）")
add("Map.MapPoint", "_quests", "E", "map.nodes[].q（Fur Coat、Spoils Map 的标记）")
add("Map.MapPoint", "coord", "E", "map.nodes[].c/r")
add("Map.MapPointTypeCounts", "<NumOfElites> <NumOfRests> <NumOfShops> <NumOfUnknowns> <PointTypesThatIgnoreRules>", "K",
    "地图生成参数；结果已在地图上")
add("Map.SpoilsActMap", "<BossMapPoint> <Grid> <StartingMapPoint>", "E", "Spoils Map 的藏宝图地图，经 map 导出")
add("Map.SpoilsActMap", "_mapLength _pointTypeCounts _treasureRow", "K", "生成参数")
add("Map.SpoilsActMap", "_rng", "H", "地图随机数")
add("Map.StandardActMap", "<BossMapPoint> <Grid> <SecondBossMapPoint> <StartingMapPoint>", "E", "map.nodes/edges/cur；第二个 boss 节点与连线")
add("Map.StandardActMap", "<ShouldReplaceTreasureWithElites>", "K", "生成参数，结果在节点类型里")
add("Map.StandardActMap", "_mapLength _pointTypeCounts", "K", "生成参数")
add("Map.StandardActMap", "_rng", "H", "地图随机数")

# ---------------------------------------------------------------- models: base
add("Models.AbstractModel", "<CategorySortingId> <EntrySortingId> <IsMutable>", "K", "排序与可变性标志")
add("Models.AbstractModel", "<Id>", "E", "所有实体的 id")
add("Models.ActModel", "_allBossEncounters _allEliteEncounters _allEncounters _allMonsters _allRegularEncounters _allWeakEncounters", "K",
    "该幕的内容池（按幕固定，玩家可从图鉴知道）")
add("Models.ActModel", "_canonicalInstance", "K", "规范模型引用；当前幕 id 经 g.act_id")
add("Models.ActModel", "_rooms", "H", "该幕预生成的房间队列（事件、遭遇顺序）；当前幕 boss 已导出；后续幕全部 H")
add("Models.ActModel", "_sharedAncientSubset", "H", "跨幕共享先古事件的预选子集")
add("Models.AfflictionModel", "_amount", "E", "card.aff_n")
add("Models.AfflictionModel", "_canonicalInstance", "E", "card.aff（id）")
add("Models.AfflictionModel", "_card", "K", "回指卡牌")
add("Models.Badges.CccComboModel", "_cardsPlayedThisTurn", "K", "徽章计数（与 played 重复）")

add("Models.CardModel", "<Affliction>", "E", "card.aff / aff_n")
add("Models.CardModel", "<CanonicalEnergyCost>", "K", "按 id 固定")
add("Models.CardModel", "<Enchantment>", "E", "card.ench / ench_n / ench_d")
add("Models.CardModel", "<HasBeenRemovedFromState>", "K", "移除标志")
add("Models.CardModel", "<Rarity>", "E", "card.rar")
add("Models.CardModel", "<ShouldShowInCardLibrary>", "K", "图鉴标志")
add("Models.CardModel", "<TargetType>", "E", "card.tgt")
add("Models.CardModel", "<Type>", "E", "card.type")
add("Models.CardModel", "_baseReplayCount", "G",
    "G3：牌面显示 'Replay X'（CardModel.cs:1428-1434），Transfigure/HiddenGem/SoldiersStew/SwordSage 会改它，未导出，见 R#6")
add("Models.CardModel", "_baseStarCost", "E", "card.star")
add("Models.CardModel", "_canonicalInstance", "E", "card.id")
add("Models.CardModel", "_cloneOf _deckVersion", "K", "克隆/牌组原件引用；IsClone 只影响 Stomp 等的费用，已体现在 cost")
add("Models.CardModel", "_currentPlayIndex _currentTarget _lastStarsSpent", "K", "一次出牌过程中的临时值")
add("Models.CardModel", "_currentUpgradeLevel", "E", "card.up")
add("Models.CardModel", "_dynamicVars", "E", "card.v / pv")
add("Models.CardModel", "_energyCost", "E", "card.cost / x（失效条件见 G10）")
add("Models.CardModel", "_exhaustOnNextPlay", "K", "Havoc/Cinder 在自动打出前一刻置位、该次打出即清除（CardPileCmd.cs:1174），决策点上恒 false")
add("Models.CardModel", "_floorAddedToDeck", "K", "记录用，无玩法读取者")
add("Models.CardModel", "_hasSingleTurnRetain", "G", "G3：Expertise 给的单回合保留，牌面写 Retain，kw 不含（CardModel.cs:1418-1424），见 R#7")
add("Models.CardModel", "_hasSingleTurnSly", "G", "G3：Hand Trick 给的单回合 Sly，牌面写 Sly，kw 不含，见 R#7")
add("Models.CardModel", "_isDupe", "K", "重放/复制产生的临时副本，打出后进 Limbo，不会停在手里")
add("Models.CardModel", "_isEnchantmentPreview _titleLocString _upgradePreviewType _wasStarCostJustUpgraded", "K", "预览/动画/本地化")
add("Models.CardModel", "_keywords", "E", "card.kw（单回合 Retain/Sly 除外，见 G3）")
add("Models.CardModel", "_owner _pool", "K", "引用")
add("Models.CardModel", "_starCostSet", "K", "惰性初始化标志")
add("Models.CardModel", "_tags", "K", "Strike/Defend/Shiv 等标签，按 id 固定（CardModel.cs:534 注释注明不变）")
add("Models.CardModel", "_temporaryStarCosts", "E", "折算进 card.star；失效条件见 G10")

add("Models.Cards.BouncingFlask", "_vfxTint", "K", "特效颜色")
add("Models.Cards.Claw", "_extraDamageFromClawPlays", "E", "card.d（OwnState）")
add("Models.Cards.Discovery", "_mockSelectedCard", "K", "测试 mock")
add("Models.Cards.Dowsing", "_roomsEntered", "E", "card.d")
add("Models.Cards.GeneticAlgorithm", "_currentBlock _increasedBlock", "E", "card.v.saved.* / card.d")
add("Models.Cards.Guilty", "_combatsSeen", "E", "card.v.saved.* / card.d")
add("Models.Cards.KinglyPunch", "_extraDamage", "E", "card.d")
add("Models.Cards.MadScience", "_mockedChaosCard", "K", "测试 mock")
add("Models.Cards.MadScience", "_tinkerTimeRider _tinkerTimeType", "G",
    "G8：Tinker Time 选定的牌型与附加效果是枚举型 [SavedProperty]，RlVars 只收数值、OwnState 跳过存档后备字段，两边都丢（MadScience.cs:112,126），见 R#10")
add("Models.Cards.Rampage", "_extraDamageFromPlays", "E", "card.d")
add("Models.Cards.SovereignBlade", "_createdThroughForge _currentDamage _currentRepeats", "E", "card.d / v.saved")
add("Models.Cards.Splash", "_mockGeneratedCard", "K", "测试 mock")
add("Models.Cards.SpoilsMap", "<SpoilsCoord>", "E", "藏宝点经 map.nodes[].q 标出（本幕）")
add("Models.Cards.SpoilsMap", "_spoilsActIndex", "E", "card.v.saved.SpoilsActIndex")
add("Models.Cards.TheScythe", "_currentDamage _increasedDamage", "E", "card.v.saved / d")
add("Models.Cards.Thrash", "_extraDamage", "E", "card.d")
add("Models.Cards.UpMySleeve", "_timesPlayedThisCombat", "E", "card.d")
add("Models.Cards.Wither", "_fakeUpgradeLevel", "E", "card.d")

add("Models.EnchantmentModel", "<Props>", "K", "存档属性容器")
add("Models.EnchantmentModel", "_amount", "E", "card.ench_n")
add("Models.EnchantmentModel", "_canonicalInstance", "E", "card.ench（id）")
add("Models.EnchantmentModel", "_card", "K", "回指卡牌")
add("Models.EnchantmentModel", "_dynamicVars", "K", "附魔自身数值按 id 固定（Glam Times=1 等）；可变量是 Amount")
add("Models.EnchantmentModel", "_iconPath", "K", "图标路径")
add("Models.EnchantmentModel", "_status", "G",
    "G8：Swift/Sown/Vigorous 首次打出后变 Disabled，牌面附魔变灰（NCard.cs:923）；基类字段，OwnState 不收，见 P")
add("Models.Enchantments.Glam", "_usedThisCombat", "E", "card.ench_d")
add("Models.Enchantments.Slither", "_testEnergyCostOverride", "K", "测试字段（_cosmeticFields 已排除）")
add("Models.EncounterModel", "_backgroundAssets _monstersWithSlots _spawnedEnemies", "K", "遭遇的生成配置与背景")
add("Models.EncounterModel", "_canonicalInstance", "G", "G11（低）：遭遇 id，同 CombatState._encounter")
add("Models.EncounterModel", "_rng", "H", "遭遇随机数")
add("Models.Encounters.GremlinMercNormal", "_goldWasStolen", "D", "是否被偷过金币，屏幕上可见、可由历史推出；只影响战后金币比例")
add("Models.ModelId", "<Category> <Entry>", "E", "id 字符串")

add("Models.MonsterModel", "<NextMove>", "E", "intents（类型、伤害、段数、张数）与 move/then（可辨认时）")
add("Models.MonsterModel", "_canonicalInstance", "E", "cr.id")
add("Models.MonsterModel", "_creature", "K", "回指生物")
add("Models.MonsterModel", "_isPerformingMove", "K", "执行招式期间为真，决策点上恒 false")
add("Models.MonsterModel", "_moveStateMachine", "E", "招式状态机：已执行招式经 prev、可辨认的下一招经 move/then；随机分支的掷骰结果按设计隐藏")
add("Models.MonsterModel", "_rng _runRng", "H", "怪物 AI 随机数")
add("Models.MonsterModel", "_spawnedThisTurn", "K",
    "只让敌方回合中途召出的怪当回合不行动（Creature.cs:717），换边即清除（MonsterModel.cs:486）；玩家决策点上不改变后续")

# monster subclasses
add("Models.Monsters.Aeonglass", "_additionalStrength _witherUpgradeCount", "E", "cr.d")
add("Models.Monsters.Axebot", "_shouldPlaySpawnAnimation", "K", "动画（_cosmeticFields 已排除）")
add("Models.Monsters.Axebot", "_stockOverrideAmount", "E", "cr.d（可空 int）")
add("Models.Monsters.BowlbugRock", "_isOffBalance", "E", "cr.d")
add("Models.Monsters.CalcifiedCultist", "_attackSfxStrength", "K", "音效（已排除）")
add("Models.Monsters.CeremonialBeast", "_beastCryState", "K", "状态机内的招式引用（结构常量）")
add("Models.Monsters.CeremonialBeast", "_inMidCharge", "E", "cr.d；只在动作执行中途为真，决策点恒 0（噪声，M-F9）")
add("Models.Monsters.CeremonialBeast", "_isInSecondPhase _isStunnedByPlowRemoval", "E", "cr.d")
add("Models.Monsters.Chomper", "_screamFirst", "E", "cr.d（由遭遇按位置设定，非随机）")
add("Models.Monsters.CorpseSlug", "_isRavenous", "E", "cr.d")
add("Models.Monsters.CorpseSlug", "_starterMoveIdx", "E",
    "cr.d；遭遇 Rng 定的起手位置，首个意图已显示同等信息；撞图标时理论上多说（M-F5，建议去掉）")
add("Models.Monsters.CrossbowRubyRaider", "_isCrossbowReloaded", "E", "cr.d")
add("Models.Monsters.Crusher", "_background", "K", "背景节点")
add("Models.Monsters.CubexConstruct", "_isBurrowed _isCharging", "E", "cr.d（动画相关，M-F9）")
add("Models.Monsters.DampCultist", "_attackSfxStrength", "K", "音效（已排除）")
add("Models.Monsters.DecimillipedeSegment", "_deadState", "K", "招式引用")
add("Models.Monsters.DecimillipedeSegment", "_starterMoveIdx", "E", "cr.d；同 CorpseSlug（M-F5）")
add("Models.Monsters.DevotedSculptor", "_ritualGain", "K", "readonly 常量")
add("Models.Monsters.Fabricator", "_lastSpawned", "G", "G11（低）：上一只召出的机器人类型（下一只不会相同，Fabricator.cs:114），引用型字段 OwnState 不收，见 M-F13")
add("Models.Monsters.FatGremlin", "_isAwake", "E", "cr.d（例中取自已逃跑的怪）")
add("Models.Monsters.FrogKnight", "_hasBeetleCharged", "E", "cr.d")
add("Models.Monsters.FuzzyWurmCrawler", "_isPuffed", "E", "cr.d")
add("Models.Monsters.GasBomb", "_hasExploded", "E", "cr.d")
add("Models.Monsters.GremlinMerc", "_hasSpoken", "K", "台词标志（已排除）")
add("Models.Monsters.Inklet", "_middleInklet", "E", "cr.d（遭遇按位置设定）")
add("Models.Monsters.KinFollower", "_startsWithDance", "E", "cr.d（遭遇设定）")
add("Models.Monsters.KinPriest", "_speechUsed", "K", "台词标志（已排除）")
add("Models.Monsters.KnowledgeDemon", "_curseOfKnowledgeCounter", "E", "cr.d")
add("Models.Monsters.KnowledgeDemon", "_isBurnt", "E", "cr.d（动画相关，M-F9）")
add("Models.Monsters.LagavulinMatriarch", "_isAwake", "E", "cr.d")
add("Models.Monsters.LagavulinMatriarch", "_isShellAwake", "E", "cr.d（动画相关，M-F9）")
add("Models.Monsters.LagavulinMatriarch", "_sleepingVfx", "K", "特效节点")
add("Models.Monsters.LivingFog", "_bloatAmount", "E", "cr.d（常量 1，噪声）")
add("Models.Monsters.LouseProgenitor", "_curled", "E", "cr.d")
add("Models.Monsters.MechaKnight", "_isWoundUp", "E", "cr.d")
add("Models.Monsters.Nibbit", "_isAlone _isFront", "E", "cr.d")
add("Models.Monsters.OwlMagistrate", "_isFlying", "E", "cr.d")
add("Models.Monsters.PhantasmalGardener", "<CurrentScale>", "K", "缩放动画，由 enlargeTriggers 算出（PhantasmalGardener.cs:154）")
add("Models.Monsters.PhantasmalGardener", "_enlargeTriggers", "E", "cr.d")
add("Models.Monsters.PunchConstruct", "_startingHpReduction", "E", "cr.d；随机值，但已体现为可见的初始血量")
add("Models.Monsters.PunchConstruct", "_startsWithFastPunch", "E", "cr.d（遭遇设定）")
add("Models.Monsters.Queen", "_amalgam", "K", "队友引用；Amalgam 本身在 enemies 里")
add("Models.Monsters.Queen", "_burnBrightForMeState _enragedState", "K", "招式引用")
add("Models.Monsters.Queen", "_hasAmalgamDied", "E", "cr.d")
add("Models.Monsters.Rocket", "_background", "K", "背景节点")
add("Models.Monsters.ScrollOfBiting", "_starterMoveIdx", "E", "cr.d；同 CorpseSlug（M-F5）")
add("Models.Monsters.SlumberingBeetle", "_isAwake", "E", "cr.d")
add("Models.Monsters.SlumberingBeetle", "_sleepingVfx", "K", "特效节点")
add("Models.Monsters.SoulFysh", "_isInvisible", "E", "cr.d")
add("Models.Monsters.SpinyToad", "_isSpiny", "E", "cr.d（与 THORNS_POWER 重复）")
add("Models.Monsters.TerrorEel", "_terrorState", "K", "招式引用")
add("Models.Monsters.TestSubject", "_deadState", "K", "招式引用")
add("Models.Monsters.TestSubject", "_extraMultiClawCount _respawns", "E", "cr.d")
add("Models.Monsters.TheInsatiable", "_hasLiquified", "E", "cr.d")
add("Models.Monsters.TheObscura", "_hasSummoned", "E", "cr.d")
add("Models.Monsters.ThievingHopper", "_isHovering", "E", "cr.d")
add("Models.Monsters.Toadpole", "_isFront", "E", "cr.d")
add("Models.Monsters.ToughEgg", "_afterHatchedState", "K", "招式引用")
add("Models.Monsters.ToughEgg", "_hatchPos", "K", "孵化位置（屏幕坐标）")
add("Models.Monsters.ToughEgg", "_hatched _isHatched", "E", "cr.d（两者重复，M-F9）")
add("Models.Monsters.Tunneler", "_isStunned", "E", "cr.d")
add("Models.Monsters.TwoTailedRat", "_callForBackupCount _turnsUntilSummonable", "E", "cr.d（由出招历史决定的计数）")
add("Models.Monsters.TwoTailedRat", "_starterMoveIndex", "E", "cr.d；同 CorpseSlug（M-F5）")
add("Models.Monsters.Vantom", "_scaleTween", "K", "动画")
add("Models.Monsters.WaterfallGiant", "_aboutToBlowState", "K", "招式引用")
add("Models.Monsters.WaterfallGiant", "_currentPressureGunDamage _isAboutToBlow _steamEruptionDamage", "E", "cr.d")
add("Models.Monsters.WaterfallGiant", "_pressureBuildupIdx", "E",
    "cr.d；纯动画轨道且只在 TestMode 关时自增，无头恒 0、客户端递增：保真风险，应列入 _cosmeticFields（M-F8）")
add("Models.Monsters.Wriggler", "_startStunned", "E", "cr.d（遭遇/InfestedPower 设定）")

add("Models.OrbModel", "<HasBeenRemovedFromState> _owner", "K", "引用/移除标志")
add("Models.OrbModel", "_canonicalInstance", "E", "orbs[].id")
add("Models.Orbs.DarkOrb", "_evokeVal", "E", "orbs[].evoke")
add("Models.Orbs.GlassOrb", "_passiveVal", "E", "orbs[].passive")
add("Models.PotionModel", "<HasBeenRemovedFromState> <IsQueued> _owner", "K", "移除/排队标志与引用")
add("Models.PotionModel", "_canonicalInstance", "E", "potions[].id")
add("Models.PotionModel", "_dynamicVars", "D", "药水数值按 id 固定；增益遗物已导出")
add("Models.Potions.Ashwater", "_tint", "K", "颜色")
add("Models.Potions.SneckoOil", "_testEnergyCostOverride", "K", "测试字段")

add("Models.PowerModel", "_amount", "E", "pw[].n")
add("Models.PowerModel", "_amountOnTurnStart", "D", "本回合开始时的层数（Creature.cs:682），只用来防止同回合中途施加的能力立刻触发；是 pw.n 的公开历史")
add("Models.PowerModel", "_applier", "D",
    "施加者；Hex/Constrict/Shrink 随施加者死亡移除，当前内容里每场施加者唯一（SpectralKnight/SlitheringStrangler/ShrinkerBeetle），可由怪物 id 推出，见 P")
add("Models.PowerModel", "_canonicalInstance", "E", "pw[].id")
add("Models.PowerModel", "_dynamicVars", "E", "pw[].v")
add("Models.PowerModel", "_internalData", "E", "pw[].d（标量）与 pw[].cards；集合型数据见各 Data 行")
add("Models.PowerModel", "_owner _target _resolvedBigIconPath", "K", "引用（_target 只在多人有意义）与图标")
add("Models.PowerModel", "_skipNextDurationTick", "K",
    "只在玩家被施加负面时置位（PowerCmd.cs:148-151），来源全在怪物招式（敌方回合）或 Shame/Doubt（玩家回合末），"
    "同一个敌方回合末即被消耗（PowerCmd.cs:194-201）；单人决策点上恒为 false，同 P 的结论")

add("Models.Powers.AdaptablePower+Data", "isReviving", "E", "pw.d")
add("Models.Powers.AfterimagePower+Data", "amountsForPlayedCards", "K", "一次出牌开始到结束之间的临时记账，决策点上为空")
add("Models.Powers.ChainsOfBindingPower+Data", "boundCardPlayed", "E", "pw.d")
add("Models.Powers.CurlUpPower+Data", "playedCard", "K", "一次出牌期间的临时引用")
add("Models.Powers.DampenPower+Data", "casters", "D", "施法者（MagiKnight）每场唯一")
add("Models.Powers.DampenPower+Data", "downgradedCardsToOldUpgradeLevels", "G",
    "G11（低）：被降级的牌原来的升级等级，Dampen 结束时恢复（DampenPower.cs:34,60）；战斗中 deck 默认不编码，见 P")
add("Models.Powers.DarkEmbracePower+Data", "etherealCount", "E", "pw.d")
add("Models.Powers.FeralPower+Data", "zeroCostAttacksPlayed", "E", "pw.d")
add("Models.Powers.GigantificationPower+Data", "commandToModify", "K", "攻击执行中的临时引用")
add("Models.Powers.HardenedShellPower+Data", "damageReceivedThisTurn", "E", "pw.d")
add("Models.Powers.IllusionPower", "_followUpStateId", "E", "pw.d（字符串；复活后接续的招式 = 历史上的最后一招，不是掷骰结果）")
add("Models.Powers.IllusionPower+Data", "isReviving", "E", "pw.d")
add("Models.Powers.NemesisPower", "_shouldApplyIntangible", "E", "pw.d")
add("Models.Powers.OrbitPower+Data", "energySpent triggerCount", "E", "pw.d")
add("Models.Powers.PanachePower+Data", "alreadyApplied", "E", "pw.d")
add("Models.Powers.PossessSpeedPower+Data", "stolenDexterity", "G", "G11（低）：已偷走的敏捷总量（死亡时归还），字典型不进 OwnState，见 P")
add("Models.Powers.PossessStrengthPower+Data", "stolenStrength", "G", "G11（低）：已偷走的力量总量（PossessStrengthPower.cs:53），见 P")
add("Models.Powers.ReaperFormPower", "_vfx", "K", "特效")
add("Models.Powers.ReattachPower+Data", "isReviving", "E", "pw.d")
add("Models.Powers.RitualPower", "_wasJustAppliedByEnemy", "E", "pw.d")
add("Models.Powers.RupturePower+Data", "playedCards", "K", "一次出牌期间的临时记账")
add("Models.Powers.SandpitPower", "_initialAmount _initialTargetPosition", "E",
    "pw.d；只在 TestMode 关时写入（SandpitPower.cs:60-66），无头恒 0、客户端为层数与像素坐标：保真风险，应列入 _cosmeticFields（P）")
add("Models.Powers.SerpentFormPower", "_vfx", "K", "特效")
add("Models.Powers.SerpentFormPower+Data", "amountsForPlayedCards", "K", "一次出牌期间的临时记账")
add("Models.Powers.SkittishPower+Data", "hasGainedBlockThisTurn", "E", "pw.d")
add("Models.Powers.SlothPower", "_cardsPlayedThisTurn", "E", "pw.d")
add("Models.Powers.StormPower+Data", "amountsForPlayedCards", "K", "一次出牌期间的临时记账")
add("Models.Powers.StranglePower+Data", "amountsForPlayedCards", "K", "一次出牌期间的临时记账")
add("Models.Powers.SubroutinePower+Data", "amountsForPlayedCards", "K", "一次出牌期间的临时记账")
add("Models.Powers.SurroundedPower", "_facing", "E", "pw.d（朝向，决定背刺）")
add("Models.Powers.SwipePower", "_stolenCard", "E", "pw.cards（只有 id，升级/附魔丢失，M-F16）")
add("Models.Powers.TenderPower", "_cardsPlayedThisTurn", "E", "pw.d")
add("Models.Powers.VigorPower+Data", "amountWhenAttackStarted", "E", "pw.d（攻击执行中的临时值，决策点恒 0，噪声）")
add("Models.Powers.VigorPower+Data", "commandToModify", "K", "攻击执行中的临时引用")

add("Models.RelicModel", "<HasBeenRemovedFromState> _owner _resolvedBigIconPath", "K", "移除标志、引用、图标")
add("Models.RelicModel", "<StackCount>", "K", "IncrementStackCount 在 v0.111.0 无调用者，恒为 1")
add("Models.RelicModel", "_canonicalInstance", "E", "relics[].id")
add("Models.RelicModel", "_dynamicVars", "E", "relics[].v")
add("Models.RelicModel", "_floorAddedToDeck", "K", "记录用")
add("Models.RelicModel", "_isMelted", "E", "relics[].v.saved.IsMelted，另有 status=Disabled（Player.cs:521-522）")
add("Models.RelicModel", "_isWax", "E",
    "relics[].v.saved.IsWax（RelicModel 基类的 [SavedProperty]，RlVars 连基类一起收）；Toy Box 的蜡制遗物每 3 场融化一个")
add("Models.RelicModel", "_status", "E", "relics[].status（Disabled/Active）")

add("Models.Relics.ArchaicTooth", "_extraHoverTips", "K", "悬浮提示")
add("Models.Relics.ArchaicTooth", "_serializableAncientCard _serializableStarterCard", "E", "relics[].cards（HeldCards 收 SerializableCard）")
add("Models.Relics.BeltBuckle", "_dexterityApplied", "E", "relics[].d")
add("Models.Relics.BingBong", "_cardsToSkip", "E", "relics[].cards")
add("Models.Relics.BookOfFiveRings", "_cardsAdded", "E", "relics[].d / cnt")
add("Models.Relics.BrilliantScarf", "_cardsPlayedThisTurn", "E", "relics[].d / cnt")
add("Models.Relics.CentennialPuzzle", "_usedThisCombat", "E", "relics[].d")
add("Models.Relics.DustyTome", "_ancientCard", "D", "选项阶段牌经候选 tips 导出（EventOption 用 HoverTipsExcludingRelic）；获得时牌已加入 deck")
add("Models.Relics.DustyTome", "_extraHoverTips", "K", "悬浮提示")
add("Models.Relics.FakeHappyFlower", "_turnsSeen", "E", "relics[].d / cnt")
add("Models.Relics.FakeOrichalcum", "_shouldTrigger", "E", "relics[].d（决策点恒 0，噪声，R#16）")
add("Models.Relics.FakeVenerableTeaSet", "_gainEnergyInNextCombat", "E", "relics[].d")
add("Models.Relics.FurCoat", "<FurCoatCoordCols> <FurCoatCoordRows> <FurCoatCoordsSet>", "E", "标记的节点经 map.nodes[].q 导出")
add("Models.Relics.FurCoat", "_furCoatActIndex", "E", "relics[].v.saved")
add("Models.Relics.Girya", "_timesLifted", "E", "relics[].v.saved / cnt")
add("Models.Relics.HappyFlower", "_turnsSeen", "E", "relics[].d / cnt")
add("Models.Relics.IronClub", "_cardsPlayed", "E", "relics[].d / cnt")
add("Models.Relics.JossPaper", "_cardsExhausted", "E", "relics[].d / cnt")
add("Models.Relics.JossPaper", "_etherealCount", "E", "relics[].d（决策点恒 0，噪声，R#16）")
add("Models.Relics.Kusarigama", "_attacksPlayedThisTurn", "E", "relics[].d / cnt")
add("Models.Relics.LastingCandy", "_combatRewardsSeen", "E", "relics[].d / cnt")
add("Models.Relics.LavaRock", "_hasTriggered", "E", "relics[].d")
add("Models.Relics.LetterOpener", "_skillsPlayedThisTurn", "E", "relics[].d / cnt")
add("Models.Relics.LizardTail", "_wasUsed", "E", "relics[].d / status")
add("Models.Relics.Nunchaku", "_attacksPlayed", "E", "relics[].d / cnt")
add("Models.Relics.PaelsEye", "_usedThisCombat", "E", "relics[].d")
add("Models.Relics.PaelsEye", "_wasOwnerPartOfLastPlayerTurn", "E", "relics[].d（单人恒 1，噪声，R#16）")
add("Models.Relics.PaelsLegion", "_affectedCardPlay", "K", "一次出牌的临时引用")
add("Models.Relics.PaelsLegion", "_cooldown", "E", "relics[].d")
add("Models.Relics.PaelsLegion", "_skin", "K", "皮肤（已排除）")
add("Models.Relics.PaelsLegion", "_triggeredBlockLastTurn", "E", "relics[].d（动画标志，噪声，R#16）")
add("Models.Relics.PaelsTears", "_hadLeftoverEnergy", "E", "relics[].d")
add("Models.Relics.Pendulum", "_turnsSeen", "E", "relics[].d / cnt")
add("Models.Relics.Permafrost", "_activatedThisCombat", "E", "relics[].d")
add("Models.Relics.RainbowRing", "_activationCountThisTurn _attacksPlayedThisTurn _powersPlayedThisTurn _skillsPlayedThisTurn", "E", "relics[].d")
add("Models.Relics.RedSkull", "_strengthApplied", "E", "relics[].d")
add("Models.Relics.SeaGlass", "_characterId", "G",
    "G13（低）：Orobas 提供 Sea Glass 时已定好角色牌池（Orobas.cs:49-50），标题上写着；ModelId 型存档属性与选项都不导出，见 R#11。获得后立即结算，无后续意义")
add("Models.Relics.Shuriken", "_attacksPlayedThisTurn", "E", "relics[].d / cnt")
add("Models.Relics.SilkenTress", "_isUsed", "E", "relics[].d")
add("Models.Relics.SilverCrucible", "_timesUsed _treasureRoomsEntered", "E", "relics[].d / cnt")
add("Models.Relics.SwordOfStone", "_elitesDefeated", "E", "relics[].d / cnt")
add("Models.Relics.ThrowingAxe", "_usedThisCombat", "E", "relics[].d")
add("Models.Relics.TouchOfOrobas", "_extraHoverTips", "K", "悬浮提示")
add("Models.Relics.TouchOfOrobas", "_starterRelic _upgradedRelic", "D", "获得时即替换起始遗物，结果在 relics 里；选项阶段经 tips 显示")
add("Models.Relics.TuningFork", "_skillsPlayed", "E", "relics[].v.saved / cnt")
add("Models.Relics.Vambrace", "_blockGainedThisCombat", "E", "relics[].d")
add("Models.Relics.Vambrace", "_triggeringCard", "E", "relics[].cards（临时引用）")
add("Models.Relics.VenerableTeaSet", "_gainEnergyInNextCombat", "E", "relics[].d")
for r, f in [("BookOfFiveRings", "_isActivating"), ("FakeHappyFlower", "_isActivating"), ("HappyFlower", "_isActivating"),
             ("IronClub", "_isActivating"), ("JossPaper", "_isActivating"), ("Kusarigama", "_isActivating"),
             ("LastingCandy", "_isActivating"), ("LetterOpener", "_isActivating"), ("Nunchaku", "_isActivating"),
             ("Pendulum", "_isActivating"), ("Shuriken", "_isActivating"), ("StoneCalendar", "_isActivating"),
             ("TuningFork", "_isActivating")]:
    add(f"Models.Relics.{r}", f, "K", "激活闪光 1 秒（_cosmeticFields 已排除）；它在客户端会临时改写 cnt/status，见 R#12 保真")
add("Models.Singleton.MultiplayerScalingModel", "_combatState _runState", "K", "多人缩放，单人无效果")

# ---------------------------------------------------------------- move state machine
add("MonsterMoves.MonsterMoveStateMachine.ConditionalBranchState", "<BranchId> <States>", "K", "图案定义（怪物图鉴知识）；条件都是可见状态（M 报告）")
add("MonsterMoves.MonsterMoveStateMachine.ConditionalBranchState+ConditionalBranch", "id", "K", "图案定义")
add("MonsterMoves.MonsterMoveStateMachine.MonsterMoveStateMachine", "<StateLog>", "G",
    "G11（低）：完整出招记录；观测只给最近 3 招 prev，'整场一次'（Mawler ROAR、Rat CALL_FOR_BACKUP）超过 3 回合后无从得知，见 M-F14")
add("MonsterMoves.MonsterMoveStateMachine.MonsterMoveStateMachine", "<States> _initialState", "K", "图案定义")
add("MonsterMoves.MonsterMoveStateMachine.MonsterMoveStateMachine", "_currentState", "E",
    "可辨认时为 move；随机分支掷出而图标不能区分时按设计隐藏（NextMoveId）")
add("MonsterMoves.MonsterMoveStateMachine.MonsterMoveStateMachine", "_performedFirstMove", "D", "prev 非空即为真")
add("MonsterMoves.MonsterMoveStateMachine.MoveState", "<FollowUpState>", "E", "then（下一招为确定招时）；构造后不再改写，不泄露")
add("MonsterMoves.MonsterMoveStateMachine.MoveState", "<FollowUpStateId> <MustPerformOnceBeforeTransitioning>", "K", "图案定义")
add("MonsterMoves.MonsterMoveStateMachine.MoveState", "<Intents>", "E", "intents[]")
add("MonsterMoves.MonsterMoveStateMachine.MoveState", "<StateId>", "E", "move / then / prev 的招式 id")
add("MonsterMoves.MonsterMoveStateMachine.MoveState", "_performedAtLeastOnce", "D", "只配合 MustPerformOnce 防止复活招被眩晕替换；可由 prev 推出")
add("MonsterMoves.MonsterMoveStateMachine.RandomBranchState", "<StateId> <States>", "K", "图案定义（权重为 lambda）")
add("MonsterMoves.MonsterMoveStateMachine.RandomBranchState+StateWeight", "cooldown maxTimes repeatType stateId", "K",
    "图案规则（NextMoveId 用它们排除不可能的分支）")

# ---------------------------------------------------------------- multiplayer / infra
add("Multiplayer.Game.RestSiteSynchronizer",
    "_hoverMessageTask _hoveredMessage _lastHoverMessageMsec _localPlayerId _logger _messageBuffer _netService _playerCollection _restSites _runLobby",
    "K", "多人休息处同步")
add("Multiplayer.Game.RestSiteSynchronizer+PlayerRestSite", "completionTaskSource hoveredOptionIndex lastChosenOptionIndex", "K", "多人同步状态")
add("Multiplayer.Game.RestSiteSynchronizer+PlayerRestSite", "options", "E", "rest 候选（RestSiteRoom.Options 同一批对象）")
add("Multiplayer.Game.RunLocationTargetedMessageBuffer",
    "<CurrentLocation> _gameService _logger _messageHandlers _messagesWaitingOnLocationChange _visitedLocations", "K", "多人消息缓冲")
add("Multiplayer.Game.RunLocationTargetedMessageBuffer+MessageHandler", "originalHandler", "K", "多人消息缓冲")
add("Multiplayer.Game.RunLocationTargetedMessageBuffer+TypeAndMessageHandlers", "handlers messageType netServiceHandler", "K", "多人消息缓冲")
add("Multiplayer.NetSingleplayerGameService", "<LocalVersion> _isLoading", "K", "联网服务")
add("Multiplayer.PeerVersionInfo", "branch gameplayAffectingMods idDatabaseHash otherMods version", "K", "版本信息")

# ---------------------------------------------------------------- odds
add("Odds.AbstractOdds", "<CurrentValue>", "G",
    "G7：卡牌稀有度保底偏移（CardRarityOdds.cs:69-80，-5% 起，每张非稀有 +1%/A7 +0.5%，出稀有归位）与药水掉率（PotionRewardOdds.cs:55-71，40% 起 ±10%）；"
    "都是公开结果的确定函数，熟手可算，未导出")
add("Odds.AbstractOdds", "_rng", "H", "掷骰随机数")
add("Odds.PlayerOddsSet", "<CardRarity> <PotionReward>", "G", "G7：容器，值见 AbstractOdds.CurrentValue")
add("Odds.RunOddsSet", "<UnknownMapPoint>", "G", "G7：问号房概率容器，见 UnknownMapPointOdds._nonEventOdds")
add("Odds.UnknownMapPointOdds", "_baseOdds", "K", "基础概率（10%/2%/3%，开局可被遗物/难度改，之后不变）")
add("Odds.UnknownMapPointOdds", "_nonEventOdds", "G",
    "G7：问号房的当前怪物/宝箱/商店概率，每进一个问号房按公开结果更新、每幕重置（UnknownMapPointOdds.cs:123-178，RunManager.cs:1385）；熟手据此规划路线，未导出")

# ---------------------------------------------------------------- random
add("Random.MegaRandom", "_s0 _s1 _s2 _s3", "H", "随机数发生器状态")
add("Random.PlayerRngSet", "<Seed> _rngs", "H", "玩家随机数种子与流")
add("Random.Rng", "_counter _random", "H", "随机数流与已用次数")

# ---------------------------------------------------------------- rewards
add("Rewards.GoldReward", "<Amount>", "E", "reward 候选 n")
add("Rewards.GoldReward", "_max _min", "K", "金币区间（已掷出 Amount）")
add("Rewards.GoldReward", "_wasGoldStolenBack", "K", "'夺回的金币'文案标志")
add("Rewards.PotionReward", "<ClaimedPotion> _icon", "K", "领取结果/图标")
add("Rewards.PotionReward", "<Potion>", "G",
    "G6：可领取时为 reward 候选的 id；药水栏满时整条奖励被跳过（Rl.cs:1121），无法判断'弃一瓶换它'，见 R#4")
add("Rewards.PotionReward", "_wasTaken", "D", "已领取的奖励不再出现在候选")
add("Rewards.RelicReward", "<ClaimedRelic>", "K", "领取结果")
add("Rewards.RelicReward", "_predeterminedRelic _relic", "E", "reward 候选 id（Relic 是纯 getter；战斗中额外奖励未 Populate 时为 null，不提前掷）")
add("Rewards.RelicReward", "_rarity", "E", "reward 候选 rar")
add("Rewards.RelicReward", "_wasTaken", "D", "已领取不再出现")
add("Rewards.Reward", "<ParentRewardSet> <Player>", "K", "关联奖励组/引用（Linked 奖励以 rt=Linked 导出）")
add("Rewards.Reward", "<SuccessfullySelected>", "E", "已领取的奖励不进候选")
add("Rewards.Reward", "_rngOverride", "H", "奖励专用随机数")
add("Rewards.SpecialCardReward", "_card", "E", "reward 候选 card")
add("Rewards.SpecialCardReward", "_customDescriptionEncounterSourceId", "K", "文案来源")
add("Rewards.SpecialCardReward", "_wasTaken", "D", "已领取不再出现")

# ---------------------------------------------------------------- rooms
add("Rooms.AbstractRoom", "<Id>", "K", "房间序号")
add("Rooms.CombatRoom", "<CombatState>", "K", "引用")
add("Rooms.CombatRoom", "<GoldProportion>", "D", "战斗结束时才算（CombatRoom.cs:235），结果在金币奖励里")
add("Rooms.CombatRoom", "<ParentEventId> <ShouldCreateCombat> <ShouldResumeParentEventAfterCombat>", "K", "事件战斗的流程标志")
add("Rooms.CombatRoom", "_extraRewards", "D", "战斗中累积的额外奖励（夺回的金币/牌、The Hunt、事件许诺），来源都是可见事件，战后在奖励界面导出")
add("Rooms.CombatRoom", "_isPreFinished", "K", "流程标志")
add("Rooms.EventRoom", "<CanonicalEvent>", "E", "ctx.event")
add("Rooms.EventRoom", "_isPreFinished", "K", "流程标志")
add("Rooms.MerchantRoom", "<Inventories>", "E", "商店候选（缺口见 G6）")
add("Rooms.MerchantRoom", "_runState", "K", "引用")
add("Rooms.RestSiteRoom", "_synchronizer", "K", "多人同步")
add("Rooms.RoomSet", "<SecondBoss> _boss", "E", "当前幕的 g.boss / g.boss2（地图上画着）；后续幕的为 H")
add("Rooms.RoomSet", "_ancient", "D", "本幕先古事件：起点节点即它，进入后 ctx.event；后续幕的为 H")
add("Rooms.RoomSet", "bossEncountersVisited", "D", "已打 boss 数 = 当前是否第二个 boss，可由地图位置推出")
add("Rooms.RoomSet", "eliteEncounters normalEncounters events", "H", "本幕预生成的精英/普通遭遇/事件队列")
add("Rooms.RoomSet", "eliteEncountersVisited", "G",
    "G5：本幕已打精英数；精英按 3 种的无放回抽袋排队（ActModel.cs:372-383），打过两种后第三个精英就确定了，熟手据此选路，未导出")
add("Rooms.RoomSet", "eventsVisited", "D", "本幕已遇事件数，只是隐藏事件队列的下标；本身无决策意义")
add("Rooms.RoomSet", "normalEncountersVisited", "G",
    "G5：本幕已打普通战斗数；前 2-3 场来自弱怪池（ActModel.cs:349-371，NumberOfWeakEncounters），熟手知道下一场是否进入强怪池，未导出")
add("Rooms.TreasureRoom", "_runState", "K", "引用")

# ---------------------------------------------------------------- run
add("Runs.ExtraRunFields", "<FreedRepy>", "K", "只影响 Queen 战的背景特效（NQueenRepyBgVfx.cs:55）")
add("Runs.ExtraRunFields", "<StartedWithNeow>", "K", "开局流程标志")
add("Runs.ExtraRunFields", "<TestSubjectKills>", "K", "TestSubject 标题里的计数（装饰）")
add("Runs.History.AncientChoiceHistoryEntry", "<Title> <WasChosen>", "K", "跑局记录（结算界面用）；结果已体现在 deck/relics")
add("Runs.History.CardChoiceHistoryEntry", "<Card> wasPicked", "K", "跑局记录；稀有度保底的影响见 G7")
add("Runs.History.CardEnchantmentHistoryEntry", "<Card> <Enchantment>", "K", "跑局记录")
add("Runs.History.CardTransformationHistoryEntry", "<FinalCard> <OriginalCard>", "K", "跑局记录")
add("Runs.History.EventOptionHistoryEntry", "<Title> <Variables>", "K", "跑局记录")
add("Runs.History.MapPointHistoryEntry", "<MapPointType> <PlayerStats>", "K", "跑局记录")
add("Runs.History.MapPointHistoryEntry", "<Rooms>", "D", "本幕进过的房间；所需摘要见 MapPointRoomHistoryEntry.ModelId")
add("Runs.History.MapPointRoomHistoryEntry", "<ModelId>", "G",
    "G5：本幕打过的遭遇 id（及问号房实际类型、遇过的事件）是推断剩余遭遇袋与问号房概率的公开历史，未导出")
add("Runs.History.MapPointRoomHistoryEntry", "<MonsterIds> <RoomType> <TurnsTaken>", "D", "同上的细节；需要的摘要是遭遇 id 与房间类型")
add("Runs.History.ModelChoiceHistoryEntry", "choice wasPicked", "K", "跑局记录")
add("Runs.MapLocation", "actIndex coord", "K", "多人消息位置")
add("Runs.PlayerMapPointHistoryEntry",
    "<AncientChoices> <BoughtColorless> <BoughtPotions> <BoughtRelics> <CardChoices> <CardsEnchanted> <CardsGained> <CardsRemoved> "
    "<CardsTransformed> <CompletedQuests> <CurrentGold> <CurrentHp> <DamageTaken> <DowngradedCards> <EventChoices> <GoldGained> "
    "<GoldLost> <GoldSpent> <GoldStolen> <HpHealed> <IsAffectedByFurCoat> <MaxHp> <MaxHpGained> <MaxHpLost> <PlayerId> "
    "<PotionChoices> <PotionDiscarded> <PotionUsed> <RelicChoices> <RelicsRemoved> <RestSiteChoices> <StolenLoot> <UpgradedCards>",
    "K", "每层的跑局统计（只写不读，供结算界面）；结果已体现在 deck/relics/gold/hp")
add("Runs.RelicGrabBag", "_deques _mpFallbackDequeue _originalRelics", "H", "预洗好的遗物顺序与池子")
add("Runs.RelicGrabBag", "_refreshAllowed", "K", "规则常量")
add("Runs.RunLocation", "mapLocation roomId", "K", "多人消息位置")
add("Runs.RunRngSet", "<Seed> <StringSeed> _rngs", "H", "跑局种子与随机数流")
add("Runs.RunState", "<ActFloor>", "E", "g.floor")
add("Runs.RunState", "<Acts>", "H", "三幕的模型与预生成房间；当前幕 id/boss 已导出，后续幕的 boss 与队列为 H")
add("Runs.RunState", "<AscensionLevel>", "E", "g.asc")
add("Runs.RunState", "<BadgeModels> <GameMode> <MultiplayerScalingModel> <NextRoomId> <UnlockState>", "K", "徽章/模式/计数器/档案")
add("Runs.RunState", "<ExtraFields>", "K", "见 ExtraRunFields")
add("Runs.RunState", "<Map>", "E", "obs.map")
add("Runs.RunState", "<Modifiers>", "K", "标准模式恒空；启用修改器时需导出")
add("Runs.RunState", "<Odds>", "G", "G7：容器，见 UnknownMapPointOdds")
add("Runs.RunState", "<Rng>", "H", "跑局随机数")
add("Runs.RunState", "<SharedRelicGrabBag>", "H", "共享遗物袋")
add("Runs.RunState", "_allCards _players", "K", "注册表/引用")
add("Runs.RunState", "_currentActIndex", "E", "g.act")
add("Runs.RunState", "_currentRooms", "E", "当前房间（g.room，事件战斗时为事件+战斗两层）")
add("Runs.RunState", "_mapPointHistory", "K", "跑局记录容器；需要的摘要见 G5")
add("Runs.RunState", "_visitedEventIds", "G",
    "G14（可忽略）：遇过的事件不再出现（RoomSet.EnsureNextEventIsValid），只轻微改变问号房事件分布，未导出")
add("Runs.RunState", "_visitedMapCoords", "D", "走过的路径；只有当前节点影响后续（WingedBoots 只用计数，与楼层同）")
add("Saves.Runs.SavedProperties", "bools cardArrays cards intArrays ints modelIds strings", "K", "存档序列化容器（跑局记录里的卡牌快照）")
add("Saves.Runs.SavedProperties+SavedProperty`1[Int32]", "name value", "K", "存档序列化")
add("Saves.Runs.SerializableCard", "<CurrentUpgradeLevel> <Enchantment> <FloorAddedToDeck> <Id> <Props>", "K",
    "存档卡牌快照（跑局记录）；ArchaicTooth 持有的快照经 relics[].cards 导出 id")
add("Saves.Runs.SerializableEnchantment", "<Amount> <Id> <Props>", "K", "存档附魔快照")
add("Unlocks.UnlockState", "<NumberOfRuns> _encountersSeen _unlockedEpochIds", "K", "档案解锁进度（引擎固定全解锁）")

# ---------------------------------------------------------------- statics
S = "static:"
add(S + "AutoSlay.AutoSlayLog", "_fileWriter", "K", "自动测试工具")
add(S + "AutoSlay.AutoSlayer", "_exitCode", "K", "自动测试工具")
add(S + "AutoSlay.MemoryProfiler", "_baseline _previous _snapshotCount", "K", "内存分析")
add(S + "Combat.CombatManager", "_staleTurnEndReported _turnLoopWaitTimeoutReported", "K", "日志去重标志")
add(S + "Combat.CombatTurnState", "_sequenceCounter", "K", "CombatId 进程级计数器；只用于比较同一战斗，数值不影响玩法")
add(S + "Debug.AssemblyHasher", "_mainAssemblyHash", "K", "调试")
add(S + "Debug.ReleaseInfoManager", "_instance", "K", "版本信息单例")
add(S + "Debug.SentryService", "_isGameInitialized _isHeadless _nativeEnvironment _platformBranchResolved _sampleRate _suppressAllEvents", "K", "崩溃上报")
add(S + "DevConsole.ConsoleCommands.ApplyPowerConsoleCmd", "_allPowers", "K", "控制台缓存")
add(S + "DevConsole.ConsoleCommands.CloudConsoleCmd", "_confirmed", "K", "控制台")
add(S + "GameActions.Multiplayer.ActionTypes", "_cache", "K", "类型缓存")
add(S + "Helpers.OneTimeInitialization", "_atlasResourceLoader _state", "K", "初始化")
add(S + "Helpers.ReflectionHelper", "_allTypes _modTypes", "K", "反射缓存")
add(S + "Helpers.StringHelper", "_stringHashCache", "K", "哈希缓冲")
add(S + "Leaderboard.LeaderboardManager", "_strategy", "K", "排行榜")
add(S + "Localization.LocManager", "_smartFormatter", "K", "本地化")
add(S + "Modding.ModManager", "_allowInitForTests _fileIo _gameVersion _hasHarmonyPatches _mods _settings _steamItemInstalledCallback", "K", "模组管理")
add(S + "Models.Events.FakeMerchant", "_dialogue", "K", "对白")
add(S + "Models.ModelDb",
    "_achievements _acts _actsByIndex _allAbstractModelSubtypes _allCardPools _allCards _allCharacterCardPools _allCharacterPotionPools "
    "_allCharacterRelicPools _allEncounters _allEvents _allPotionPools _allPotions _allPowers _allRelics _allSharedEvents "
    "_allSharedPotionPools _badges _eventEncounters", "K", "规范内容数据库的惰性缓存")
add(S + "Models.Powers.Mocks.MockGatedTurnStartPower", "_heldGate", "K", "测试 mock")
add(S + "Models.Relics.YummyCookie", "_cachedRandomCharacter", "K", "未拥有时图标随机选的角色（Rng.Chaotic），纯显示")
add(S + "Multiplayer.Connection.HandshakeManager", "skipTimeoutTcs", "K", "联网")
add(S + "Multiplayer.Serialization.MessageTypes", "_cache", "K", "类型缓存")
add(S + "Multiplayer.Serialization.ModelIdSerializationCache", "_initialized", "K", "序列化缓存")
add(S + "Odds.CardRarityOdds", "regularCommonOdds", "K",
    "跑局级常量（A7 Scarcity 下 0.615），但游戏把它写成静态字段初始化一次；引擎每局按当前难度重置（Harness.cs:192-196）。可由 g.asc 推出")
add(S + "Platform.Steam.SteamInitializer", "_disconnectCts", "K", "平台")
add(S + "Platform.Steam.SteamStatsManager", "_globalArchitectDamage _globalStatsReady _userStatsReady _userStatsReceivedCallback", "K", "平台统计")
add(S + "Rooms.MerchantRoom", "_dialogue", "K", "对白")
add(S + "Saves.AccountScopeUserDataMigrator", "_migrationPerformed", "K", "存档迁移")
add(S + "Saves.Managers.ProgressSaveManager", "_eliteEncounters", "K", "档案统计缓存")
add(S + "Saves.ProfileAccountScopeMigrator", "_migrationPerformed", "K", "存档迁移")
add(S + "Saves.SaveManager", "_instance _mockInstance", "K", "存档单例")
add(S + "TestSupport.TestRngInjector", "_combatCardGenerationOverride _relicOverride _relicRarityOverride", "K", "测试注入（恒 null）")
add(S + "Timeline.EpochModel", "_agnosticUnlockOrder _agnosticUnlocks _allEpochIds _epochIdsHashSet", "K", "解锁时间线缓存")
add(S + "MegaCrit.Sts2.addons.mega_text.MegaLabel", "_cachedParagraph", "K", "UI 文本缓存")
add(S + "MegaCrit.Sts2.addons.mega_text.MegaRichTextLabel", "_cachedParagraph", "K", "UI 文本缓存")


# Status of each affected row at HEAD b7e59be (fixes committed while this audit ran); the class above is relative to
# the audited snapshot (tmp/obs_audit/engine_snapshot = RunSimulator.Rl.cs before b7e59be).
FIXED = "HEAD b7e59be 已修："
OPEN = "HEAD b7e59be 仍缺"
HEAD = {
    "Combat.CombatState::_enemies": FIXED + "CombatEnemies() 含等待复活的敌人（alive=false）",
    "Entities.Players.PlayerCombatState::<DrawPile>": FIXED + "combat.draw_top（KnownTop.cs）；draw 按完整卡牌 JSON 排序",
    "Models.CardModel::_baseReplayCount": FIXED + "card.replay",
    "Models.CardModel::_hasSingleTurnRetain": FIXED + "kw 补 Retain",
    "Models.CardModel::_hasSingleTurnSly": FIXED + "kw 补 Sly",
    "Combat.History.Entries.CardPlayFinishedEntry::<CardPlay>": FIXED + "combat.turn.first/attacks/skills/powers/last_attack，card.played_turn",
    "Combat.History.Entries.CardDrawnEntry::<Card>": FIXED + "combat.turn.status_drawn / drawn_extra",
    "Combat.History.Entries.CardExhaustedEntry::<Card>": FIXED + "combat.turn.exhausted，手牌 glow",
    "Combat.History.Entries.DamageReceivedEntry::<Result>": FIXED + "combat.turn.hp_lost，手牌 glow",
    "Combat.History.Entries.PowerReceivedEntry::<Power>": FIXED + "经手牌 glow（Death's Door）",
    "Combat.History.Entries.CreatureAttackedEntry::<DamageResults>": FIXED + "经手牌 glow（Flatten）",
    "Entities.Merchant.MerchantInventory::_characterCardEntries": FIXED + "ctx.unavailable",
    "Entities.Merchant.MerchantInventory::_colorlessCardEntries": FIXED + "ctx.unavailable",
    "Entities.Merchant.MerchantInventory::_relicEntries": FIXED + "ctx.unavailable",
    "Entities.Merchant.MerchantInventory::_potionEntries": FIXED + "ctx.unavailable",
    "Rewards.PotionReward::<Potion>": FIXED + "ctx.blocked",
    "Models.Cards.MadScience::_tinkerTimeRider": FIXED + "OwnState 收文本型 [SavedProperty]",
    "Models.Cards.MadScience::_tinkerTimeType": FIXED + "OwnState 收文本型 [SavedProperty]",
    "Models.EnchantmentModel::_status": FIXED + "card.ench_s",
    "Entities.Players.PlayerCombatState::<TurnNumber>": FIXED + "combat.pturn",
    "Combat.CombatState::_encounter": FIXED + "combat.enc",
    "Models.EncounterModel::_canonicalInstance": FIXED + "combat.enc",
    "Entities.Creatures.Creature::<SlotName>": FIXED + "敌人 slot",
    "Models.Monsters.Fabricator::_lastSpawned": FIXED + "OwnState 把模型引用导出为 id",
    "MonsterMoves.MonsterMoveStateMachine.MonsterMoveStateMachine::<StateLog>": FIXED + "敌人 done（每招已执行次数）",
    "Models.Powers.PossessSpeedPower+Data::stolenDexterity": FIXED + "d.stolenDexterity.sum",
    "Models.Powers.PossessStrengthPower+Data::stolenStrength": FIXED + "d.stolenStrength.sum",
    "Models.Powers.DampenPower+Data::downgradedCardsToOldUpgradeLevels": "HEAD b7e59be 部分：只导出原等级之和（.sum），不知道是哪张牌被降级",
    "Models.Relics.SeaGlass::_characterId": FIXED + "事件选项 relic_d 含 CharacterId",
    "Rooms.RoomSet::eliteEncountersVisited": OPEN,
    "Rooms.RoomSet::normalEncountersVisited": OPEN,
    "Runs.History.MapPointRoomHistoryEntry::<ModelId>": OPEN,
    "Odds.AbstractOdds::<CurrentValue>": OPEN,
    "Odds.UnknownMapPointOdds::_nonEventOdds": OPEN,
    "Odds.PlayerOddsSet::<CardRarity>": OPEN,
    "Odds.PlayerOddsSet::<PotionReward>": OPEN,
    "Odds.RunOddsSet::<UnknownMapPoint>": OPEN,
    "Entities.Players.Player::<PlayerOdds>": OPEN,
    "Runs.RunState::<Odds>": OPEN,
    "Entities.Cards.CardEnergyCost::_localModifiers": OPEN,
    "Entities.Cards.LocalCostModifier::<Expiration>": OPEN,
    "Entities.Cards.TemporaryCardCost::<ClearsWhenCardIsPlayed>": OPEN,
    "Entities.Cards.TemporaryCardCost::<ClearsWhenTurnEnds>": OPEN,
    "Entities.Players.ExtraPlayerFields::<CardShopRemovalsUsed>": OPEN,
    "Runs.RunState::_visitedEventIds": OPEN,
    # E rows whose export changed at HEAD
    "Models.Monsters.CorpseSlug::_starterMoveIdx": "HEAD b7e59be：列入 _cosmeticFields，不再导出（M-F5 已处理）",
    "Models.Monsters.DecimillipedeSegment::_starterMoveIdx": "HEAD b7e59be：列入 _cosmeticFields，不再导出",
    "Models.Monsters.ScrollOfBiting::_starterMoveIdx": "HEAD b7e59be：列入 _cosmeticFields，不再导出",
    "Models.Monsters.TwoTailedRat::_starterMoveIndex": "HEAD b7e59be：列入 _cosmeticFields，不再导出",
    "Models.Monsters.WaterfallGiant::_pressureBuildupIdx": "HEAD b7e59be：列入 _cosmeticFields，不再导出",
    "Models.Powers.SandpitPower::_initialAmount": "HEAD b7e59be：列入 _cosmeticFields，不再导出",
    "Models.Powers.SandpitPower::_initialTargetPosition": "HEAD b7e59be：列入 _cosmeticFields，不再导出",
    "Entities.Orbs.OrbQueue::_orbs": "HEAD b7e59be：orbs[].slot 已加（R#2）",
    "Models.Powers.SwipePower::_stolenCard": "HEAD b7e59be：cards 带 +升级与 @附魔（M-F16）",
    "Entities.Cards.CardPile::_cards": "HEAD b7e59be：draw 按完整卡牌 JSON 排序（R#15），已知顶牌见 draw_top",
}


def key_of(r):
    t = r["type"].replace("MegaCrit.Sts2.Core.", "")
    if t.startswith("Saves.Runs.SavedProperties+SavedProperty"):
        t = "Saves.Runs.SavedProperties+SavedProperty`1[Int32]"
    f = r["field"].replace("k__BackingField", "")
    return ("static:" if r["kind"] == "static" else "") + f"{t}::{f}"


def main():
    rows = list(csv.DictReader(open(SRC), delimiter="\t"))
    missing, used = [], set()
    out = []
    for r in rows:
        k = key_of(r)
        if k not in C:
            missing.append(k)
            continue
        used.add(k)
        cls, why = C[k]
        hk = k.split("::")[0] + "::" + k.split("::")[1].split(">")[0] + (">" if ">" in k.split("::")[1] else "")
        note = HEAD.get(k) or HEAD.get(hk)
        if note:
            why = f"{why}【{note}】"
        out.append(dict(r, **{"class": cls, "how_or_why": why}))
    unused = sorted(set(C) - used)
    bad = [h for h in HEAD if h not in C]
    if bad:
        print("HEAD keys not in table:", bad)
        sys.exit(1)
    if missing or unused:
        print("missing:", *missing, sep="\n  ")
        print("unused:", *unused, sep="\n  ")
        sys.exit(1)
    cols = list(rows[0].keys()) + ["class", "how_or_why"]
    with open(OUT, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(out)
    tot = Counter(o["class"] for o in out)
    by = Counter((o["kind"], o["class"]) for o in out)
    print(len(out), "rows ->", OUT)
    print("total:", dict(sorted(tot.items())))
    print("instance:", {c: by[("instance", c)] for c in "EHDKG"}, "static:", {c: by[("static", c)] for c in "EHDKG"})


if __name__ == "__main__":
    main()
