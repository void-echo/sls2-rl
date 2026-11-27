#!/usr/bin/env python3
"""v0.111.0/v0.113.1 patch-ablation scenarios. Each case runs with its own ID selected and with no reverts.
Expected values and hook ordering come from the two decompiled model definitions.
Run from the worktree: python3 tools/revert_tests.py [MODEL.ID ...] --workers 6
"""
import argparse
import concurrent.futures as cf
import multiprocessing
import os
import time
import traceback
from engine_runner import Engine
from scenario_tests import S

TESTS = {}

def case(id):
    def register(fn):
        TESTS[id] = fn
        return fn
    return register

class R(S):
    def __init__(self, eng, char='Ironclad', relics=(), encounter='SLIMES_WEAK', room='Monster', deck=None):
        super().__init__(eng, char=char, seed='revert_case')
        self.map()
        self.send(dict(cmd='set_player', hp=9000, max_hp=10000, relics=list(relics),
                       potions=[], deck=deck or ['STRIKE_IRONCLAD','DEFEND_IRONCLAD','BASH','ANGER','POMMEL_STRIKE']))
        self.send(dict(cmd='rl_enter_room',type=room,encounter=encounter))
    def dbg(self, op, **kw):
        return self.send(dict(cmd='dbg_revert',op=op,**kw))
    def state(self):
        return self.send(dict(cmd='dbg_revert_state'))
    def setup(self, hand=(), draw=(), discard=(), exhaust=(), **kw):
        self.dbg('setup',hand=list(hand),draw=list(draw),discard=list(discard),exhaust=list(exhaust),**kw)
    def play(self, id):
        for i,c in enumerate(self.r['cands']):
            if c['k']=='play' and self.hand[c['hand']]['id']==id:
                return self.act(i)
        raise AssertionError((id,'not playable',self.r['cands'],self.hand))
    def power(self,id,n=1,**kw):self.dbg('power',id=id,n=n,**kw)
    def invoke(self,subject,member,*args,**kw):
        self.dbg('invoke',subject=subject,member=member,args=list(args),**kw)
        self.resolve()
        return self.state().get('value')
    def field(self,subject,member,value):self.dbg('field',subject=subject,member=member,value=value)
    def get(self,subject,member):return self.dbg('get',subject=subject,member=member).get('value')
    def resolve(self):
        for _ in range(100):
            if self.r['decision'] in ('combat_play','map_select','event_choice') or self.r['done']:return
            c=self.r['cands']
            indices=[i for i,x in enumerate(c) if x['k'] in ('pick','confirm','reward','reward_card','proceed','bundle')]
            self.act(indices[0] if indices else 0)
        raise AssertionError('selection cap')
    @property
    def combat(self):return self.r['obs']['combat']
    @property
    def hand(self):return self.combat['hand']
    @property
    def hp(self):return self.r['obs']['g']['hp']
    def card(self,id):return next(c for c in self.hand if c['id']==id)
    def amount(self,id,target='player'):
        obj=self.combat['player'] if target=='player' else self.combat['enemies'][0]
        return next((p['n'] for p in obj['pw'] if p['id']==id),0)

for _id in ['DEBT','HAND_OF_GREED','ROYALTIES']:
    def blitz_card(eng,old,id=_id):
        s=R(eng);s.setup([id])
        assert s.get('CARD.'+id,'IsAllowedInBlitz')==old
    TESTS['CARD.'+_id]=blitz_card

@case('RELIC.POTION_BELT')
def belt(eng,old):
    s=R(eng,relics=['POTION_BELT']);s.setup(['DEFEND_IRONCLAD'])
    assert s.invoke('RELIC.POTION_BELT','IsAllowedInternal','$late_blitz')==old

@case('CARD.EXPECT_A_FIGHT')
def expect(eng,old):
    for up in (0,1):
        s=R(eng);s.setup([dict(id='EXPECT_A_FIGHT',up=up),'STRIKE_IRONCLAD','WHIRLWIND'])
        s.power('STRENGTH_POWER',3)
        c=s.card('CARD.EXPECT_A_FIGHT')
        assert ('CalculatedBlock' in c['v'])==old,c
        s.play('CARD.EXPECT_A_FIGHT')
        assert s.combat['player']['block']==((40 if up else 30) if old else (20 if up else 15)),s.combat
        assert s.card('CARD.STRIKE_IRONCLAD')['cost']==(1 if old else 0)
        assert s.card('CARD.WHIRLWIND')['x']

@case('CARD.FORGOTTEN_RITUAL')
def ritual(eng,old):
    s=R(eng);s.setup(['FORGOTTEN_RITUAL'],energy=1);s.play('CARD.FORGOTTEN_RITUAL')
    assert s.combat['energy']==(3 if old else 0),s.combat
    s=R(eng);s.setup(['FORGOTTEN_RITUAL'],energy=1)
    s.invoke('$history','CardExhausted','$combat','$card')
    s.play('CARD.FORGOTTEN_RITUAL');assert s.combat['energy']==3

for _id,_old,_new,_delta in [('SALVO',12,10,4),('MINION_DIVE_BOMB',13,11,None)]:
    def damage(eng,old,id=_id,a=_old,b=_new,delta=_delta):
        for up in (0,1):
            s=R(eng);s.setup([dict(id=id,up=up)])
            val=(a if old else b)+((delta if delta is not None else (3 if old else 2)) if up else 0)
            assert s.card('CARD.'+id)['v']['Damage']==val
            hp=s.combat['enemies'][0]['hp'];s.play('CARD.'+id)
            assert hp-s.combat['enemies'][0]['hp']==val,s.combat
    TESTS['CARD.'+_id]=damage

for _id,_var,_old,_new,_delta in [('PHANTOM_BLADES','PhantomBladesPower',9,8,3),('SPEEDSTER','SpeedsterPower',2,3,0)]:
    def buff(eng,old,id=_id,var=_var,a=_old,b=_new,d=_delta):
        for up in (0,1):
            s=R(eng,'Silent');s.setup([dict(id=id,up=up)])
            val=(a if old else b)+d*up
            assert s.card('CARD.'+id)['v'][var]==val
            s.play('CARD.'+id)
            assert s.amount('POWER.'+('PHANTOM_BLADES_POWER' if var=='PhantomBladesPower' else 'SPEEDSTER_POWER'))==val,s.combat
    TESTS['CARD.'+_id]=buff

@case('CARD.MEMENTO_MORI')
def memento(eng,old):
    for up in (0,1):
        s=R(eng,'Silent');s.setup([dict(id='MEMENTO_MORI',up=up),'PREPARED','DEFEND_SILENT'])
        c=s.card('CARD.MEMENTO_MORI');assert c['cost']==(1 if old else 3)
        assert ('Sly' in c['kw'])==(not old);assert c['tgt']==('AnyEnemy' if old else 'RandomEnemy')
        hp=s.combat['enemies'][0]['hp'];s.invoke('$history','CardDiscarded','$combat','$CARD.DEFEND_SILENT')
        s.invoke('$history','CardDiscarded','$combat','$CARD.DEFEND_SILENT')
        s.play('CARD.MEMENTO_MORI')
        assert hp-s.combat['enemies'][0]['hp']==((21 if up else 17) if old else (23 if up else 19)),s.combat

@case('CARD.ENLIGHTENMENT')
def enlightenment(eng,old):
    s=R(eng);s.setup(['ENLIGHTENMENT','BLUDGEON']);s.play('CARD.ENLIGHTENMENT')
    assert s.card('CARD.BLUDGEON')['cost']==1
    pile='exhaust' if old else 'discard'
    assert any(c['id']=='CARD.ENLIGHTENMENT' for c in s.combat[pile])

@case('CARD.EXPERTISE')
def expertise(eng,old):
    s=R(eng,'Silent');s.setup(['EXPERTISE','DEFEND_SILENT'],draw=['STRIKE_SILENT','DAGGER_SPRAY'])
    s.play('CARD.EXPERTISE')
    assert 'Retain' not in s.card('CARD.DEFEND_SILENT')['kw']
    assert s.amount('POWER.RETAIN_HAND_POWER')==(0 if old else 1)
    assert all(('Retain' in s.card(id)['kw'])==old for id in ['CARD.STRIKE_SILENT','CARD.DAGGER_SPRAY'])

@case('CARD.DOWSING')
def dowsing(eng,old):
    s=R(eng);s.setup(['DOWSING']);assert s.card('CARD.DOWSING')['v']['Rooms']==(5 if old else 4)
    s.field('CARD.DOWSING','RoomsEntered',3)
    assert s.card('CARD.DOWSING')['v']['Rooms']==(2 if old else 1)
    types=s.invoke('CARD.DOWSING','ModifyUnknownMapPointRoomTypes','$rooms')
    assert ('Monster' in types)==old and ('Elite' in types)==old,types

@case('CARD.BYRDONIS_EGG')
def egg(eng,old):
    s=R(eng);s.setup(['BYRDONIS_EGG'])
    assert s.invoke('CARD.BYRDONIS_EGG','TryModifyRestSiteOptions','$owner','$hatch_options')==old

for _id,_pile,_oldrng in [('BEAT_DOWN','discard','Shuffle'),('CATASTROPHE','draw','Shuffle'),('UPROAR','draw','Shuffle')]:
    def autoplay(eng,old,id=_id,pile=_pile,stream=_oldrng):
        s=R(eng);kw={pile:['STRIKE_IRONCLAD','BASH','ANGER','BLUDGEON']}
        s.setup([id],**kw);before=s.state();s.play('CARD.'+id);after=s.state()
        chosen=stream if old else 'CombatCardSelection'
        assert after['run_rng'][chosen]>before['run_rng'][chosen],(before,after)
        other='CombatCardSelection' if old else stream
        assert after['run_rng'][other]==before['run_rng'][other],(before,after)
        assert any(e['hp']<10000 for e in s.combat['enemies'])
    TESTS['CARD.'+_id]=autoplay

@case('CARD.HELIX_DRILL')
def helix(eng,old):
    s=R(eng,'Defect');s.setup([dict(id='HELIX_DRILL',cost=1)])
    s.invoke('$history','EnergySpent','$combat',1,'$owner')
    s.dbg('move',subject='CARD.HELIX_DRILL',pile='Play')
    s.invoke('$history','CardPlayStarted','$combat','$play')
    s.invoke('CARD.HELIX_DRILL','OnPlay','$context','$play')
    # Outside a real play, old subtracts the live modified cost; new uses the captured zero expenditure.
    damage=10000-s.combat['enemies'][0]['hp']
    assert damage==(0 if old else 3),damage

@case('CARD.OMNISLICE')
def omnislice(eng,old):
    s=R(eng);s.setup(['OMNISLICE']);s.power('THORNS_POWER',2,target='enemy2')
    s.play('CARD.OMNISLICE')
    assert s.hp==(9000 if old else 8998),s.hp
    assert all(e['hp']==9992 for e in s.combat['enemies']),s.combat

@case('CARD.SPOILS_MAP')
def spoils(eng,old):
    s=R(eng);s.setup(['SPOILS_MAP'])
    s.dbg('spoils_coord')
    before=s.state()['spoils_quests'];s.dbg('move',subject='CARD.SPOILS_MAP',pile='Deck')
    assert s.state()['spoils_quests']-before==(0 if old else 1)

# Random-source regressions: invoke the real acquisition/combat hook and inspect the consumed game RNG streams.
RNG_RELICS={
 'ALCHEMICAL_COFFER':('AfterObtained',(),('run_rng','CombatPotionGeneration'),('player_rng','NonCombatGeneration')),
 'ASTROLABE':('AfterObtained',(),('run_rng','Niche'),('player_rng','Transformations')),
 'BEAUTIFUL_BRACELET':('AfterObtained',(),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'DELICATE_FROND':('BeforeCombatStart',(),('run_rng','CombatPotionGeneration'),('player_rng','NonCombatGeneration')),
 'DISTINGUISHED_CAPE':('AfterObtained',(),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'FISHING_ROD':('AfterCombatEnd',('$room',),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'FRAGRANT_MUSHROOM':('AfterObtained',(),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'KALEIDOSCOPE':('AfterObtained',(),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'NEOWS_BONES':('AfterObtained',(),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'NEW_LEAF':('AfterObtained',(),('run_rng','Niche'),('player_rng','Transformations')),
 'PANDORAS_BOX':('AfterObtained',(),('run_rng','Niche'),('player_rng','Transformations')),
 'PHIAL_HOLSTER':('AfterObtained',(),('run_rng','CombatPotionGeneration'),('player_rng','NonCombatGeneration')),
 'ROYAL_STAMP':('AfterObtained',(),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'SAND_CASTLE':('AfterObtained',(),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'WAR_HAMMER':('AfterCombatVictory',('$room',),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'WAR_PAINT':('AfterObtained',(),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'WHETSTONE':('AfterObtained',(),('run_rng','Niche'),('player_rng','NonCombatGeneration')),
 'WING_CHARM':('TryModifyCardRewardOptionsLate',('$owner','$card_rewards',None),('run_rng','Niche'),('player_rng','Rewards')),
}
for _id,(_hook,_args,_a,_b) in RNG_RELICS.items():
    def rng_relic(eng,old,id=_id,hook=_hook,args=_args,a=_a,b=_b):
        s=R(eng,relics=[id],encounter='BYRDONIS_ELITE' if id=='WAR_HAMMER' else 'SLIMES_WEAK',room='Elite' if id=='WAR_HAMMER' else 'Monster')
        s.setup(['DEFEND_IRONCLAD','STRIKE_IRONCLAD'])
        if id=='DELICATE_FROND':s.send(dict(cmd='set_player',potions=[]))
        if id=='FISHING_ROD':s.field('RELIC.'+id,'CombatsSeen',2)
        before=s.state();s.invoke('RELIC.'+id,hook,*args);after=s.state()
        bucket,stream=a if old else b
        assert after[bucket][stream]>before[bucket][stream],(id,before,after)
        bucket,stream=b if old else a
        # Other reward effects can consume Rewards; the migrated Niche stream must remain untouched when off.
        if stream!='Rewards':assert after[bucket][stream]==before[bucket][stream],(id,before,after)
    TESTS['RELIC.'+_id]=rng_relic

@case('POTION.ENTROPIC_BREW')
def brew(eng,old):
    s=R(eng);s.setup(['DEFEND_IRONCLAD']);s.send(dict(cmd='dbg_procure_potion',id='ENTROPIC_BREW'))
    before=s.state();s.act(next(i for i,c in enumerate(s.r['cands']) if c['k']=='potion'));after=s.state()
    a,b=('run_rng','CombatPotionGeneration') if old else ('player_rng','NonCombatGeneration')
    assert after[a][b]>before[a][b]
    assert len(s.r['obs']['potions'])==3

@case('RELIC.BRILLIANT_SCARF')
def scarf(eng,old):
    s=R(eng,relics=['BRILLIANT_SCARF']);s.setup(['DEFEND_IRONCLAD'])
    s.field('RELIC.BRILLIANT_SCARF','CardsPlayedThisTurn',4)
    assert s.card('CARD.DEFEND_IRONCLAD')['cost']==(0 if old else 1)
    assert next(x for x in s.r['obs']['relics'] if x['id']=='RELIC.BRILLIANT_SCARF')['v']['Cards']==(5 if old else 6)

@case('RELIC.JEWELED_MASK')
def mask(eng,old):
    s=R(eng,relics=['JEWELED_MASK']);s.setup([],draw=['DEMON_FORM','BARRICADE','INFLAME'])
    s.invoke('RELIC.JEWELED_MASK','BeforeHandDraw','$owner','$context','$combat')
    card=next(c for c in s.hand if c['cost']==0)
    assert card['cost_mods'][0]==('ThisTurn' if old else 'ThisCombat'),card

@case('RELIC.JUZU_BRACELET')
def juzu(eng,old):
    s=R(eng,relics=['JUZU_BRACELET']);s.setup(['DEFEND_IRONCLAD'])
    types=s.invoke('RELIC.JUZU_BRACELET','ModifyUnknownMapPointRoomTypes','$rooms')
    assert 'Monster' not in types and ('Elite' in types)==old

@case('RELIC.BYRDPIP')
def byrdpip(eng,old):
    s=R(eng,relics=['BYRDPIP']);s.setup(['BYRD_SWOOP'])
    for _ in range(6):s.invoke('RELIC.BYRDPIP','AfterCardPlayed','$context','$play')
    assert s.get('RELIC.BYRDPIP','SwoopsPlayed')==(0 if old else 6)
    assert s.invoke('RELIC.BYRDPIP','TryModifyRestSiteOptions','$owner','$options')==(not old)

@case('RELIC.RINGING_TRIANGLE')
def triangle(eng,old):
    s=R(eng,relics=['RINGING_TRIANGLE']);s.setup(['DEFEND_IRONCLAD'])
    assert s.invoke('RELIC.RINGING_TRIANGLE','ShouldFlush','$owner')==(not old)
    assert s.amount('POWER.RETAIN_HAND_POWER')==(0 if old else 1)

@case('RELIC.VELVET_CHOKER')
def choker(eng,old):
    s=R(eng,relics=['VELVET_CHOKER']);s.setup(['DEFEND_IRONCLAD'])
    s.invoke('$history','CardPlayStarted','$combat','$play')
    assert s.get('RELIC.VELVET_CHOKER','DisplayAmount')==(0 if old else 1)
    s.invoke('RELIC.VELVET_CHOKER','AfterCardPlayed','$context','$play')
    assert s.get('RELIC.VELVET_CHOKER','DisplayAmount')==1

@case('RELIC.BING_BONG')
def bing(eng,old):
    s=R(eng,relics=['BING_BONG']);s.setup(['ANGER'])
    n=len(s.r['obs']['deck']);s.dbg('move',subject='CARD.ANGER',pile='Deck')
    assert len(s.r['obs']['deck'])==n+2
    s.dbg('move',subject='$clone',pile='Hand');s.dbg('move',subject='$clone',pile='Deck')
    assert len(s.r['obs']['deck'])==n+(2 if old else 3)

@case('RELIC.BOOK_REPAIR_KNIFE')
def knife(eng,old):
    s=R(eng,'Necrobinder',relics=['BOOK_REPAIR_KNIFE']);s.setup(['DEFEND_NECROBINDER'])
    s.dbg('summon_ally')
    hp=s.hp;s.invoke('RELIC.BOOK_REPAIR_KNIFE','AfterDiedToDoom','$context','$allies')
    assert s.hp-hp==(3 if old else 0),(old,hp,s.hp,s.combat)

# Hook-order cases use a second game hook to observe the state while an effect is in flight.
for _id,_member,_value in [('BURNING_STICKS','WasUsedThisCombat',False),('PERMAFROST','ActivatedThisCombat',False),('RAINBOW_RING','ActivationCountThisTurn',0)]:
    def hook_order(eng,old,id=_id,member=_member,value=_value):
        s=R(eng,relics=[id]);s.setup(['DEFEND_IRONCLAD','DEMON_FORM','ANGER'])
        # dbg_watch records the actual field at CardPileCmd/CreatureCmd/PowerCmd entry during the hook.
        if id=='BURNING_STICKS':
            s.dbg('watch',subject='RELIC.'+id,member=member);s.invoke('RELIC.'+id,'AfterCardExhausted','$context','$card',False)
        elif id=='PERMAFROST':
            s.setup(['DEMON_FORM']);s.dbg('watch',subject='RELIC.'+id,member=member);s.invoke('RELIC.'+id,'AfterCardPlayed','$context','$play')
        else:
            s.field('RELIC.'+id,'AttacksPlayedThisTurn',1);s.field('RELIC.'+id,'SkillsPlayedThisTurn',1)
            s.setup(['DEMON_FORM']);s.dbg('watch',subject='RELIC.'+id,member=member);s.invoke('RELIC.'+id,'AfterCardPlayed','$context','$play')
        assert s.state()['watch_value']==(value if old else (1 if id=='RAINBOW_RING' else True)),s.state()
    TESTS['RELIC.'+_id]=hook_order

@case('RELIC.JOSS_PAPER')
def joss(eng,old):
    s=R(eng,relics=['JOSS_PAPER']);s.setup(['DEFEND_IRONCLAD'],draw=['STRIKE_IRONCLAD'])
    s.field('RELIC.JOSS_PAPER','CardsExhausted',4)
    s.dbg('watch',subject='RELIC.JOSS_PAPER',member='CardsExhausted')
    s.invoke('RELIC.JOSS_PAPER','AfterCardExhausted','$context','$card',False)
    assert s.state()['watch_value']==(5 if old else 0),s.state()
    assert any(c['id']=='CARD.STRIKE_IRONCLAD' for c in s.hand)
    assert s.get('RELIC.JOSS_PAPER','CardsExhausted')==0

@case('RELIC.POWER_CELL')
def cell(eng,old):
    s=R(eng,relics=['POWER_CELL']);s.setup([],draw=[dict(id='BACKSTAB'),'ANGER','CLASH','SLICE'])
    s.invoke('RELIC.POWER_CELL','BeforeSideTurnStart','$context','Player','$participants','$combat')
    assert [c['id'] for c in s.hand]==(['CARD.BACKSTAB','CARD.SLICE'] if old else ['CARD.SLICE','CARD.CLASH']),(old,s.hand,s.state())

@case('RELIC.PAELS_LEGION')
def legion(eng,old):
    s=R(eng,relics=['PAELS_LEGION']);s.setup(['DEFEND_IRONCLAD'])
    s.invoke('RELIC.PAELS_LEGION','AfterModifyingBlockAmount',5,'$card','$play')
    val=s.invoke('RELIC.PAELS_LEGION','ModifyBlockMultiplicative','$player',5,'Move','$card','$play')
    assert val==(2 if old else 1),val

@case('RELIC.PAELS_EYE')
def eye(eng,old):
    s=R(eng,relics=['PAELS_EYE']);s.setup(['DEFEND_IRONCLAD'])
    s.field('RELIC.PAELS_EYE','WasOwnerPartOfLastPlayerTurn',True)
    assert s.invoke('RELIC.PAELS_EYE','ShouldTakeExtraTurnEarly','$owner')==(not old)
    assert s.invoke('RELIC.PAELS_EYE','ShouldTakeExtraTurn','$owner')==old

# Powers: direct real hook calls in a live combat isolate conditions hard to reach with a random policy.
@case('POWER.DOOM_POWER')
def doom(eng,old):
    s=R(eng);s.setup(['DEFEND_IRONCLAD']);s.power('DOOM_POWER',target='enemy')
    s.invoke('$enemy','SetCurrentHpInternal',0)
    count=len(s.combat['enemies'])
    s.invoke('POWER.DOOM_POWER','DoomKill','$dead_enemy')
    assert len(s.combat['enemies'])==count-(1 if old else 0),s.combat

@case('POWER.BUFFER_POWER')
def buffer(eng,old):
    s=R(eng);s.setup(['BRIGHTEST_FLAME']);s.power('BUFFER_POWER')
    val=s.invoke('POWER.BUFFER_POWER','ModifyHpLostAfterOstyLate','$player',5,'Unpowered',None,'$card')
    assert val==(0 if old else 5)

@case('POWER.ONE_FOR_ALL_POWER')
def oneforall(eng,old):
    s=R(eng);s.setup(['BLUDGEON']);s.power('ONE_FOR_ALL_POWER',5)
    val=s.invoke('POWER.ONE_FOR_ALL_POWER','ModifyDamageAdditive','$enemy',10,'Move','$player','$card','$play',auto=True,energy_value=3)
    assert val==(5 if old else 0)

@case('POWER.PHANTOM_BLADES_POWER')
def phantom(eng,old):
    s=R(eng,'Silent');s.setup(['SHIV']);s.power('PHANTOM_BLADES_POWER',8)
    assert 'Retain' in s.card('CARD.SHIV')['kw']
    s.dbg('remove_power',subject='POWER.PHANTOM_BLADES_POWER')
    assert ('Retain' in s.card('CARD.SHIV')['kw'])==old

@case('POWER.RITUAL_POWER')
def ritual_power(eng,old):
    s=R(eng);s.setup(['DEFEND_IRONCLAD']);s.power('RITUAL_POWER',2,target='enemy',applier='enemy')
    assert s.get('POWER.RITUAL_POWER','WasJustAppliedByEnemy')==old

@case('POWER.SIC_EM_POWER')
def sicem(eng,old):
    s=R(eng,'Necrobinder');s.setup(['SIC_EM']);s.power('SIC_EM_POWER',2,target='enemy')
    assert s.get('POWER.SIC_EM_POWER','InstanceType')==('None' if old else 'InstancedPerApplier')

@case('POWER.THORNS_POWER')
def thorns(eng,old):
    s=R(eng);s.setup(['OMNISLICE']);s.power('THORNS_POWER',3,target='enemy')
    hp=s.hp;s.invoke('POWER.THORNS_POWER','BeforeDamageReceived','$context','$enemy',8,'Unpowered, Move','$player','$card')
    assert hp-s.hp==(3 if old else 0)

@case('POWER.NIGHTMARE_POWER')
def nightmare(eng,old):
    s=R(eng,'Silent');s.setup(['STRIKE_SILENT'])
    s.power('NIGHTMARE_POWER',1)
    s.invoke('$card','SetToFreeThisTurn')
    s.invoke('POWER.NIGHTMARE_POWER','SetSelectedCard','$card')
    s.invoke('POWER.NIGHTMARE_POWER','BeforeHandDraw','$owner','$context','$combat')
    # EndOfTurnCleanup clears the temporary zero cost on the stored clone in the new definition.
    clones=[c for c in s.hand if c['id']=='CARD.STRIKE_SILENT']
    assert sorted(c['cost'] for c in clones)==([0,0] if old else [0,1]),clones

@case('POWER.JUGGLING_POWER')
def juggling(eng,old):
    s=R(eng,'Silent');s.setup(['ANGER']);s.power('JUGGLING_POWER',1)
    for _ in range(3):s.invoke('POWER.JUGGLING_POWER','BeforeCardPlayed','$play')
    assert len(s.hand)==(2 if old else 1)
    s.invoke('POWER.JUGGLING_POWER','AfterCardPlayed','$context','$play')
    assert len(s.hand)==2

@case('POWER.LETHALITY_POWER')
def lethality(eng,old):
    s=R(eng);s.setup(['STRIKE_IRONCLAD']);s.power('LETHALITY_POWER',50)
    s.dbg('move',subject='CARD.STRIKE_IRONCLAD',pile='Play')
    s.invoke('$history','CardPlayStarted','$combat','$play')
    val=s.invoke('POWER.LETHALITY_POWER','ModifyDamageMultiplicative','$enemy',6,'Move','$player','$card',None)
    assert val==(1.5 if old else 1),val

@case('POWER.DRAW_CARDS_NEXT_TURN_POWER')
def drawnext(eng,old):
    s=R(eng);s.setup(['DEFEND_IRONCLAD']);s.power('DRAW_CARDS_NEXT_TURN_POWER',2)
    s.invoke('POWER.DRAW_CARDS_NEXT_TURN_POWER','AfterModifyingHandDraw')
    assert s.amount('POWER.DRAW_CARDS_NEXT_TURN_POWER')==(2 if old else 0)

@case('POWER.SHRINK_POWER')
def shrink(eng,old):
    s=R(eng);s.setup(['DEFEND_IRONCLAD']);s.power('SHRINK_POWER',-1,target='enemy')
    s.invoke('POWER.SHRINK_POWER','AfterDeath','$context','$player',None,False,0.0)
    assert s.amount('POWER.SHRINK_POWER','enemy')==(0 if old else -1)

for _id in ['COVERED_POWER','INTERCEPT_POWER','TANK_POWER']:
    def remove_guard(eng,old,id=_id):
        s=R(eng);s.setup(['DEFEND_IRONCLAD'])
        if id=='TANK_POWER':
            s.dbg('guard_team')
            s.dbg('remove_power',subject='POWER.'+id)
            # Real Tank only cleans teammates; explicit owner-side setup is provided by dbg_team.
            assert (s.amount('POWER.GUARDED_POWER')>0)==old,s.combat
        else:
            s.power('COVERED_POWER')
            subject='POWER.'+id;s.dbg('remove_power',subject=subject)
            remaining='POWER.INTERCEPT_POWER' if id=='COVERED_POWER' else 'POWER.COVERED_POWER'
            assert (s.amount(remaining)>0)==old,s.combat
    TESTS['POWER.'+_id]=remove_guard

for _id,_hook,_args,_a,_b in [
 ('ENTROPY_POWER','AfterPlayerTurnStart',('$context','$owner'),'CombatCardSelection','CombatCardGeneration'),
 ('STAMPEDE_POWER','AfterAutoPostPlayPhaseEntered',('$context','$owner'),'Shuffle','CombatCardSelection')]:
    def rng_power(eng,old,id=_id,hook=_hook,args=_args,a=_a,b=_b):
        s=R(eng);s.setup(['STRIKE_IRONCLAD','BASH','ANGER']);s.power(id,1)
        before=s.state();s.invoke('POWER.'+id,hook,*args);after=s.state()
        stream=a if old else b
        assert after['run_rng'][stream]>before['run_rng'][stream],(before,after)
    TESTS['POWER.'+_id]=rng_power

@case('POWER.IMITATION_LEARNING_POWER')
def imitation(eng,old):
    s=R(eng,relics=['VELVET_CHOKER']);s.setup(['DEMON_FORM']);s.power('IMITATION_LEARNING_POWER',2)
    s.field('POWER.IMITATION_LEARNING_POWER','PlayerTarget','$owner')
    for _ in range(6):s.invoke('$history','CardPlayStarted','$combat','$play')
    s.invoke('POWER.IMITATION_LEARNING_POWER','BeforeCardPlayed','$play')
    s.invoke('POWER.IMITATION_LEARNING_POWER','AfterCardPlayed','$context','$play')
    assert s.amount('POWER.IMITATION_LEARNING_POWER')==(1 if old else 2)

for _id,_enc,_member,_expected in [('AXEBOT','AXEBOTS_NORMAL','MaxInitialHp',78),('TUNNELER','TUNNELER_WEAK','FromBelowDamage',23)]:
    def monster(eng,old,id=_id,enc=_enc,member=_member,expected=_expected):
        s=R(eng,encounter=enc);s.setup(['DEFEND_IRONCLAD'])
        val=s.get('MONSTER.'+id,member)
        assert val==(expected if old else (70 if id=='AXEBOT' else 24)),val
    TESTS['MONSTER.'+_id]=monster

for _id in ['DOORS_OF_LIGHT_AND_DARK','WATERLOGGED_SCRIPTORIUM','ZEN_WEAVER']:
    def event(eng,old,id=_id):
        s=R(eng);s.send(dict(cmd='set_player',gold=1000))
        before=s.r['obs'];s.send(dict(cmd='rl_enter_room',type='Event',event=id))
        if id=='WATERLOGGED_SCRIPTORIUM':
            assert any((99 if old else 120) in c.get('v',{}).values() for c in s.r['cands']),s.r['cands']
        else:
            s.act(0);s.resolve();after=s.r['obs']
            if id=='ZEN_WEAVER':
                assert before['g']['gold']-after['g']['gold']==(50 if old else 25)
                assert sum(c['id']=='CARD.ENLIGHTENMENT' for c in after['deck'])==(2 if old else 1)
            else:assert sum(c['up']>0 for c in after['deck'])==(2 if old else 1)
    TESTS['EVENT.'+_id]=event

def worker(id):
    start=time.monotonic()
    try:
        for old in (False,True):
            if old:os.environ['SLS2_REVERT']=id
            else:os.environ.pop('SLS2_REVERT',None)
            eng=Engine()
            try:TESTS[id](eng,old)
            finally:eng.close()
        return id,True,round(time.monotonic()-start,2),''
    except Exception:return id,False,round(time.monotonic()-start,2),traceback.format_exc()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('ids',nargs='*');ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    started=time.monotonic();ids=a.ids or sorted(TESTS)
    with cf.ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        results=list(pool.map(worker,ids))
    for id,ok,seconds,error in results:
        print(('PASS' if ok else 'FAIL'),id,f'{seconds}s',error,flush=True)
    failed=sum(not r[1] for r in results)
    print(f'{len(results)-failed}/{len(results)} models passed (off + individual ID), {time.monotonic()-started:.1f}s')
    raise SystemExit(bool(failed))

if __name__=='__main__':main()
